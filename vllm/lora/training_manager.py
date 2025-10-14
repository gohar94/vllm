# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TrainingManager for managing LoRA adapters during training.

This module provides functionality to create and manage LoRA adapters
specifically for training purposes, including:
- Initializing trainable LoRA parameters
- Freezing base model parameters
- Managing gradient flow for LoRA training
"""

import json
import os
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora import LoRALayerWeights
from vllm.lora.models import LoRAModel
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.model_executor.models import SupportsLoRA

logger = init_logger(__name__)

# TODO(girfan): Make these configurable.
RANK = 8
ALPHA = 16


class TrainingManager:
    """Manages LoRA adapter training, including parameter initialization and gradient control."""

    LoRA_PATH = "./llama3_dummy_lora"

    def __init__(
        self,
        model_runner,  # Changed from model: SupportsLoRA to model_runner to get wrapped model
        lora_manager: WorkerLoRAManager,
        lora_config: LoRAConfig,
        device: torch.device,
        dtype: torch.dtype,
        sub_modules: Optional[list[str]] = None,
        rank: int = RANK,
        alpha: int = ALPHA,
        target_modules: Optional[List[str]] = None,
    ):
        """Initialize the Training Manager.
        
        Args:
            model_runner: The model runner (to access wrapped/unwrapped model)
            lora_manager: Worker LoRA manager for handling LoRA operations
            lora_config: LoRA configuration
            device: Device to place LoRA weights on
            dtype: Data type for LoRA weights
            sub_modules: Specific submodules to apply LoRA to (if None, use target_modules)
            rank: LoRA rank (dimension of low-rank matrices)
            alpha: LoRA alpha (scaling factor)
            target_modules: List of module name patterns to apply LoRA (e.g., ["q_proj", "v_proj"])
        """
        self.model_runner = model_runner
        self.lora_manager = lora_manager
        self.lora_config = lora_config
        self.device = device
        self.dtype = dtype
        self.rank = rank
        self.alpha = alpha
        self._next_lora_id = 1

        # Track which modules have LoRA applied
        self.lora_modules: Set[str] = set()
        self.trainable_lora_params: Dict[str, nn.Parameter] = {}

        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.current_lora_id: Optional[int] = None
        self.training_step: int = 0
        self.gradient_accumulation_steps: int = 1
        self.trained_lora_ids: set = set()  # Track which LoRA IDs have been trained
        self.gradient_accumulation_counter: int = 0

        # Determine target modules
        if sub_modules:
            self.sub_modules = sub_modules
        elif target_modules:
            self.sub_modules = self._find_target_modules(target_modules)
        else:
            # Default: apply to attention projection layers
            self.sub_modules = self._find_target_modules(
                ["q_proj", "k_proj", "v_proj", "o_proj"])

        logger.info(
            f"[TrainingManager] Initialized with {len(self.sub_modules)} target modules"
        )
        logger.info(
            f"[TrainingManager] LoRA rank={self.rank}, alpha={self.alpha}")
        
        # Register this TrainingManager with the LoRAModelManager so it can avoid
        # resetting stacked tensors for modules with trained parameters
        logger.info(f"[TrainingManager] Checking lora_manager attributes: {dir(self.lora_manager)}")
        if hasattr(self.lora_manager, '_adapter_manager'):
            logger.info("[TrainingManager] Found _adapter_manager, registering...")
            self.lora_manager._adapter_manager._training_manager = self
            logger.info("[TrainingManager] Registered with LoRAModelManager")
        else:
            logger.warning("[TrainingManager] No _adapter_manager found in lora_manager")

    @property 
    def target_modules(self) -> set[str]:
        """Get the set of target module names that have trained parameters."""
        return set(self.sub_modules.keys())

    @property
    def model(self):
        """Get the actual model, unwrapping if needed.
        
        The model_runner.model may be wrapped in UBatchWrapper. We need to unwrap it
        to access the actual model with LoRA layers.
        """
        logger.info(f"[TRACE] TrainingManager.model property called")
        logger.info(f"[TRACE] model_runner type: {type(self.model_runner).__name__}, id: {id(self.model_runner)}")
        
        if hasattr(self.model_runner, 'model'):
            model = self.model_runner.model
            logger.info(f"[TRACE] model_runner.model type: {type(model).__name__}, id: {id(model)}")
            
            # Unwrap if it's a UBatchWrapper or similar
            if hasattr(model, 'unwrap'):
                unwrapped = model.unwrap()
                logger.info(f"[TRACE] Unwrapped model from {type(model).__name__} to {type(unwrapped).__name__}, id: {id(unwrapped)}")
                return unwrapped
            logger.info(f"[TRACE] No unwrapping needed, returning model id: {id(model)}")
            return model
        # Fallback: model_runner might be the model itself
        logger.info(f"[TRACE] Fallback: returning model_runner as model, id: {id(self.model_runner)}")
        return self.model_runner

    def _find_target_modules(self, target_patterns: List[str]) -> List[str]:
        """Find all module names that match the target patterns.
        
        Args:
            target_patterns: List of patterns to match (e.g., ["q_proj", "v_proj"])
            
        Returns:
            List of full module names that match any pattern
        """
        target_modules = []
        for name, module in self.model.named_modules():
            # Check if this module matches any target pattern
            if any(pattern in name for pattern in target_patterns):
                # Only include Linear layers and their LoRA variants
                if isinstance(module, (nn.Linear, BaseLayerWithLoRA)):
                    target_modules.append(name)
                    logger.debug(
                        f"[TrainingManager] Found target module: {name}")

        logger.info(
            f"[TrainingManager] Found {len(target_modules)} modules matching patterns {target_patterns}"
        )
        return target_modules

    def create_lora(
        self,
        model: nn.Module,
        sub_modules: list[str],
        device: torch.device,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
    ) -> LoRAModel:
        """Create LoRA adapters for the specified submodules.
        
        Args:
            model: The model to create LoRA adapters for
            sub_modules: List of module names to apply LoRA to
            device: Device to place LoRA weights on
            rank: LoRA rank (uses self.rank if None)
            alpha: LoRA alpha (uses self.alpha if None)
            
        Returns:
            LoRAModel containing the created LoRA adapters
        """
        if rank is None:
            rank = self.rank
        if alpha is None:
            alpha = self.alpha

        loras: dict[str, LoRALayerWeights] = {}

        for name in sub_modules:
            try:
                module = model.get_submodule(name)

                # Get weight tensor from module
                if hasattr(module, 'weight'):
                    w = module.weight
                elif isinstance(module, BaseLayerWithLoRA):
                    # For LoRA layers, get base layer weight
                    w = module.base_layer.weight
                else:
                    logger.warning(
                        f"[TrainingManager] Module {name} has no weight, skipping"
                    )
                    continue

                # Initialize LoRA weights following PEFT's default initialization:
                # - lora_a (Matrix A): Kaiming uniform (same as nn.Linear)
                # - lora_b (Matrix B): zeros
                # This ensures ΔW = B @ A = 0 initially (zero-init principle)

                # LoRA A: [input_dim, rank]
                lora_a = torch.empty(
                    [w.shape[1], rank],
                    dtype=self.dtype,
                    device=device,
                    requires_grad=True  # CRITICAL: Enable gradients for training
                )
                torch.nn.init.kaiming_uniform_(lora_a, a=5**0.5)

                # LoRA B: [rank, output_dim]
                lora_b = torch.zeros(
                    [rank, w.shape[0]],
                    dtype=self.dtype,
                    device=device,
                    requires_grad=True  # CRITICAL: Enable gradients for training
                )

                loras[name] = LoRALayerWeights(
                    name,
                    rank,
                    alpha,
                    lora_a,
                    lora_b,
                )

                # Track this as a LoRA module
                self.lora_modules.add(name)

                logger.info(
                    f"[TrainingManager] Created LoRA for {name}: "
                    f"A={lora_a.shape}, B={lora_b.shape}, "
                    f"A_grad={lora_a.requires_grad}, B_grad={lora_b.requires_grad}"
                )

            except Exception as e:
                logger.error(
                    f"[TrainingManager] Failed to create LoRA for {name}: {e}")
                continue

        lora_id = self._next_lora_id
        self._next_lora_id += 1

        lora_model = LoRAModel(lora_id, rank, loras)

        logger.info(f"[TrainingManager] Created LoRAModel (id={lora_id}) with "
                    f"{len(loras)} adapters, rank={rank}, alpha={alpha}")

        return lora_model

    def freeze_base_model(self, verbose: bool = True) -> Dict[str, int]:
        """Freeze all base model parameters (non-LoRA).
        
        This ensures only LoRA parameters receive gradients during training.
        
        NOTE: When LoRA is loaded from disk, the LoRA parameters are managed
        separately by vLLM's LoRA system and are NOT in model.named_parameters().
        This function primarily freezes the base model. For training loaded LoRA
        adapters, you need to access them through the LoRA manager.
        
        Args:
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary with counts of frozen/trainable parameters
        """
        frozen_count = 0
        total_count = 0
        trainable_count = 0

        # Debug: First, let's see what parameters we have
        logger.debug("[TrainingManager] Scanning model parameters...")
        logger.debug("[TrainingManager] Sample parameter names (first 10):")
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i < 10:
                logger.debug(
                    f"[TrainingManager]   {i}: {name} (shape={param.shape})")

        for name, param in self.model.named_parameters():
            total_count += 1

            # Check if this is a LoRA parameter
            # LoRA parameters typically have 'lora_a' or 'lora_b' in their name
            # or 'lora_a_stacked'/'lora_b_stacked' for vLLM's internal LoRA layers
            is_lora_param = any([
                'lora_a' in name.lower(), 'lora_b' in name.lower(),
                'lora' in name.lower() and 'stacked' in name.lower(),
                'lora' in name.lower()
                and ('weight' in name.lower() or 'bias' in name.lower())
            ])

            if not is_lora_param:
                param.requires_grad = False
                frozen_count += 1
                if verbose:
                    logger.debug(f"[TrainingManager] Frozen: {name}")
            else:
                param.requires_grad = True
                trainable_count += 1
                self.trainable_lora_params[name] = param
                if verbose:
                    logger.debug(
                        f"[TrainingManager] Trainable: {name} (shape={param.shape})"
                    )

        # Check if we found any trainable LoRA parameters in model.named_parameters()
        # Note: Loaded LoRA adapters are managed separately by LoRAModelManager and won't appear here
        if trainable_count == 0 and hasattr(
                self, 'lora_manager') and self.lora_manager:
            logger.debug(
                "[TrainingManager] No LoRA parameters found in model.named_parameters(). "
                "This is EXPECTED for loaded LoRA adapters - they are managed separately by LoRAModelManager. "
                "LoRA parameters were already converted to trainable Parameters via make_lora_trainable()."
            )

        stats = {
            'total': total_count,
            'frozen': frozen_count,
            'trainable': trainable_count,
        }

        logger.debug(f"[TrainingManager] Parameter status: "
                    f"{stats['frozen']}/{stats['total']} frozen, "
                    f"{stats['trainable']}/{stats['total']} trainable "
                    f"({100 * stats['trainable'] / stats['total']:.2f}%)")

        return stats

    def make_lora_trainable(
        self,
        lora_id: int,
        learning_rate: float = 1e-4,
        num_training_steps: Optional[int] = None,
        num_warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        scheduler_type: str = "cosine",
    ) -> Dict[str, int]:
        """Convert a loaded LoRA adapter to trainable Parameters and setup optimizer.
        
        This method takes a LoRA adapter that was loaded from disk (stored as
        tensors in LoRAModelManager) and converts its weights to nn.Parameters
        with requires_grad=True, making them trainable. It also automatically
        sets up the optimizer and learning rate scheduler for training.
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter
            learning_rate: Learning rate for optimizer (default: 1e-4)
            num_training_steps: Total number of training steps for scheduler
            num_warmup_steps: Number of warmup steps (default: 0)
            gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)
            weight_decay: Weight decay for optimizer (default: 0.0)
            scheduler_type: Type of scheduler ("cosine" or "linear", default: "cosine")
            
        Returns:
            Dictionary with statistics about converted parameters
        """
        # DEBUG: Log when this function is called
        import traceback
        logger.info(f"[DEBUG] make_lora_trainable called for LoRA {lora_id}")
        logger.info(f"[DEBUG] make_lora_trainable stack trace:\n{''.join(traceback.format_stack()[-3:])}")
        
        if not self.lora_manager:
            raise ValueError("LoRA manager not available")

        # Check if this LoRA is already set up for training
        logger.info(f"[DEBUG] Checking if LoRA {lora_id} already trainable: current_lora_id={self.current_lora_id}, optimizer={self.optimizer is not None}")
        if self.current_lora_id == lora_id and self.optimizer is not None:
            logger.info(
                f"[TrainingManager] LoRA {lora_id} already trainable, updating configuration..."
            )
            # Update configuration if parameters are different
            if self.gradient_accumulation_steps != gradient_accumulation_steps:
                logger.warning(
                    f"[TrainingManager] Updating gradient_accumulation_steps from "
                    f"{self.gradient_accumulation_steps} to {gradient_accumulation_steps}"
                )
                self.gradient_accumulation_steps = gradient_accumulation_steps
            
            # Return existing stats
            return {
                'lora_id': lora_id,
                'already_trainable': True,
                'optimizer_setup': True,
                'scheduler_setup': self.scheduler is not None,
                'learning_rate': learning_rate,
                'gradient_accumulation_steps': gradient_accumulation_steps,
            }

        # Get the loaded LoRA model from the manager
        # Note: lora_manager.list_adapters() returns a set of IDs
        # We need to access _adapter_manager to get the actual LoRAModel
        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            raise ValueError(
                f"LoRA adapter {lora_id} not found in LoRA manager")

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            raise ValueError(f"LoRA adapter {lora_id} could not be retrieved")

        trainable_count = 0
        total_tensors = 0

        logger.info(
            f"[TrainingManager] Converting LoRA {lora_id} to trainable Parameters..."
        )

        # Convert each LoRA weight tensor to a Parameter
        for module_name, lora_weights in lora_model.loras.items():
            # Check if this is a packed LoRA (lora_a and lora_b are lists)
            is_packed = lora_weights.is_packed

            if is_packed:
                # PackedLoRALayerWeights: lora_a and lora_b are lists of tensors
                logger.debug(
                    f"[TrainingManager] Processing packed LoRA: {module_name}")

                # Convert each tensor in the lora_a list
                if lora_weights.lora_a is not None:
                    for i, tensor in enumerate(lora_weights.lora_a):
                        if tensor is not None:
                            total_tensors += 1
                            if not isinstance(tensor, nn.Parameter):
                                # Convert to Parameter with requires_grad=True
                                lora_weights.lora_a[i] = nn.Parameter(
                                    tensor.clone().detach(),
                                    requires_grad=True)
                                trainable_count += 1
                                logger.debug(
                                    f"[TrainingManager] Converted {module_name}.lora_a[{i}] to Parameter "
                                    f"(shape={lora_weights.lora_a[i].shape})")
                            else:
                                lora_weights.lora_a[i].requires_grad = True
                                trainable_count += 1

                # Convert each tensor in the lora_b list
                if lora_weights.lora_b is not None:
                    for i, tensor in enumerate(lora_weights.lora_b):
                        if tensor is not None:
                            total_tensors += 1
                            if not isinstance(tensor, nn.Parameter):
                                # Convert to Parameter with requires_grad=True
                                lora_weights.lora_b[i] = nn.Parameter(
                                    tensor.clone().detach(),
                                    requires_grad=True)
                                trainable_count += 1
                                logger.debug(
                                    f"[TrainingManager] Converted {module_name}.lora_b[{i}] to Parameter "
                                    f"(shape={lora_weights.lora_b[i].shape})")
                            else:
                                lora_weights.lora_b[i].requires_grad = True
                                trainable_count += 1
            else:
                # Regular LoRALayerWeights: lora_a and lora_b are single tensors
                total_tensors += 2  # lora_a and lora_b

                # Convert lora_a to Parameter
                if lora_weights.lora_a is not None:
                    # CRITICAL FIX: Enable gradients IN-PLACE without creating new objects!
                    # Stacked tensors reference this same tensor, so we can't replace it
                    if not lora_weights.lora_a.requires_grad:
                        lora_weights.lora_a.requires_grad_(True)
                        trainable_count += 1
                        logger.debug(
                            f"[TrainingManager] Converted {module_name}.lora_a to Parameter "
                            f"(shape={lora_weights.lora_a.shape})")
                    else:
                        lora_weights.lora_a.requires_grad = True
                        trainable_count += 1

                # Convert lora_b to Parameter
                if lora_weights.lora_b is not None:
                    # CRITICAL FIX: Enable gradients IN-PLACE without creating new objects!
                    # Stacked tensors reference this same tensor, so we can't replace it
                    if not lora_weights.lora_b.requires_grad:
                        lora_weights.lora_b.requires_grad_(True)
                        trainable_count += 1
                        logger.debug(
                            f"[TrainingManager] Converted {module_name}.lora_b to Parameter "
                            f"(shape={lora_weights.lora_b.shape})")
                    else:
                        lora_weights.lora_b.requires_grad = True
                        trainable_count += 1

        stats = {
            'lora_id': lora_id,
            'total_tensors': total_tensors,
            'trainable_params': trainable_count,
        }

        logger.info(
            f"[TrainingManager] LoRA {lora_id} conversion complete: "
            f"{trainable_count}/{total_tensors} tensors converted to trainable Parameters"
        )

        # Debug: Print all converted layer names
        logger.info(
            f"[TrainingManager] Converted LoRA layers for adapter {lora_id}:")
        for module_name, lora_weights in lora_model.loras.items():
            is_packed = lora_weights.is_packed
            if is_packed:
                num_components = len(
                    [t for t in lora_weights.lora_a
                     if t is not None]) if lora_weights.lora_a else 0
                logger.info(
                    f"[TrainingManager]   - {module_name} (PACKED, {num_components} components)"
                )
            else:
                logger.info(f"[TrainingManager]   - {module_name} (regular)")

        # Verify that parameters have requires_grad=True
        logger.info(
            f"[TrainingManager] Verifying gradient settings for adapter {lora_id}:"
        )
        grad_enabled_count = 0
        for module_name, lora_weights in lora_model.loras.items():
            if lora_weights.is_packed:
                # Check packed LoRA
                for i, tensor in enumerate(lora_weights.lora_a or []):
                    if tensor is not None and tensor.requires_grad:
                        grad_enabled_count += 1
                for i, tensor in enumerate(lora_weights.lora_b or []):
                    if tensor is not None and tensor.requires_grad:
                        grad_enabled_count += 1
            else:
                # Check regular LoRA
                if lora_weights.lora_a is not None and lora_weights.lora_a.requires_grad:
                    grad_enabled_count += 1
                if lora_weights.lora_b is not None and lora_weights.lora_b.requires_grad:
                    grad_enabled_count += 1

        logger.info(
            f"[TrainingManager] ✓ {grad_enabled_count}/{trainable_count} LoRA tensors have requires_grad=True"
        )

        # CRITICAL: Enable gradients on stacked tensors
        # The Parameters have requires_grad=True, but the stacked tensors (used in forward pass)
        # also need requires_grad=True for autograd to work
        logger.info(f"[TrainingManager] Enabling gradients on stacked tensors...")
        stacked_stats = self.enable_stacked_tensor_gradients(lora_id)
        logger.info(f"[TrainingManager] ✓ Enabled gradients on {stacked_stats['enabled']} stacked tensors")

        # CRITICAL FIX: Copy initial Parameter values to stacked tensors
        # When we converted tensors to Parameters above, we created new objects.
        # The stacked tensors still have the old values from activation.
        # We need to copy the Parameter values (which are the same, but as Parameters)
        # to the stacked tensors so inference uses the correct values.
        logger.info(f"[TrainingManager] Copying initial Parameter values to stacked tensors...")
        
        # DEBUG: Check Parameter checksums BEFORE copy
        param_checksums_before = {}
        for module_name, lora_weights in lora_model.loras.items():
            if lora_weights.is_packed and lora_weights.lora_a:
                for i, param in enumerate(lora_weights.lora_a):
                    if param is not None:
                        param_checksums_before[f"{module_name}.lora_a[{i}]"] = param.sum().item()
                        if i == 0:  # Log first param
                            logger.info(f"[TrainingManager] DEBUG: Parameter {module_name}.lora_a[0] checksum BEFORE copy: {param.sum().item():.6f}")
                        break
            elif lora_weights.lora_a is not None:
                param_checksums_before[f"{module_name}.lora_a"] = lora_weights.lora_a.sum().item()
                logger.info(f"[TrainingManager] DEBUG: Parameter {module_name}.lora_a checksum BEFORE copy: {lora_weights.lora_a.sum().item():.6f}")
                break
        
        initial_copy_stats = self.copy_parameters_to_stacked(lora_id)
        logger.info(f"[TrainingManager] ✓ Initial sync complete: {initial_copy_stats}")

        # Store current LoRA ID and gradient accumulation config
        self.current_lora_id = lora_id
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_counter = 0
        
        # Mark this LoRA as trained
        self.trained_lora_ids.add(lora_id)
        logger.info(f"[TrainingManager] Marked LoRA {lora_id} as trained. Trained LoRAs: {self.trained_lora_ids}")

        # Automatically setup optimizer
        logger.info(f"[TrainingManager] Setting up optimizer for LoRA {lora_id}...")
        self.optimizer = self.setup_optimizer(
            lora_id=lora_id,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup scheduler if training steps provided
        if num_training_steps is not None and num_training_steps > 0:
            logger.info(f"[TrainingManager] Setting up {scheduler_type} scheduler...")
            self.scheduler = self.setup_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                scheduler_type=scheduler_type,
            )
        else:
            self.scheduler = None
            logger.info("[TrainingManager] No scheduler setup (num_training_steps not provided)")

        stats['optimizer_setup'] = True
        stats['scheduler_setup'] = self.scheduler is not None
        stats['learning_rate'] = learning_rate
        stats['gradient_accumulation_steps'] = gradient_accumulation_steps

        return stats

    def enable_lora_gradients(self, lora_model: LoRAModel) -> None:
        """Ensure all LoRA parameters have gradients enabled.
        
        Args:
            lora_model: The LoRA model to enable gradients for
        """
        for module_name, lora_weights in lora_model.loras.items():
            if lora_weights.lora_a is not None:
                lora_weights.lora_a.requires_grad = True
                logger.debug(
                    f"[TrainingManager] Enabled gradients for {module_name}.lora_a "
                    f"(shape={lora_weights.lora_a.shape})")

            if lora_weights.lora_b is not None:
                lora_weights.lora_b.requires_grad = True
                logger.debug(
                    f"[TrainingManager] Enabled gradients for {module_name}.lora_b "
                    f"(shape={lora_weights.lora_b.shape})")

    def collect_lora_gradients(self,
                               lora_id: Optional[int] = None
                               ) -> Dict[str, torch.Tensor]:
        """Collect gradients from LoRA parameters.
        
        Args:
            lora_id: If provided, collect gradients from this specific loaded LoRA.
                    If None, collect from model.named_parameters() (created LoRAs).
        
        Returns:
            Dictionary mapping parameter names to their gradients
        """
        gradients = {}

        if lora_id is not None:
            # Collect gradients from loaded LoRA adapter
            if not self.lora_manager:
                logger.warning("[TrainingManager] LoRA manager not available")
                return gradients

            lora_adapters = self.lora_manager.list_adapters()
            if lora_id not in lora_adapters:
                logger.warning(f"[TrainingManager] LoRA {lora_id} not found")
                return gradients

            lora_model = self.lora_manager._adapter_manager.get_adapter(
                lora_id)
            if lora_model is None:
                logger.warning(
                    f"[TrainingManager] LoRA {lora_id} could not be retrieved")
                return gradients

            # Collect gradients from LoRA weights
            for module_name, lora_weights in lora_model.loras.items():
                is_packed = lora_weights.is_packed

                if is_packed:
                    # PackedLoRALayerWeights: collect from lists
                    if lora_weights.lora_a is not None:
                        for i, tensor in enumerate(lora_weights.lora_a):
                            if tensor is not None and tensor.grad is not None:
                                gradients[
                                    f"{module_name}.lora_a[{i}]"] = tensor.grad.clone(
                                    )

                    if lora_weights.lora_b is not None:
                        for i, tensor in enumerate(lora_weights.lora_b):
                            if tensor is not None and tensor.grad is not None:
                                gradients[
                                    f"{module_name}.lora_b[{i}]"] = tensor.grad.clone(
                                    )
                else:
                    # Regular LoRALayerWeights: collect from single tensors
                    if lora_weights.lora_a is not None and lora_weights.lora_a.grad is not None:
                        gradients[
                            f"{module_name}.lora_a"] = lora_weights.lora_a.grad.clone(
                            )

                    if lora_weights.lora_b is not None and lora_weights.lora_b.grad is not None:
                        gradients[
                            f"{module_name}.lora_b"] = lora_weights.lora_b.grad.clone(
                            )
        else:
            # Collect gradients from model parameters (created LoRAs)
            for name, param in self.model.named_parameters():
                if param.grad is not None and 'lora' in name.lower():
                    gradients[name] = param.grad.clone()

        logger.info(
            f"[TrainingManager] Collected {len(gradients)} LoRA gradients")

        return gradients

    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        """Get all trainable LoRA parameters.
        
        Returns:
            Dictionary of trainable parameter names to parameters
        """
        trainable = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                trainable[name] = param

        return trainable

    def get_lora_parameters(self, lora_id: int) -> list[nn.Parameter]:
        """Get all trainable parameters for a specific loaded LoRA adapter.
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter
            
        Returns:
            List of trainable Parameters for this LoRA adapter
        """
        if not self.lora_manager:
            logger.warning("[TrainingManager] LoRA manager not available")
            return []

        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            logger.warning(f"[TrainingManager] LoRA {lora_id} not found")
            return []

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            logger.warning(
                f"[TrainingManager] LoRA {lora_id} could not be retrieved")
            return []

        parameters = []
        for module_name, lora_weights in lora_model.loras.items():
            is_packed = lora_weights.is_packed

            if is_packed:
                # PackedLoRALayerWeights: collect from lists
                if lora_weights.lora_a is not None:
                    for tensor in lora_weights.lora_a:
                        if tensor is not None and isinstance(
                                tensor, nn.Parameter):
                            parameters.append(tensor)

                if lora_weights.lora_b is not None:
                    for tensor in lora_weights.lora_b:
                        if tensor is not None and isinstance(
                                tensor, nn.Parameter):
                            parameters.append(tensor)
            else:
                # Regular LoRALayerWeights: collect from single tensors
                if lora_weights.lora_a is not None and isinstance(
                        lora_weights.lora_a, nn.Parameter):
                    parameters.append(lora_weights.lora_a)

                if lora_weights.lora_b is not None and isinstance(
                        lora_weights.lora_b, nn.Parameter):
                    parameters.append(lora_weights.lora_b)

        logger.info(
            f"[TrainingManager] Collected {len(parameters)} trainable parameters for LoRA {lora_id}"
        )
        return parameters

    def setup_optimizer(
        self,
        lora_id: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> torch.optim.Optimizer:
        """Setup AdamW optimizer for LoRA parameters.
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            
        Returns:
            Configured AdamW optimizer
        """
        parameters = self.get_lora_parameters(lora_id)

        if not parameters:
            raise ValueError(
                f"No trainable parameters found for LoRA {lora_id}. "
                "Make sure to call make_lora_trainable() first.")

        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        logger.info(
            f"[TrainingManager] Created AdamW optimizer with lr={learning_rate}, "
            f"weight_decay={weight_decay}, {len(parameters)} parameters")

        return optimizer

    def setup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        scheduler_type: str = "cosine",
    ):
        """Setup learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            scheduler_type: Type of scheduler ("cosine" or "linear")
            
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # For cosine with warmup, we need to handle warmup separately
            if num_warmup_steps > 0:
                # Use a lambda scheduler for warmup then cosine
                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(
                            max(1, num_warmup_steps))
                    progress = float(current_step - num_warmup_steps) / float(
                        max(1, num_training_steps - num_warmup_steps))
                    return max(0.0, 0.5 * (1.0 + torch.cos(
                        torch.tensor(progress * 3.141592653589793))))

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda)
            else:
                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=num_training_steps)
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(optimizer,
                                 start_factor=1.0,
                                 end_factor=0.0,
                                 total_iters=num_training_steps)
        else:
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}. "
                "Use 'cosine' or 'linear'.")

        logger.info(
            f"[TrainingManager] Created {scheduler_type} scheduler with "
            f"{num_training_steps} training steps, {num_warmup_steps} warmup steps"
        )

        return scheduler

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Perform optimizer step with optional gradient clipping.
        
        Args:
            optimizer: The optimizer to step
            scheduler: Optional learning rate scheduler to step
            max_grad_norm: Optional maximum gradient norm for clipping
            
        Returns:
            Dictionary with optimizer statistics (grad_norm, learning_rate)
        """
        stats = {}

        # Gradient clipping if requested
        if max_grad_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]['params'], max_grad_norm)
            stats['grad_norm'] = total_norm.item()
        else:
            # Calculate gradient norm without clipping
            total_norm = 0.0
            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            stats['grad_norm'] = total_norm

        # Optimizer step
        optimizer.step()

        # Get current learning rate
        stats['learning_rate'] = optimizer.param_groups[0]['lr']

        # Scheduler step if provided
        if scheduler is not None:
            scheduler.step()

        logger.debug(
            f"[TrainingManager] Optimizer step: "
            f"grad_norm={stats['grad_norm']:.6f}, lr={stats['learning_rate']:.6e}"
        )

        return stats

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Zero out all gradients.
        
        Args:
            optimizer: The optimizer whose gradients to zero
        """
        optimizer.zero_grad()
        logger.debug("[TrainingManager] Gradients zeroed")

    def compute_lora_checksum(self, lora_id: int) -> Dict[str, float]:
        """Compute checksums for all LoRA parameters.
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter
            
        Returns:
            Dictionary mapping parameter names to their checksums (sum of all values)
        """
        if not self.lora_manager:
            logger.warning("[TrainingManager] LoRA manager not available")
            return {}

        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            logger.warning(f"[TrainingManager] LoRA {lora_id} not found")
            return {}

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            logger.warning(f"[TrainingManager] LoRA {lora_id} could not be retrieved")
            return {}

        checksums = {}
        
        # CRITICAL FIX: Compute checksums on the trainable nn.Parameter objects, not original LoRA tensors
        # The optimizer updates self.trainable_lora_params, so we need to check those for changes
        logger.info(f"[CHECKSUM_DEBUG] Computing checksums on trainable parameters, not original LoRA tensors")
        logger.info(f"[CHECKSUM_DEBUG] trainable_lora_params has {len(self.trainable_lora_params)} entries")
        logger.info(f"[CHECKSUM_DEBUG] current_lora_id={self.current_lora_id}, requested_lora_id={lora_id}")
        
        # BETTER FIX: Compute checksums on stacked tensors (which contain the updated values)
        # instead of relying on trainable_lora_params (which gets cleared)
        logger.info(f"[CHECKSUM_FIX] Computing checksums on stacked tensors instead")
        
        # Get the stacked tensor index from lora_id
        lora_index_to_id = self.lora_manager._adapter_manager.lora_index_to_id
        try:
            stacked_index = lora_index_to_id.index(lora_id)
        except ValueError:
            logger.warning(f"[CHECKSUM_FIX] LoRA {lora_id} not found in lora_index_to_id mapping")
            stacked_index = 0  # fallback
        
        # Iterate through model modules and compute checksums on stacked tensors
        for module_name, module in self.model.named_modules():
            if hasattr(module, 'lora_a_stacked') and hasattr(module, 'lora_b_stacked'):
                if len(module.lora_a_stacked) > 0 and len(module.lora_b_stacked) > 0:
                    # Compute checksum on the specific LoRA slot in stacked tensors
                    lora_a_checksum = module.lora_a_stacked[0][stacked_index].sum().item()
                    lora_b_checksum = module.lora_b_stacked[0][stacked_index].sum().item()
                    
                    checksums[f"{module_name}.lora_a"] = lora_a_checksum
                    checksums[f"{module_name}.lora_b"] = lora_b_checksum
                    
                    logger.info(f"[CHECKSUM_FIX] {module_name}: lora_a={lora_a_checksum:.6f}, lora_b={lora_b_checksum:.6f}")
        
        logger.info(f"[CHECKSUM_DEBUG] Computed {len(checksums)} parameter checksums")
        return checksums

    def enable_stacked_tensor_gradients(self, lora_id: int) -> Dict[str, int]:
        """Enable gradients on stacked tensors for training.
        
        vLLM's stacked tensors (lora_a_stacked, lora_b_stacked) are created as
        plain tensors without requires_grad=True. For training, we must enable
        gradients on these tensors so the backward pass can compute gradients.
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter
            
        Returns:
            Dictionary with statistics about tensors that had gradients enabled
        """
        if not self.lora_manager:
            logger.warning("[TrainingManager] LoRA manager not available")
            return {'enabled': 0}

        enabled_count = 0

        # Iterate through all model modules to find LoRA layers with stacked tensors
        for module_name, module in self.model.named_modules():
            # Check if this module has LoRA (has lora_a_stacked attribute)
            if not hasattr(module, 'lora_a_stacked') or not hasattr(module, 'lora_b_stacked'):
                continue

            # Enable gradients on lora_a_stacked
            if module.lora_a_stacked is not None:
                if isinstance(module.lora_a_stacked, tuple):
                    for stacked in module.lora_a_stacked:
                        if stacked is not None and not stacked.requires_grad:
                            stacked.requires_grad_(True)
                            enabled_count += 1
                else:
                    if not module.lora_a_stacked.requires_grad:
                        module.lora_a_stacked.requires_grad_(True)
                        enabled_count += 1

            # Enable gradients on lora_b_stacked
            if module.lora_b_stacked is not None:
                if isinstance(module.lora_b_stacked, tuple):
                    for stacked in module.lora_b_stacked:
                        if stacked is not None and not stacked.requires_grad:
                            stacked.requires_grad_(True)
                            enabled_count += 1
                else:
                    if not module.lora_b_stacked.requires_grad:
                        module.lora_b_stacked.requires_grad_(True)
                        enabled_count += 1

        logger.info(
            f"[TrainingManager] Enabled gradients on {enabled_count} stacked tensors"
        )

        return {'enabled': enabled_count}

    def copy_gradients_to_parameters(self, lora_id: int) -> Dict[str, int]:
        """Copy gradients from stacked tensors back to LoRA Parameters.
        
        This is the critical fix for training: vLLM's inference engine copies LoRA
        weights into stacked tensors (lora_a_stacked, lora_b_stacked) for batched
        inference. During training, gradients accumulate in these stacked tensors,
        but the optimizer tracks the original Parameters. We must copy gradients
        back to Parameters before optimizer.step().
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter (lora_int_id)
            
        Returns:
            Dictionary with statistics about copied gradients
        """
        if not self.lora_manager:
            logger.warning("[TrainingManager] LoRA manager not available")
            return {'copied': 0, 'skipped': 0}

        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            logger.warning(f"[TrainingManager] LoRA {lora_id} not found")
            return {'copied': 0, 'skipped': 0}

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            logger.warning(f"[TrainingManager] LoRA {lora_id} could not be retrieved")
            return {'copied': 0, 'skipped': 0}

        # CRITICAL: Get the stacked tensor index from lora_id
        # lora_id (lora_int_id) is mapped to an index in the stacked tensors
        lora_index_to_id = self.lora_manager._adapter_manager.lora_index_to_id
        try:
            stacked_index = lora_index_to_id.index(lora_id)
        except ValueError:
            logger.warning(f"[TrainingManager] LoRA {lora_id} not found in lora_index_to_id mapping")
            return {'copied': 0, 'skipped': 0}
        
        logger.debug(f"[TrainingManager] LoRA ID {lora_id} maps to stacked index {stacked_index}")

        copied_count = 0
        skipped_count = 0
        missing_grads = 0
        
        # DEBUG: First pass to see which stacked tensors have gradients
        logger.info(f"[TrainingManager] DEBUG: Checking which stacked tensors have gradients...")
        grad_status = []
        for module_name, module in self.model.named_modules():
            if hasattr(module, 'lora_a_stacked') and module.lora_a_stacked is not None:
                if isinstance(module.lora_a_stacked, (list, tuple)):
                    for i, stacked in enumerate(module.lora_a_stacked):
                        has_grad = stacked.grad is not None if isinstance(stacked, torch.Tensor) else False
                        grad_status.append(f"{module_name}.lora_a_stacked[{i}]: grad={has_grad}")
                else:
                    has_grad = module.lora_a_stacked.grad is not None
                    grad_status.append(f"{module_name}.lora_a_stacked: grad={has_grad}")
        logger.info(f"[TrainingManager] Gradient status:\n" + "\n".join(grad_status[:10]) + f"\n... ({len(grad_status)} total)")

        # Iterate through all model modules to find LoRA layers with stacked tensors
        for module_name, module in self.model.named_modules():
            # Check if this module has LoRA (has lora_a_stacked attribute)
            if not hasattr(module, 'lora_a_stacked') or not hasattr(module, 'lora_b_stacked'):
                continue

            # Get the corresponding LoRA weights from the LoRA model
            lora_weights = lora_model.loras.get(module_name)
            if lora_weights is None:
                continue

            is_packed = lora_weights.is_packed

            if is_packed:
                # PackedLoRALayerWeights: handle lists of tensors
                if lora_weights.lora_a is not None and module.lora_a_stacked is not None:
                    for i, param in enumerate(lora_weights.lora_a):
                        if param is None or not isinstance(param, nn.Parameter):
                            skipped_count += 1
                            continue

                        # Get stacked tensor at this slice index
                        if i < len(module.lora_a_stacked):
                            stacked = module.lora_a_stacked[i]
                            if stacked.grad is not None:
                                # Copy gradient from stacked tensor to Parameter
                                # Stacked shape: [max_loras, 1, rank, input_size]
                                # Need to extract [stacked_index, 0, :, :] and transpose
                                stacked_grad = stacked[stacked_index, 0, :param.shape[1], :param.shape[0]]
                                # Move to same device as parameter before assigning
                                param.grad = stacked_grad.T.clone().to(param.device)
                                copied_count += 1
                            else:
                                missing_grads += 1

                if lora_weights.lora_b is not None and module.lora_b_stacked is not None:
                    for i, param in enumerate(lora_weights.lora_b):
                        if param is None or not isinstance(param, nn.Parameter):
                            skipped_count += 1
                            continue

                        # Get stacked tensor at this slice index
                        if i < len(module.lora_b_stacked):
                            stacked = module.lora_b_stacked[i]
                            if stacked.grad is not None:
                                # Copy gradient from stacked tensor to Parameter
                                # Stacked shape: [max_loras, 1, output_size, rank]
                                # Need to extract [stacked_index, 0, :, :] and transpose
                                stacked_grad = stacked[stacked_index, 0, :param.shape[1], :param.shape[0]]
                                # Move to same device as parameter before assigning
                                param.grad = stacked_grad.T.clone().to(param.device)
                                copied_count += 1
                            else:
                                missing_grads += 1
            else:
                # Regular LoRALayerWeights: handle single tensors
                if lora_weights.lora_a is not None and isinstance(lora_weights.lora_a, nn.Parameter):
                    param = lora_weights.lora_a
                    # lora_a_stacked is a tuple of tensors for different slices
                    if isinstance(module.lora_a_stacked, tuple) and len(module.lora_a_stacked) > 0:
                        stacked = module.lora_a_stacked[0]  # Use first slice for regular (non-packed)
                        if stacked.grad is not None:
                            # Copy gradient: stacked shape [max_loras, 1, rank, input_size]
                            # Parameter shape: [input_size, rank], so need transpose
                            stacked_grad = stacked[stacked_index, 0, :param.shape[1], :param.shape[0]]
                            # Move to same device as parameter before assigning
                            param.grad = stacked_grad.T.clone().to(param.device)
                            copied_count += 1
                        else:
                            missing_grads += 1
                    else:
                        skipped_count += 1

                if lora_weights.lora_b is not None and isinstance(lora_weights.lora_b, nn.Parameter):
                    param = lora_weights.lora_b
                    # lora_b_stacked is a tuple of tensors for different slices
                    if isinstance(module.lora_b_stacked, tuple) and len(module.lora_b_stacked) > 0:
                        stacked = module.lora_b_stacked[0]  # Use first slice for regular (non-packed)
                        if stacked.grad is not None:
                            # Copy gradient: stacked shape [max_loras, 1, output_size, rank]
                            # Parameter shape: [rank, output_size], so need transpose
                            stacked_grad = stacked[stacked_index, 0, :param.shape[1], :param.shape[0]]
                            # Move to same device as parameter before assigning
                            param.grad = stacked_grad.T.clone().to(param.device)
                            copied_count += 1
                        else:
                            missing_grads += 1
                    else:
                        skipped_count += 1

        stats = {
            'copied': copied_count,
            'skipped': skipped_count,
            'missing_grads': missing_grads,
        }

        logger.info(
            f"[TrainingManager] Gradient copy: {copied_count} gradients copied, "
            f"{skipped_count} skipped, {missing_grads} missing from stacked tensors"
        )
        
        return stats

    def copy_parameters_to_stacked(self, lora_id: int) -> Dict[str, int]:
        """Copy updated Parameters back to stacked tensors after optimizer step.
        
        After optimizer.step() updates the Parameters, we must copy the updated
        values back to the stacked tensors so they're used in the next forward pass.
        
        Args:
            lora_id: The integer ID of the loaded LoRA adapter (lora_int_id)
            
        Returns:
            Dictionary with statistics about copied parameters
        """
        if not self.lora_manager:
            logger.warning("[TrainingManager] LoRA manager not available")
            return {'copied': 0, 'skipped': 0}

        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            logger.warning(f"[TrainingManager] LoRA {lora_id} not found")
            return {'copied': 0, 'skipped': 0}

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            logger.warning(f"[TrainingManager] LoRA {lora_id} could not be retrieved")
            return {'copied': 0, 'skipped': 0}

        # CRITICAL: Get the stacked tensor index from lora_id
        # lora_id (lora_int_id) is mapped to an index in the stacked tensors
        lora_index_to_id = self.lora_manager._adapter_manager.lora_index_to_id
        try:
            stacked_index = lora_index_to_id.index(lora_id)
        except ValueError:
            logger.warning(f"[TrainingManager] LoRA {lora_id} not found in lora_index_to_id mapping")
            return {'copied': 0, 'skipped': 0}
        
        logger.debug(f"[TrainingManager] LoRA ID {lora_id} maps to stacked index {stacked_index}")
        logger.info(f"[INDEX_DEBUG] UPDATE: lora_id={lora_id} -> stacked_index={stacked_index}")
        logger.info(f"[INDEX_DEBUG] UPDATE: lora_index_to_id mapping={lora_index_to_id}")
        logger.info(f"[TRACE] copy_parameters_to_stacked: self.model id={id(self.model)}, type={type(self.model).__name__}")
        logger.info(f"[TRACE] copy_parameters_to_stacked: self.model_runner id={id(self.model_runner)}")
        logger.info(f"[TRACE] copy_parameters_to_stacked: self.model_runner.model id={id(self.model_runner.model)}")

        copied_count = 0
        skipped_count = 0
        first_lora_module_logged = False

        # CRITICAL FIX: Use model_runner.model directly to ensure we sync to the same model used in forward pass
        # This bypasses the @property model which might return a different instance
        forward_model = self.model_runner.model
        logger.info(f"[TRACE] Using forward_model directly: id={id(forward_model)}, type={type(forward_model).__name__}")
        
        # Iterate through forward model modules to find LoRA layers with stacked tensors
        for module_name, module in forward_model.named_modules():
            # Check if this module has LoRA (has lora_a_stacked attribute)
            if not hasattr(module, 'lora_a_stacked') or not hasattr(module, 'lora_b_stacked'):
                continue

            # Log first LoRA module found for tensor ID tracing
            if not first_lora_module_logged:
                if hasattr(module, 'lora_a_stacked') and len(module.lora_a_stacked) > 0:
                    logger.info(f"[TRACE] First LoRA module '{module_name}': lora_a_stacked[0] tensor id={id(module.lora_a_stacked[0])}")
                first_lora_module_logged = True

            # Get the corresponding LoRA weights from the LoRA model
            lora_weights = lora_model.loras.get(module_name)
            if lora_weights is None:
                continue

            is_packed = lora_weights.is_packed

            if is_packed:
                # PackedLoRALayerWeights: handle lists of tensors
                if lora_weights.lora_a is not None and module.lora_a_stacked is not None:
                    for i, param in enumerate(lora_weights.lora_a):
                        if param is None or not isinstance(param, nn.Parameter):
                            skipped_count += 1
                            continue

                        # Copy Parameter to stacked tensor at this slice index
                        if i < len(module.lora_a_stacked):
                            stacked = module.lora_a_stacked[i]
                            # Copy transposed: param shape [input_size, rank], stacked needs [rank, input_size]
                            # DEBUG: Log copy operation details for evaluation sync
                            param_sum = param.sum().item()
                            before_sum = stacked.data[stacked_index, 0].sum().item()
                            logger.info(f"[INDEX_DEBUG] UPDATE: Writing to stacked[{stacked_index}, 0] for {module_name}")
                            stacked.data[stacked_index, 0, :param.shape[1], :param.shape[0]].copy_(param.data.T)
                            after_sum = stacked.data[stacked_index, 0].sum().item()
                            
            # Log if we're copying non-zero values or if there's a change
            if abs(param_sum) > 1e-6 or abs(after_sum - before_sum) > 1e-6:
                logger.info(f"[SYNC] {module_name}: param_sum={param_sum:.6f}, before={before_sum:.6f}, after={after_sum:.6f}")
                logger.info(f"[TENSOR_ID_SYNC] {id(stacked)}")
                # Also check the full tensor checksum after copy
                full_checksum = stacked.sum().item()
                logger.info(f"[SYNC] {module_name}: full_tensor_checksum_after_copy={full_checksum:.6f}")
                
                # CRITICAL: Add CUDA synchronization barrier
                if stacked.is_cuda:
                    torch.cuda.synchronize()
                    logger.info(f"[SYNC_BARRIER] CUDA synchronized after copy to {module_name}")
                
                # IMMEDIATE VERIFICATION: Check tensor right after sync
                immediate_verify = stacked[stacked_index].sum().item()
                logger.info(f"[IMMEDIATE_VERIFY] {module_name}: checksum_after_sync={immediate_verify:.6f}")
                
                # Store first tensor for later verification
                if not hasattr(self, '_debug_first_tensor'):
                    self._debug_first_tensor = stacked
                    self._debug_first_module = module_name
                    logger.info(f"[DEBUG_TRACK] Tracking tensor {id(stacked)} from {module_name}")
                    
                    # CRITICAL: Add tensor monitoring hook to catch when it gets zeroed
                    def tensor_monitor_hook(tensor):
                        def hook_fn(*args):
                            current_sum = tensor[stacked_index].sum().item()
                            if abs(current_sum) < 1e-6:
                                import traceback
                                logger.error(f"[TENSOR_ZEROED] ❌ Tensor {id(tensor)} was zeroed!")
                                logger.error(f"[TENSOR_ZEROED] Stack trace:\n{''.join(traceback.format_stack())}")
                        return hook_fn
                    
                    # Register hook to detect when tensor is modified
                    try:
                        stacked.register_hook(tensor_monitor_hook(stacked))
                        logger.info(f"[HOOK_REGISTERED] Monitoring tensor {id(stacked)} for modifications")
                    except Exception as e:
                        logger.warning(f"[HOOK_FAILED] Could not register hook: {e}")
                
                copied_count += 1

                if lora_weights.lora_b is not None and module.lora_b_stacked is not None:
                    for i, param in enumerate(lora_weights.lora_b):
                        if param is None or not isinstance(param, nn.Parameter):
                            skipped_count += 1
                            continue

                        # Copy Parameter to stacked tensor at this slice index
                        if i < len(module.lora_b_stacked):
                            stacked = module.lora_b_stacked[i]
                            # Copy transposed: param shape [rank, output_size], stacked needs [output_size, rank]
                            stacked.data[stacked_index, 0, :param.shape[1], :param.shape[0]].copy_(param.data.T)
                            copied_count += 1
            else:
                # Regular LoRALayerWeights: handle single tensors
                if lora_weights.lora_a is not None and isinstance(lora_weights.lora_a, nn.Parameter):
                    param = lora_weights.lora_a
                    if isinstance(module.lora_a_stacked, tuple) and len(module.lora_a_stacked) > 0:
                        stacked = module.lora_a_stacked[0]
                        stacked.data[stacked_index, 0, :param.shape[1], :param.shape[0]].copy_(param.data.T)
                        copied_count += 1
                    else:
                        skipped_count += 1

                if lora_weights.lora_b is not None and isinstance(lora_weights.lora_b, nn.Parameter):
                    param = lora_weights.lora_b
                    if isinstance(module.lora_b_stacked, tuple) and len(module.lora_b_stacked) > 0:
                        stacked = module.lora_b_stacked[0]
                        stacked.data[stacked_index, 0, :param.shape[1], :param.shape[0]].copy_(param.data.T)
                        copied_count += 1
                    else:
                        skipped_count += 1

        stats = {
            'copied': copied_count,
            'skipped': skipped_count,
        }

        # FINAL VERIFICATION: Check tracked tensor before returning
        if hasattr(self, '_debug_first_tensor') and hasattr(self, '_debug_first_module'):
            final_verify = self._debug_first_tensor[stacked_index].sum().item()
            logger.info(f"[FINAL_VERIFY] {self._debug_first_module}: checksum_before_return={final_verify:.6f}")
            logger.info(f"[FINAL_VERIFY] Tensor ID: {id(self._debug_first_tensor)}")
        
        # Force CUDA synchronization before returning
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info(f"[SYNC_COMPLETE] All CUDA operations synchronized")

        logger.info(
            f"[TrainingManager] Parameter->Stacked copy: {copied_count} parameters copied, {skipped_count} skipped"
        )

        return stats

    def step_with_accumulation(
        self,
        max_grad_norm: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Handle gradient accumulation and optimizer step.
        
        This method should be called after each backward pass. It increments
        the accumulation counter and performs an optimizer step when the
        accumulation steps are reached.
        
        Args:
            max_grad_norm: Optional maximum gradient norm for clipping
            
        Returns:
            Dictionary with optimizer statistics if step was taken, None otherwise
        """
        if self.optimizer is None:
            logger.warning("[TrainingManager] No optimizer configured, skipping step")
            return None

        self.gradient_accumulation_counter += 1
        self.training_step += 1

        # Check if we should perform an optimizer step
        if self.gradient_accumulation_counter >= self.gradient_accumulation_steps:
            # Compute checksums before optimizer step (for verification)
            checksums_before = {}
            if self.current_lora_id is not None:
                checksums_before = self.compute_lora_checksum(self.current_lora_id)
            
            # CRITICAL FIX: Copy gradients from stacked tensors to Parameters
            # This bridges the gap between vLLM's inference design and training needs
            if self.current_lora_id is not None:
                logger.debug("[TrainingManager] Copying gradients from stacked tensors to Parameters...")
                grad_copy_stats = self.copy_gradients_to_parameters(self.current_lora_id)
                logger.debug(f"[TrainingManager] Gradient copy complete: {grad_copy_stats}")
            
            # Perform optimizer step (updates Parameters)
            stats = self.optimizer_step(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                max_grad_norm=max_grad_norm,
            )
            
            # CRITICAL FIX: Copy updated Parameters back to stacked tensors
            # The optimizer updated the Parameters, now sync them to stacked tensors for next forward pass
            if self.current_lora_id is not None:
                logger.debug("[TrainingManager] Copying updated Parameters back to stacked tensors...")
                param_copy_stats = self.copy_parameters_to_stacked(self.current_lora_id)
                logger.debug(f"[TrainingManager] Parameter copy complete: {param_copy_stats}")
            
            # Verify parameters changed after optimizer step
            if self.current_lora_id is not None:
                checksums_after = self.compute_lora_checksum(self.current_lora_id)
                
                # Count total changes
                params_changed = 0
                params_unchanged = 0
                for name in checksums_after.keys():
                    checksum_before = checksums_before.get(name, 0.0)
                    checksum_after = checksums_after[name]
                    diff = abs(checksum_after - checksum_before)
                    if diff > 1e-10:
                        params_changed += 1
                        logger.info(f"[PARAM_CHANGE] ✅ {name}: {checksum_before:.6f} -> {checksum_after:.6f} (diff={diff:.6f})")
                    else:
                        params_unchanged += 1
                        if abs(checksum_before) > 1e-6:  # Only log non-zero unchanged parameters
                            logger.info(f"[PARAM_UNCHANGED] ❌ {name}: {checksum_before:.6f} -> {checksum_after:.6f} (diff={diff:.10f})")
                
                logger.debug(f"[TrainingManager] Parameter update: {params_changed}/{len(checksums_after)} changed")
                
                if params_changed == 0:
                    logger.warning("[TrainingManager] ⚠️ NO PARAMETERS CHANGED after optimizer step!")
            
            # Zero gradients after step
            self.zero_grad(self.optimizer)
            
            # Reset accumulation counter
            self.gradient_accumulation_counter = 0
            
            stats['training_step'] = self.training_step
            logger.debug(
                f"[TrainingManager] Optimizer step at training_step={self.training_step}")
            
            return stats
        else:
            logger.debug(
                f"[TrainingManager] Accumulating gradients "
                f"({self.gradient_accumulation_counter}/{self.gradient_accumulation_steps})")
            return None

    def save_lora_checkpoint(
        self,
        lora_model: LoRAModel,
        output_dir: str,
        adapter_name: str = "adapter",
    ) -> str:
        """Save LoRA adapter weights to disk.
        
        Args:
            lora_model: The LoRA model to save
            output_dir: Directory to save the adapter to
            adapter_name: Name for the adapter
            
        Returns:
            Path to the saved adapter directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare tensors for saving
        tensors = {}
        for module_name, lora_weights in lora_model.loras.items():
            # Save in PEFT format: base_model.model.{module_name}.lora_A.weight
            base_name = f"base_model.model.{module_name}"
            
            # Handle packed layers (lora_a/lora_b are lists) vs regular layers (tensors)
            if isinstance(lora_weights.lora_a, list):
                # Packed layer (e.g., QKV) - save each component separately
                for i, (lora_a_tensor, lora_b_tensor) in enumerate(zip(lora_weights.lora_a, lora_weights.lora_b)):
                    if lora_a_tensor is not None:
                        tensor_cpu = lora_a_tensor.detach().cpu() if isinstance(lora_a_tensor, torch.nn.Parameter) else lora_a_tensor.cpu()
                        tensors[f"{base_name}.lora_A.weight.{i}"] = tensor_cpu.contiguous()
                    if lora_b_tensor is not None:
                        tensor_cpu = lora_b_tensor.detach().cpu() if isinstance(lora_b_tensor, torch.nn.Parameter) else lora_b_tensor.cpu()
                        tensors[f"{base_name}.lora_B.weight.{i}"] = tensor_cpu.contiguous()
            else:
                # Regular layer - save as single tensor
                if lora_weights.lora_a is not None:
                    tensor_cpu = lora_weights.lora_a.detach().cpu() if isinstance(lora_weights.lora_a, torch.nn.Parameter) else lora_weights.lora_a.cpu()
                    tensors[f"{base_name}.lora_A.weight"] = tensor_cpu.contiguous()
                if lora_weights.lora_b is not None:
                    tensor_cpu = lora_weights.lora_b.detach().cpu() if isinstance(lora_weights.lora_b, torch.nn.Parameter) else lora_weights.lora_b.cpu()
                    tensors[f"{base_name}.lora_B.weight"] = tensor_cpu.contiguous()

        # Save adapter_config.json
        config = {
            "peft_type": "LORA",
            "r": lora_model.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": 0.0,
            "target_modules": list(self.lora_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        save_file(tensors, os.path.join(output_dir,
                                        "adapter_model.safetensors"))

        logger.info(f"[TrainingManager] Saved LoRA adapter to {output_dir}")
        logger.info(f"[TrainingManager] Saved {len(tensors)} tensors")

        return output_dir
