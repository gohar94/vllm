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
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.models import LoRAModel
from vllm.lora.worker_manager import WorkerLoRAManager

logger = init_logger(__name__)

# Default LoRA configuration
RANK = 8
ALPHA = 16


class TrainingManager:
    """Manages LoRA adapter training with simplified, clean architecture."""

    def __init__(
        self,
        model_runner,
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

        # Track trainable parameters and training state
        self.trainable_lora_params: Dict[str, nn.Parameter] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.current_lora_id: Optional[int] = None
        self.training_step: int = 0
        self.gradient_accumulation_steps: int = 1
        self.gradient_accumulation_counter: int = 0
        self.trained_lora_ids: set = set()

        # Determine target modules
        if sub_modules:
            self.sub_modules = sub_modules
        elif target_modules:
            self.sub_modules = self._find_target_modules(target_modules)
        else:
            # Default: apply to attention projection layers (matching PEFT default)
            self.sub_modules = self._find_target_modules(
                ["q_proj", "v_proj"])  # Match PEFT default target modules

        logger.info(f"[TrainingManager] Initialized with {len(self.sub_modules)} target modules")
        logger.info(f"[TrainingManager] LoRA rank={self.rank}, alpha={self.alpha}")
        
        # Register with LoRAModelManager
        if hasattr(self.lora_manager, '_adapter_manager'):
            self.lora_manager._adapter_manager._training_manager = self
            logger.info("[TrainingManager] Registered with LoRAModelManager")

    @property 
    def target_modules(self) -> set[str]:
        """Get the set of target module names that have trained parameters."""
        return set(self.sub_modules.keys())

    @property
    def model(self):
        """Get the actual model, unwrapping if needed."""
        if hasattr(self.model_runner, 'model'):
            model = self.model_runner.model
            # Unwrap if it's a UBatchWrapper or similar
            if hasattr(model, 'unwrap'):
                return model.unwrap()
            return model
        return self.model_runner

    def _find_target_modules(self, target_patterns: List[str]) -> List[str]:
        """Find all module names that match the target patterns."""
        target_modules = []
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in target_patterns):
                if isinstance(module, (nn.Linear, BaseLayerWithLoRA)):
                    target_modules.append(name)

        logger.info(f"[TrainingManager] Found {len(target_modules)} modules matching patterns {target_patterns}")
        return target_modules

    def freeze_base_model(self, verbose: bool = True) -> Dict[str, int]:
        """Freeze all base model parameters (non-LoRA)."""
        frozen_count = 0
        total_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            total_count += 1
            # Check if this is a LoRA parameter (including stacked tensors)
            is_lora_param = any([
                'lora_a' in name.lower(), 'lora_b' in name.lower(),
                'lora' in name.lower() and 'stacked' in name.lower(),
                'lora' in name.lower() and ('weight' in name.lower() or 'bias' in name.lower())
            ])

            if not is_lora_param:
                param.requires_grad = False
                frozen_count += 1
            else:
                # Don't set requires_grad here - let make_lora_trainable handle it
                trainable_count += 1

        # Count trainable stacked tensors separately (they're managed by make_lora_trainable)
        stacked_trainable_count = len(self.trainable_lora_params)

        stats = {
            'total': total_count,
            'frozen': frozen_count,
            'trainable': trainable_count,
            'stacked_trainable': stacked_trainable_count,
        }

        # Base model frozen, LoRA tensors trainable
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
        """Convert a loaded LoRA adapter to trainable Parameters and setup optimizer."""
        if not self.lora_manager:
            raise ValueError("LoRA manager not available")

        # Check if already set up
        if self.current_lora_id == lora_id and self.optimizer is not None:
            # Already set up, just update gradient accumulation if needed
            if self.gradient_accumulation_steps != gradient_accumulation_steps:
                self.gradient_accumulation_steps = gradient_accumulation_steps
            
            return {
                'lora_id': lora_id,
                'already_trainable': True,
                'optimizer_setup': True,
                'scheduler_setup': self.scheduler is not None,
                'learning_rate': learning_rate,
                'gradient_accumulation_steps': gradient_accumulation_steps,
            }

        # Get LoRA model and stacked tensor index
        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            raise ValueError(f"LoRA adapter {lora_id} not found in LoRA manager")

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            raise ValueError(f"LoRA adapter {lora_id} could not be retrieved")

        lora_index_to_id = self.lora_manager._adapter_manager.lora_index_to_id
        try:
            stacked_index = lora_index_to_id.index(lora_id)
        except ValueError:
            raise ValueError(f"LoRA {lora_id} not found in lora_index_to_id mapping")

        # Making stacked tensors trainable

        # Clear existing trainable parameters and make stacked tensors trainable
        trainable_params = []
        trainable_count = 0
        self.trainable_lora_params.clear()

        # Make stacked tensors directly trainable
        for module_name, module in self.model.named_modules():
            if not hasattr(module, 'lora_a_stacked') or not hasattr(module, 'lora_b_stacked'):
                continue
            
            # Check if this module has loaded LoRA weights
            if module_name not in lora_model.loras:
                continue  # Skip modules that don't have LoRA loaded
            
            # Process lora_a_stacked
            if len(module.lora_a_stacked) > 0:
                stacked_tensor = module.lora_a_stacked[0]
                stacked_tensor.requires_grad_(True)
                param_name = f"{module_name}.lora_a_stacked"
                self.trainable_lora_params[param_name] = stacked_tensor
                trainable_params.append(stacked_tensor)
                trainable_count += 1

            # Process lora_b_stacked
            if len(module.lora_b_stacked) > 0:
                stacked_tensor = module.lora_b_stacked[0]
                stacked_tensor.requires_grad_(True)
                param_name = f"{module_name}.lora_b_stacked"
                self.trainable_lora_params[param_name] = stacked_tensor
                trainable_params.append(stacked_tensor)
                trainable_count += 1

        # Setup complete
        
        # [VLLM/LORA] Print trainable LoRA parameters details
        print("\n" + "=" * 70)
        print("[VLLM/LORA] Trainable LoRA Parameters (After make_lora_trainable)")
        print("=" * 70)
        print(f"[VLLM/LORA] Total trainable parameters: {trainable_count}")
        for param_name, param_tensor in self.trainable_lora_params.items():
            print(f"[VLLM/LORA]   {param_name}:")
            print(f"[VLLM/LORA]     Shape: {param_tensor.shape}, dtype: {param_tensor.dtype}")
            print(f"[VLLM/LORA]     Stats: mean={param_tensor.mean().item():.6f}, "
                  f"std={param_tensor.std().item():.6f}, "
                  f"min={param_tensor.min().item():.6f}, "
                  f"max={param_tensor.max().item():.6f}")
            checksum = param_tensor.sum().item()
            print(f"[VLLM/LORA]     Checksum (sum): {checksum:.6f}")
            
            # If this is a packed qkv parameter, analyze its structure
            if 'qkv_proj' in param_name:
                print(f"[VLLM/LORA/QKV] Analyzing packed QKV structure for {param_name}")
                # For lora_a, the output dim is stacked [rank, rank, rank]
                # For lora_b, the input dim is stacked, output dim is [hidden, hidden, hidden]
                if 'lora_a' in param_name:
                    # lora_a shape is typically [hidden_size, 3*rank] for packed qkv
                    if param_tensor.shape[1] % 3 == 0:
                        slice_size = param_tensor.shape[1] // 3
                        q_slice = param_tensor.data[:, :slice_size]
                        k_slice = param_tensor.data[:, slice_size:2*slice_size]
                        v_slice = param_tensor.data[:, 2*slice_size:]
                        print(f"[VLLM/LORA/QKV]   Q slice: shape={q_slice.shape}, mean={q_slice.mean().item():.6f}, std={q_slice.std().item():.6f}, checksum={q_slice.sum().item():.6f}")
                        print(f"[VLLM/LORA/QKV]   K slice: shape={k_slice.shape}, mean={k_slice.mean().item():.6f}, std={k_slice.std().item():.6f}, checksum={k_slice.sum().item():.6f}")
                        print(f"[VLLM/LORA/QKV]   V slice: shape={v_slice.shape}, mean={v_slice.mean().item():.6f}, std={v_slice.std().item():.6f}, checksum={v_slice.sum().item():.6f}")
                elif 'lora_b' in param_name:
                    # lora_b shape is typically [3*rank, hidden_size] for packed qkv
                    if param_tensor.shape[0] % 3 == 0:
                        slice_size = param_tensor.shape[0] // 3
                        q_slice = param_tensor.data[:slice_size, :]
                        k_slice = param_tensor.data[slice_size:2*slice_size, :]
                        v_slice = param_tensor.data[2*slice_size:, :]
                        print(f"[VLLM/LORA/QKV]   Q slice: shape={q_slice.shape}, mean={q_slice.mean().item():.6f}, std={q_slice.std().item():.6f}, checksum={q_slice.sum().item():.6f}")
                        print(f"[VLLM/LORA/QKV]   K slice: shape={k_slice.shape}, mean={k_slice.mean().item():.6f}, std={k_slice.std().item():.6f}, checksum={k_slice.sum().item():.6f}")
                        print(f"[VLLM/LORA/QKV]   V slice: shape={v_slice.shape}, mean={v_slice.mean().item():.6f}, std={v_slice.std().item():.6f}, checksum={v_slice.sum().item():.6f}")
        print("=" * 70)

        # Setup optimizer and scheduler
        self.current_lora_id = lora_id
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_counter = 0
        self.trained_lora_ids.add(lora_id)

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        if num_training_steps is not None and num_training_steps > 0:
            self.scheduler = self.setup_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                scheduler_type=scheduler_type,
            )
        else:
            self.scheduler = None

        return {
            'lora_id': lora_id,
            'trainable_params': trainable_count,
            'optimizer_setup': True,
            'scheduler_setup': self.scheduler is not None,
            'learning_rate': learning_rate,
            'gradient_accumulation_steps': gradient_accumulation_steps,
        }

    def setup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        scheduler_type: str = "cosine",
    ):
        """Setup learning rate scheduler."""
        if scheduler_type == "cosine":
            if num_warmup_steps > 0:
                def lr_lambda(current_step: int):
                    # Fix: PyTorch LambdaLR scheduler starts from step 0, but we want step 1 to be the first training step
                    actual_step = current_step + 1
                    
                    if actual_step <= num_warmup_steps:
                        lr_factor = float(actual_step) / float(max(1, num_warmup_steps))
                        return lr_factor
                    progress = float(actual_step - num_warmup_steps) / float(
                        max(1, num_training_steps - num_warmup_steps))
                    lr_factor = max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
                    return lr_factor

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

            # Scheduler created
        return scheduler

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Perform optimizer step with optional gradient clipping."""
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
        stats['learning_rate'] = optimizer.param_groups[0]['lr']

        # Scheduler step if provided
        if scheduler is not None:
            scheduler.step()

        return stats

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Zero out all gradients."""
        optimizer.zero_grad()

    def step_with_accumulation(
        self,
        max_grad_norm: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Handle gradient accumulation and optimizer step."""
        if self.optimizer is None:
            logger.warning("[TrainingManager] No optimizer configured, skipping step")
            return None

        # Check if gradients are computed
        grad_count = 0
        for param in self.trainable_lora_params.values():
            if param.grad is not None:
                grad_count += 1
        
        if grad_count == 0:
            logger.warning(f"[TrainingManager] No gradients found in {len(self.trainable_lora_params)} LoRA parameters")
            return None

        # [VLLM/GRAD] Print gradient information (first step only)
        if self.training_step == 0:
            print("\n" + "=" * 70)
            print("[VLLM/GRAD] Gradients After Backward Pass (First Step)")
            print("=" * 70)
            print(f"[VLLM/GRAD] Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"[VLLM/GRAD] Current accumulation counter: {self.gradient_accumulation_counter}")
            total_grad_norm = 0.0
            for param_name, param in self.trainable_lora_params.items():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    grad_min = param.grad.min().item()
                    grad_max = param.grad.max().item()
                    total_grad_norm += grad_norm ** 2
                    print(f"[VLLM/GRAD] {param_name}:")
                    print(f"[VLLM/GRAD]   Grad norm: {grad_norm:.6e}, mean: {grad_mean:.6e}, std: {grad_std:.6e}")
                    print(f"[VLLM/GRAD]   Grad min: {grad_min:.6e}, max: {grad_max:.6e}")
                    print(f"[VLLM/GRAD]   Param dtype: {param.dtype}, grad dtype: {param.grad.dtype}")
                else:
                    print(f"[VLLM/GRAD] {param_name}: NO GRADIENT")
            print(f"[VLLM/GRAD] Total gradient norm (all params): {(total_grad_norm ** 0.5):.6e}")
            print("=" * 70)

        self.gradient_accumulation_counter += 1
        self.training_step += 1

        # Check if we should perform an optimizer step
        if self.gradient_accumulation_counter >= self.gradient_accumulation_steps:
            # [VLLM/GRAD] Save parameters before optimizer step (first step only)
            if self.training_step == 1:
                params_before = {}
                for param_name, param in self.trainable_lora_params.items():
                    params_before[param_name] = param.data.clone()
            
            # Perform optimizer step (directly updates the stacked tensor parameters)
            stats = self.optimizer_step(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                max_grad_norm=max_grad_norm,
            )
            
            # [VLLM/GRAD] Check parameter updates after first optimizer step
            if self.training_step == 1:
                print("\n" + "=" * 70)
                print("[VLLM/GRAD] After First Optimizer Step")
                print("=" * 70)
                for param_name, param in self.trainable_lora_params.items():
                    if param_name in params_before:
                        param_delta = (param.data - params_before[param_name]).abs()
                        delta_mean = param_delta.mean().item()
                        delta_max = param_delta.max().item()
                        new_mean = param.data.mean().item()
                        print(f"[VLLM/GRAD] {param_name}:")
                        print(f"[VLLM/GRAD]   Delta mean: {delta_mean:.6e}, max: {delta_max:.6e}")
                        print(f"[VLLM/GRAD]   New param mean: {new_mean:.6f}")
                print("=" * 70)
            
            # Zero gradients and reset accumulation counter
            self.zero_grad(self.optimizer)
            self.gradient_accumulation_counter = 0
            stats['training_step'] = self.training_step
            
            return stats
        else:
            return None

    def verify_training_health(self, lora_id: int) -> bool:
        """Verify gradients are flowing and parameters are updating."""
        if self.optimizer is None:
            logger.warning("[TrainingManager] No optimizer configured")
            return False
        
        # Check if any parameters have gradients
        has_gradients = any(p.grad is not None for p in self.optimizer.param_groups[0]['params'])
        
        # Check if parameters changed after step (simple checksum)
        if hasattr(self, '_last_param_checksum'):
            current_checksum = sum(p.sum().item() for p in self.optimizer.param_groups[0]['params'])
            params_changed = abs(current_checksum - self._last_param_checksum) > 1e-10
            self._last_param_checksum = current_checksum
            return has_gradients and params_changed
        else:
            self._last_param_checksum = sum(p.sum().item() for p in self.optimizer.param_groups[0]['params'])
            return has_gradients

    def save_lora_checkpoint(
        self,
        lora_model: LoRAModel,
        output_dir: str,
        adapter_name: str = "adapter",
    ) -> str:
        """Save LoRA adapter weights to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Prepare tensors for saving
        tensors = {}
        for module_name, lora_weights in lora_model.loras.items():
            base_name = f"base_model.model.{module_name}"
            
            # Handle packed layers vs regular layers
            if isinstance(lora_weights.lora_a, list):
                # Packed layer - save each component separately
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
            "target_modules": list(self.sub_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        save_file(tensors, os.path.join(output_dir, "adapter_model.safetensors"))

        logger.info(f"[TrainingManager] Saved LoRA adapter to {output_dir}")
        logger.info(f"[TrainingManager] Saved {len(tensors)} tensors")

        return output_dir