# SPDX-License-Identifier: Apache-2.0
# Thin alias module to expose TorchSDPABackend under the expected qualname.

from .cpu_attn import (
    TorchSDPABackend,
    TorchSDPAMetadata,
    TorchSDPABackendImpl,
    TorchSDPAMetadataBuilderV1,
)

# Provide legacy/exported class names expected by selector/platform mapping
TorchSDPAAttentionBackend = TorchSDPABackend
TorchSDPAAttentionMetadata = TorchSDPAMetadata
TorchSDPAAttentionBackendImpl = TorchSDPABackendImpl
TorchSDPAAttentionMetadataBuilderV1 = TorchSDPAMetadataBuilderV1


