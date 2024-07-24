# adapted from https://github.com/InternLM/xtuner/tree/main/xtuner/model/modules/dispatch/triton_kernels/__init__.py

# Copyright (c) OpenMMLab. All rights reserved.
from .rms_norm import rms_norm_forward
from .rotary import apply_rotary_emb

__all__ = ["rms_norm_forward", "apply_rotary_emb"]
