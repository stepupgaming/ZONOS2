from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.platform == "win32":
    from .fallback import fast_compare_key
else:
    from functools import lru_cache
    from .utils import load_aot

    if TYPE_CHECKING:
        import torch
        from tvm_ffi import Module

    @lru_cache(maxsize=None)
    def _load_radix_module() -> Module:
        return load_aot("radix", cpp_files=["radix.cpp"])

    def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
        # compare 2 1-D int cpu tensors for equality
        return _load_radix_module().fast_compare_key(x, y)
