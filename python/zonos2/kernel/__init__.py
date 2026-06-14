import sys

from .index import indexing
from .radix import fast_compare_key
from .store import store_cache
from .tensor import test_tensor

if sys.platform == "win32":
    PyNCCLCommunicator = None

    def init_pynccl(*args, **kwargs):
        raise RuntimeError("PyNCCL is not available on native Windows.")

else:
    from .pynccl import PyNCCLCommunicator, init_pynccl

__all__ = [
    "indexing",
    "fast_compare_key",
    "store_cache",
    "test_tensor",
    "init_pynccl",
    "PyNCCLCommunicator",
]
