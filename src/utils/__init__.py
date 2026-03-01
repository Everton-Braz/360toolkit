"""
Advanced GPU optimization utilities for 360toolkit.
Provides pinned memory pools, CUDA streams, ring buffers, and more.
"""

from importlib import import_module
from typing import Any

# Non-torch utilities (always available)
from .ring_buffer import AdaptiveRingBuffer

_LAZY_SYMBOLS = {
    'PinnedMemoryPool': ('.pinned_memory_pool', 'PinnedMemoryPool'),
    'get_pinned_pool': ('.pinned_memory_pool', 'get_pinned_pool'),
    'CUDAStreamManager': ('.cuda_stream_manager', 'CUDAStreamManager'),
    'PredictivePrefetcher': ('.predictive_prefetch', 'PredictivePrefetcher'),
    'CUDAGraphCache': ('.cuda_graph_cache', 'CUDAGraphCache'),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, symbol_name = _LAZY_SYMBOLS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value

__all__ = [
    'PinnedMemoryPool',
    'get_pinned_pool',
    'CUDAStreamManager',
    'AdaptiveRingBuffer',
    'PredictivePrefetcher',
    'CUDAGraphCache',
]
