"""
Advanced GPU optimization utilities for 360toolkit.
Provides pinned memory pools, CUDA streams, ring buffers, and more.
"""

# Non-torch utilities (always available)
from .ring_buffer import AdaptiveRingBuffer

# GPU/torch-dependent utilities - lazy import to avoid torch circular import in frozen apps
try:
    from .pinned_memory_pool import PinnedMemoryPool, get_pinned_pool
    from .cuda_stream_manager import CUDAStreamManager
    from .predictive_prefetch import PredictivePrefetcher
    from .cuda_graph_cache import CUDAGraphCache
except Exception:
    PinnedMemoryPool = None
    get_pinned_pool = None
    CUDAStreamManager = None
    PredictivePrefetcher = None
    CUDAGraphCache = None

__all__ = [
    'PinnedMemoryPool',
    'get_pinned_pool',
    'CUDAStreamManager',
    'AdaptiveRingBuffer',
    'PredictivePrefetcher',
    'CUDAGraphCache',
]
