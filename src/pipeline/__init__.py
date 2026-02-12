"""
Pipeline Module
Batch orchestration and metadata handling.
"""

# Lazy import for GUI components to avoid PyQt6 dependency in CLI-only usage
try:
    from .batch_orchestrator import BatchOrchestrator, PipelineWorker
    _has_gui = True
except ImportError:
    BatchOrchestrator = None
    PipelineWorker = None
    _has_gui = False

from .metadata_handler import MetadataHandler

__all__ = ['BatchOrchestrator', 'PipelineWorker', 'MetadataHandler']
