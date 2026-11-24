# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""DART (Detect Anything in Real Time) package integration for 360toolkit."""

import sys
from pathlib import Path

# Ensure sam3 namespace is registered for internal relative/absolute imports
_dart_dir = Path(__file__).resolve().parent
if str(_dart_dir.parent) not in sys.path:
    sys.path.insert(0, str(_dart_dir.parent))
if str(_dart_dir) not in sys.path:
    sys.path.insert(0, str(_dart_dir))

if 'sam3' not in sys.modules:
    import src.masking.dart as _dart_pkg
    sys.modules['sam3'] = _dart_pkg

from .model_builder import build_sam3_image_model, load_pruned_config, build_pruned_sam3_image_model
from .model.sam3_multiclass import Sam3MultiClassPredictor
from .model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

__version__ = "0.1.0"

__all__ = [
    "build_sam3_image_model",
    "load_pruned_config",
    "build_pruned_sam3_image_model",
    "Sam3MultiClassPredictor",
    "Sam3MultiClassPredictorFast",
]
