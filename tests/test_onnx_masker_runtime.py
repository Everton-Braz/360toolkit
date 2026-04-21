from pathlib import Path

import cv2
import numpy as np
import pytest


class _RetryingFakeNet:
    def __init__(self):
        self.forward_calls = 0
        self.backend_calls = []

    def setPreferableBackend(self, backend):
        self.backend_calls.append(('backend', backend))

    def setPreferableTarget(self, target):
        self.backend_calls.append(('target', target))

    def setInput(self, _tensor):
        return None

    def forward(self, _output_names):
        self.forward_calls += 1
        if self.forward_calls == 1:
            raise cv2.error('OpenCV(4.13.0)', 'forward', 'invalid CUDA target', '', -215)
        return [np.zeros((1, 116, 8400), dtype=np.float32), np.zeros((1, 32, 160, 160), dtype=np.float32)]


def test_run_inference_falls_back_to_cpu_after_cv2dnn_gpu_error() -> None:
    from src.masking.onnx_masker import ONNXMasker

    masker = ONNXMasker.__new__(ONNXMasker)
    masker.backend = 'cv2dnn'
    masker.net = _RetryingFakeNet()
    masker.output_names = ['output0', 'output1']
    masker._cv2dnn_uses_gpu = True
    masker._cv2dnn_fallback_to_cpu = False

    outputs = masker._run_inference(np.zeros((1, 3, 640, 640), dtype=np.float32))

    assert len(outputs) == 2
    assert masker.net.forward_calls == 2
    assert masker._cv2dnn_uses_gpu is False
    assert masker._cv2dnn_fallback_to_cpu is True
    assert ('backend', cv2.dnn.DNN_BACKEND_OPENCV) in masker.net.backend_calls
    assert ('target', cv2.dnn.DNN_TARGET_CPU) in masker.net.backend_calls


def test_onnx_masker_loads_and_runs_cpu_fallback(tmp_path) -> None:
    pytest.importorskip('onnxruntime')

    from src.masking.onnx_masker import ONNXMasker

    model_path = Path('yolov8s-seg.onnx')
    if not model_path.exists():
        pytest.skip('yolov8s-seg.onnx is not available in the workspace root')

    image_path = tmp_path / 'blank.png'
    cv2.imwrite(str(image_path), np.zeros((320, 640, 3), dtype=np.uint8))

    masker = ONNXMasker(model_path=str(model_path), confidence_threshold=0.5, use_gpu=True)
    mask = masker.generate_mask(str(image_path))

    assert mask is not None
    assert mask.shape == (320, 640)
    assert masker.session.get_providers() == ['CPUExecutionProvider']