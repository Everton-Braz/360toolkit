from pathlib import Path

import cv2
import numpy as np
import src.pipeline.batch_orchestrator as batch_orchestrator
from src.extraction.sdk_extractor import IncompleteSDKExtractionError
from src.pipeline.batch_orchestrator import PipelineWorker


class _DummySignal:
    def emit(self, *args, **kwargs):
        return None


class _RecordingSignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FailingSDKExtractor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def is_available(self):
        return True

    def extract_frames(self, **kwargs):
        (self.output_dir / "0.jpg").write_bytes(b"partial")
        (self.output_dir / "12.jpg").write_bytes(b"partial")
        raise IncompleteSDKExtractionError(
            "SDK stalled before completing extraction: 2/3 frames",
            expected_count=3,
            actual_count=2,
            missing_indices=[24],
        )


class _UnavailableSDKExtractor:
    def is_available(self):
        return False


class _RecordingSDKExtractor:
    def __init__(self):
        self.called = False

    def is_available(self):
        return True

    def extract_frames(self, **kwargs):
        self.called = True
        raise AssertionError('SDK extractor should not be used for MP4 input')


class _RecordingFrameExtractor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.called_with = None
        self.saw_clean_output_dir = None

    def extract_frames(self, **kwargs):
        self.called_with = kwargs
        image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.saw_clean_output_dir = not any(
            path.is_file() and path.suffix.lower() in image_exts
            for path in self.output_dir.rglob('*')
        )
        restored = self.output_dir / "24.jpg"
        restored.write_bytes(b"ffmpeg")
        return {
            'success': True,
            'output_files': [str(restored)],
            'frames': [str(restored)],
            'count': 1,
            'method': kwargs['method'],
        }


def _make_worker(tmp_path: Path, input_name: str, sdk_extractor, frame_extractor=None) -> tuple[PipelineWorker, Path]:
    input_file = tmp_path / input_name
    input_file.write_bytes(b"video")
    output_root = tmp_path / "output"
    extracted_dir = output_root / "extracted_frames"
    extracted_dir.mkdir(parents=True)

    worker = PipelineWorker.__new__(PipelineWorker)
    worker.config = {
        'input_file': str(input_file),
        'output_dir': str(output_root),
        'fps': 1.0,
        'extraction_method': 'sdk',
        'allow_fallback': True,
    }
    worker.progress = _DummySignal()
    worker.sdk_extractor = sdk_extractor
    worker.frame_extractor = frame_extractor or _RecordingFrameExtractor(extracted_dir)
    worker._apply_frame_rotation = lambda *_args, **_kwargs: None

    return worker, extracted_dir


def test_execute_stage1_rejects_ffmpeg_fallback_for_stitched_insv(tmp_path) -> None:
    extracted_dir = tmp_path / "output" / "extracted_frames"
    worker, extracted_dir = _make_worker(
        tmp_path,
        "sample.insv",
        _FailingSDKExtractor(extracted_dir),
    )

    result = worker._execute_stage1()

    assert result['success'] is False
    assert 'FFmpeg stitched fallback is disabled for .insv' in result['error']
    assert worker.frame_extractor.called_with is None
    assert list(extracted_dir.glob('*.jpg')) == []


def test_execute_stage1_allows_mp4_sdk_fallback_to_ffmpeg_stitched(tmp_path) -> None:
    extracted_dir = tmp_path / "output" / "extracted_frames"
    worker, extracted_dir = _make_worker(
        tmp_path,
        "sample.mp4",
        _FailingSDKExtractor(extracted_dir),
    )

    result = worker._execute_stage1()

    assert result['success'] is True
    assert worker.frame_extractor.called_with is not None
    assert worker.frame_extractor.called_with['method'] == 'ffmpeg_stitched'
    assert worker.frame_extractor.saw_clean_output_dir is True
    assert sorted(path.name for path in extracted_dir.glob('*.jpg')) == ['24.jpg']


def test_execute_stage1_forces_mp4_sdk_selection_to_ffmpeg_without_sdk_call(tmp_path) -> None:
    extracted_dir = tmp_path / "output" / "extracted_frames"
    sdk_extractor = _RecordingSDKExtractor()
    worker, extracted_dir = _make_worker(
        tmp_path,
        "sample.mp4",
        sdk_extractor,
    )

    result = worker._execute_stage1()

    assert result['success'] is True
    assert sdk_extractor.called is False
    assert worker.frame_extractor.called_with is not None
    assert worker.frame_extractor.called_with['method'] == 'ffmpeg_stitched'


def test_execute_stage1_rejects_insv_when_sdk_missing(tmp_path) -> None:
    worker, extracted_dir = _make_worker(tmp_path, 'sample.insv', _UnavailableSDKExtractor())

    result = worker._execute_stage1()

    assert result['success'] is False
    assert 'requires Insta360 MediaSDK' in result['error']
    assert worker.frame_extractor.called_with is None
    assert list(extracted_dir.glob('*.jpg')) == []


def test_should_mask_before_split_only_for_png_outputs() -> None:
    worker = PipelineWorker.__new__(PipelineWorker)
    worker.config = {
        'enable_stage2': True,
        'enable_stage3': True,
        'skip_transform': False,
        'stage2_format': 'png',
    }

    assert worker._should_mask_before_split() is True

    worker.config['stage2_format'] = 'jpg'
    assert worker._should_mask_before_split() is False

    worker.config['stage2_format'] = 'png'
    worker.config['enable_stage3'] = False
    assert worker._should_mask_before_split() is False


def test_run_uses_pre_split_masking_for_png_outputs(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / 'output'
    extracted_dir = output_root / 'extracted_frames'
    extracted_dir.mkdir(parents=True)
    (extracted_dir / 'frame_00001.png').write_bytes(b'frame')

    worker = PipelineWorker.__new__(PipelineWorker)
    worker.config = {
        'output_dir': str(output_root),
        'enable_stage1': True,
        'enable_stage2': True,
        'enable_stage3': True,
        'skip_transform': False,
        'stage2_format': 'png',
    }
    worker.is_cancelled = False
    worker.is_paused = False
    worker.progress = _DummySignal()
    worker.error = _RecordingSignal()
    worker.finished = _RecordingSignal()
    worker.stage_complete = _RecordingSignal()

    call_order = []

    def _stage1():
        call_order.append('stage1')
        return {'success': True}

    def _stage3():
        call_order.append((
            'stage3',
            worker.config.get('stage3_input_dir'),
            worker.config.get('stage3_image_source'),
            worker.config.get('sam3_alpha_export'),
            worker.config.get('sam3_alpha_only'),
        ))
        alpha_dir = output_root / 'alpha_cutouts'
        alpha_dir.mkdir(parents=True, exist_ok=True)
        (alpha_dir / 'frame_00001_alpha.png').write_bytes(b'alpha')
        return {'success': True, 'masks_dir': str(alpha_dir)}

    def _stage2():
        call_order.append(('stage2', worker.config.get('stage2_input_dir')))
        return {'success': True, 'output_files': []}

    worker._execute_stage1 = _stage1
    worker._execute_stage2 = _stage2
    worker._execute_stage3 = _stage3
    worker._execute_realityscan_export_only = lambda: {'success': True}

    monkeypatch.setattr(batch_orchestrator, '_test_torch_cuda', lambda: None)

    worker.run()

    assert call_order[0] == 'stage1'
    assert call_order[1][0] == 'stage3'
    assert call_order[1][1] == str(extracted_dir)
    assert call_order[1][2] == 'equirect'
    assert call_order[1][3] is True
    assert call_order[1][4] is True
    assert call_order[2] == ('stage2', str(output_root / 'alpha_cutouts'))
    assert 'stage2_input_dir' not in worker.config
    assert [args[0] for args, _ in worker.stage_complete.calls] == [1, 3, 2]


def test_run_keeps_split_then_mask_for_jpeg_outputs(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / 'output'
    extracted_dir = output_root / 'extracted_frames'
    extracted_dir.mkdir(parents=True)
    (extracted_dir / 'frame_00001.jpg').write_bytes(b'frame')

    worker = PipelineWorker.__new__(PipelineWorker)
    worker.config = {
        'output_dir': str(output_root),
        'enable_stage1': True,
        'enable_stage2': True,
        'enable_stage3': True,
        'skip_transform': False,
        'stage2_format': 'jpg',
    }
    worker.is_cancelled = False
    worker.is_paused = False
    worker.progress = _DummySignal()
    worker.error = _RecordingSignal()
    worker.finished = _RecordingSignal()
    worker.stage_complete = _RecordingSignal()

    call_order = []

    worker._execute_stage1 = lambda: call_order.append('stage1') or {'success': True}
    worker._execute_stage2 = lambda: call_order.append(('stage2', worker.config.get('stage2_input_dir'))) or {'success': True, 'output_files': []}
    worker._execute_stage3 = lambda: call_order.append(('stage3', worker.config.get('stage3_input_dir'))) or {'success': True}
    worker._execute_realityscan_export_only = lambda: {'success': True}

    monkeypatch.setattr(batch_orchestrator, '_test_torch_cuda', lambda: None)

    worker.run()

    assert call_order == [
        'stage1',
        ('stage2', None),
        ('stage3', None),
    ]
    assert [args[0] for args, _ in worker.stage_complete.calls] == [1, 2, 3]


def test_realityscan_export_skips_auto_masks_for_alpha_images(tmp_path) -> None:
    output_root = tmp_path / 'output'
    perspective_dir = output_root / 'perspective_views'
    masks_dir = output_root / 'masks_perspective'
    perspective_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 10
    rgba[..., 1] = 20
    rgba[..., 2] = 30
    rgba[..., 3] = 255
    cv2.imwrite(str(perspective_dir / 'frame_00001_cam_00.png'), rgba)

    mask = np.full((6, 8), 255, dtype=np.uint8)
    cv2.imwrite(str(masks_dir / 'frame_00001_cam_00_mask.png'), mask)

    worker = PipelineWorker.__new__(PipelineWorker)
    worker.config = {
        'output_dir': str(output_root),
        'export_include_masks': True,
        'export_mask_source': 'auto',
        'alignment_mode': 'perspective_reconstruction',
    }

    result = worker._execute_realityscan_export_only()

    assert result['success'] is True
    assert result['image_count'] == 1
    assert result['mask_count'] == 0
    assert not (output_root / 'realityscan_export' / 'masks').exists()


def test_realityscan_export_keeps_auto_masks_for_non_alpha_images(tmp_path) -> None:
    output_root = tmp_path / 'output'
    perspective_dir = output_root / 'perspective_views'
    masks_dir = output_root / 'masks_perspective'
    perspective_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    cv2.imwrite(str(perspective_dir / 'frame_00001_cam_00.jpg'), rgb)

    mask = np.full((6, 8), 255, dtype=np.uint8)
    cv2.imwrite(str(masks_dir / 'frame_00001_cam_00_mask.png'), mask)

    worker = PipelineWorker.__new__(PipelineWorker)
    worker.config = {
        'output_dir': str(output_root),
        'export_include_masks': True,
        'export_mask_source': 'auto',
        'alignment_mode': 'perspective_reconstruction',
    }

    result = worker._execute_realityscan_export_only()

    assert result['success'] is True
    assert result['image_count'] == 1
    assert result['mask_count'] == 1
    assert (output_root / 'realityscan_export' / 'masks' / 'frame_00001_cam_00_mask.png').exists()