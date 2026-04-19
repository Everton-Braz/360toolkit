from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from src.ui.preview_panels import EquirectPreviewWidget, build_ffmpeg_still_preview_command
from src.ui.widgets.sam3_preview_widget import SAM3PreviewWidget


_APP = None


def _get_app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_build_ffmpeg_still_preview_command_uses_timestamp_and_stream() -> None:
    command = build_ffmpeg_still_preview_command(
        'ffmpeg',
        'input.mp4',
        'frame.png',
        stream_index=2,
        timestamp=3.25,
    )

    assert command == [
        'ffmpeg',
        '-y',
        '-ss', '3.250',
        '-i', 'input.mp4',
        '-map', '0:v:2',
        '-frames:v', '1',
        'frame.png',
    ]


def test_export_current_preview_frame_falls_back_to_mp4_extraction(tmp_path) -> None:
    input_path = tmp_path / 'sample.mp4'
    input_path.write_bytes(b'video')
    output_path = tmp_path / 'preview.png'

    widget = EquirectPreviewWidget.__new__(EquirectPreviewWidget)
    widget._video_path = str(input_path)
    widget._preview_timestamp = 4.5
    widget._ffmpeg_extractor = None
    widget._build_full_resolution_preview_frame = lambda: None

    def _fake_extract(path: str, target_path: Path):
        target_path.write_bytes(b'png')
        return target_path

    widget._extract_standard_video_preview_to_path = _fake_extract

    exported = EquirectPreviewWidget.export_current_preview_frame(widget, output_path)

    assert exported == output_path
    assert output_path.exists()


def test_preview_timestamp_slider_and_spin_stay_in_sync() -> None:
    _get_app()
    widget = EquirectPreviewWidget()

    widget.set_preview_timestamp_maximum(12.5)
    observed = []
    widget.preview_timestamp_changed.connect(observed.append)

    widget._on_preview_time_slider_changed(325)

    assert widget._preview_time_spin.value() == 3.25
    assert widget._preview_time_slider.value() == 325
    assert widget._preview_timestamp == 3.25
    assert observed[-1] == 3.25
    widget.close()


def test_sam3_refresh_auto_source_image_reloads_source_and_clears_overlay(tmp_path) -> None:
    _get_app()
    image_path = tmp_path / 'source.png'
    cv2.imwrite(str(image_path), np.full((24, 32, 3), 200, dtype=np.uint8))

    widget = SAM3PreviewWidget()
    widget.set_auto_image_resolver(lambda: image_path)
    widget._preview_has_run = True
    widget._clear_image(widget.overlay_label, 'Old Overlay')

    refreshed = widget.refresh_auto_source_image(force=True, mark_overlay_stale=True)

    assert refreshed is True
    assert widget.preview_image_edit.text() == str(image_path)
    assert widget._using_auto_image is True
    assert widget._preview_has_run is False
    assert widget.overlay_label.text() == 'Mask Overlay'
    assert 'Source image refreshed' in widget.status_label.text()
    widget.close()