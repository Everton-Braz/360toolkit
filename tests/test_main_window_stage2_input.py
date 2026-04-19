from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.ui.main_window import MainWindow


_APP = None


def _get_app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def test_stage2_input_dir_syncs_between_pages_and_config(tmp_path) -> None:
    _get_app()
    window = MainWindow()
    try:
        stage2_dir = tmp_path / 'equirect_frames'
        stage2_dir.mkdir()

        window._set_stage2_input_dir(str(stage2_dir))

        assert window.stage2_input_dir_edit.text() == str(stage2_dir)
        assert window.stage2_input_dir_edit_cubemap.text() == str(stage2_dir)
        assert window.get_current_config()['stage2_input_dir'] == str(stage2_dir)
    finally:
        window.close()


def test_start_pipeline_allows_stage2_only_with_selected_input_folder(tmp_path, monkeypatch) -> None:
    _get_app()
    window = MainWindow()
    try:
        stage2_dir = tmp_path / 'equirect_frames'
        stage2_dir.mkdir()
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(stage2_dir / 'frame_00001.png'), image)

        output_dir = tmp_path / 'output'
        output_dir.mkdir()

        warning_calls = []
        captured = {}

        monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: warning_calls.append((args, kwargs)))

        def _fake_run_pipeline(*, config, progress_callback, stage_complete_callback, finished_callback, error_callback):
            captured['config'] = config

        window.orchestrator.run_pipeline = _fake_run_pipeline
        window.input_file_edit.setText('')
        window.output_dir_edit.setText(str(output_dir))
        window.stage1_enable.setChecked(False)
        window.stage2_enable.setChecked(True)
        window._set_stage2_input_dir(str(stage2_dir))

        window.start_pipeline()

        assert warning_calls == []
        assert captured['config']['stage2_input_dir'] == str(stage2_dir)
        assert captured['config']['enable_stage1'] is False
        assert captured['config']['enable_stage2'] is True
    finally:
        window.close()