import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


def test_masking_engine_combo_exposes_sam3_and_yolo() -> None:
    _get_app()
    window = MainWindow()
    try:
        items = [
            (window.masking_engine_combo.itemText(index), window.masking_engine_combo.itemData(index))
            for index in range(window.masking_engine_combo.count())
        ]

        assert ("SAM 3", "sam3_cpp") in items
        assert ("YOLO Segmentation", "yolo") in items

        window._set_combo_data(window.masking_engine_combo, "yolo")
        window.on_masking_engine_changed(window.masking_engine_combo.currentIndex())

        assert window._normalize_masking_engine(window.masking_engine_combo.currentData()) == "yolo"
        assert window.sam3_options_container.isHidden() is True
        assert window.model_size_container.isHidden() is False
        assert window.mask_output_mode_container.isHidden() is False
    finally:
        window.close()


def test_apply_loaded_config_preserves_yolo_masking_engine(monkeypatch) -> None:
    _get_app()
    window = MainWindow()
    try:
        monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
        window.apply_loaded_config({"masking_engine": "yolo"})

        assert window.masking_engine_combo.currentData() == "yolo"
        assert window._normalize_masking_engine(window.masking_engine_combo.currentData()) == "yolo"
    finally:
        window.close()


def test_apply_loaded_config_preserves_generic_mask_output_mode(monkeypatch) -> None:
    _get_app()
    window = MainWindow()
    try:
        monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
        window.apply_loaded_config({"masking_engine": "yolo", "mask_output_mode": "both"})

        assert window._get_mask_output_mode() == "both"
    finally:
        window.close()
