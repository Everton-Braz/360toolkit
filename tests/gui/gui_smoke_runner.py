from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton, QDialog

from src.ui.main_window import MainWindow


def _safe_call(name: str, fn: Callable[[], None], failures: List[Dict[str, str]]) -> None:
    try:
        fn()
    except Exception as exc:
        failures.append({"action": name, "error": str(exc)})


def _patch_dialogs(sample_input: str, sample_output: str):
    original = {
        "getOpenFileName": QFileDialog.getOpenFileName,
        "getExistingDirectory": QFileDialog.getExistingDirectory,
        "information": QMessageBox.information,
        "warning": QMessageBox.warning,
        "critical": QMessageBox.critical,
        "question": QMessageBox.question,
        "about": QMessageBox.about,
        "exec": QDialog.exec,
    }

    QFileDialog.getOpenFileName = staticmethod(lambda *args, **kwargs: (sample_input, ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *args, **kwargs: sample_output)

    QMessageBox.information = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.warning = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    QMessageBox.about = staticmethod(lambda *args, **kwargs: None)

    QDialog.exec = lambda self: int(QDialog.DialogCode.Accepted)
    return original


def _restore_dialogs(original) -> None:
    QFileDialog.getOpenFileName = original["getOpenFileName"]
    QFileDialog.getExistingDirectory = original["getExistingDirectory"]
    QMessageBox.information = original["information"]
    QMessageBox.warning = original["warning"]
    QMessageBox.critical = original["critical"]
    QMessageBox.question = original["question"]
    QMessageBox.about = original["about"]
    QDialog.exec = original["exec"]


def main() -> int:
    app = QApplication.instance() or QApplication([])

    sample_input = str(PROJECT_ROOT / "test_export" / "benchmark_test" / "input")
    sample_output = str(PROJECT_ROOT / "test_export" / "gui_smoke_output")
    Path(sample_output).mkdir(parents=True, exist_ok=True)

    original_dialogs = _patch_dialogs(sample_input, sample_output)

    failures: List[Dict[str, str]] = []
    clicked: List[str] = []

    try:
        window = MainWindow()
        window.show()
        app.processEvents()

        # Set deterministic values
        window.input_file_edit.setText(sample_input)
        window.output_dir_edit.setText(sample_output)
        colmap = PROJECT_ROOT / "bin" / "colmap" / "colmap.exe"
        glomap = PROJECT_ROOT / "bin" / "glomap" / "glomap.exe"
        if hasattr(window, "colmap_path_edit"):
            window.colmap_path_edit.setText(str(colmap))
        if hasattr(window, "glomap_path_edit"):
            window.glomap_path_edit.setText(str(glomap))

        # Sidebar navigation
        for index, nav_btn in enumerate(getattr(window, "nav_buttons", [])):
            name = f"nav_{index}_{nav_btn.text().strip()}"
            _safe_call(name, lambda b=nav_btn: QTest.mouseClick(b, Qt.MouseButton.LeftButton), failures)
            clicked.append(name)
            app.processEvents()

        # Trigger all menu actions except Exit
        for top_action in window.menuBar().actions():
            menu = top_action.menu()
            if menu is None:
                continue
            for action in menu.actions():
                if action.isSeparator():
                    continue
                text = action.text().strip()
                if text.lower() in {"exit", "quit"}:
                    continue
                name = f"menu_{text}"
                _safe_call(name, lambda a=action: a.trigger(), failures)
                clicked.append(name)
                app.processEvents()

        # Click enabled push buttons visible on each page
        page_count = window.page_stack.count()
        for page_index in range(page_count):
            window.page_stack.setCurrentIndex(page_index)
            app.processEvents()

            buttons = window.page_stack.currentWidget().findChildren(QPushButton)
            for button in buttons:
                if not button.isEnabled():
                    continue
                text = (button.text() or "<icon>").strip()
                name = f"page{page_index}_btn_{button.objectName() or text}"
                _safe_call(name, lambda b=button: QTest.mouseClick(b, Qt.MouseButton.LeftButton), failures)
                clicked.append(name)
                app.processEvents()

        # Control bar buttons
        for button in [
            window.input_browse_btn,
            window.output_browse_btn,
            window.start_button,
            window.pause_button,
            window.stop_button,
            window.log_toggle_button,
            window.theme_toggle_button,
        ]:
            if button.isEnabled():
                text = (button.text() or "<icon>").strip()
                name = f"control_btn_{button.objectName() or text}"
                _safe_call(name, lambda b=button: QTest.mouseClick(b, Qt.MouseButton.LeftButton), failures)
                clicked.append(name)
                app.processEvents()

        # Run stage-level validators directly
        for stage_index in range(7):
            name = f"validate_stage_{stage_index}"
            _safe_call(name, lambda idx=stage_index: window._validate_stage_config(idx), failures)
            clicked.append(name)
            app.processEvents()

        report = {
            "clicked_actions": len(clicked),
            "failures": failures,
            "failure_count": len(failures),
            "sample_input": sample_input,
            "sample_output": sample_output,
        }

        print(json.dumps(report, indent=2))
        window.close()
        app.processEvents()
        return 1 if failures else 0
    finally:
        _restore_dialogs(original_dialogs)


if __name__ == "__main__":
    raise SystemExit(main())
