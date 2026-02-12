"""Reusable UI scaffolding widgets for stage pages."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class StageHeader(QFrame):
    """Reusable stage header with title and description."""

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 16)

        title_label = QLabel(title)
        title_label.setObjectName("stageTitle")
        layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setObjectName("stageDescription")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)


class CardSection(QFrame):
    """Styled card container for grouped settings."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 16, 20, 16)
        self._layout.setSpacing(12)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            self._layout.addWidget(title_label)

    def addWidget(self, widget):
        self._layout.addWidget(widget)

    def addLayout(self, layout):
        self._layout.addLayout(layout)

    def addSpacing(self, size: int):
        self._layout.addSpacing(size)


class StageSummaryStrip(QFrame):
    """Compact summary strip shown at top of each stage page."""

    def __init__(self, stage_label: str, summary: str, parent=None):
        super().__init__(parent)
        self.setObjectName("stageSummaryStrip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        badge = QLabel(stage_label)
        badge.setObjectName("stageSummaryBadge")
        layout.addWidget(badge)

        text = QLabel(summary)
        text.setObjectName("stageSummaryText")
        text.setWordWrap(True)
        layout.addWidget(text, 1)


class StageActionFooter(QFrame):
    """Consistent page footer with primary stage action and validation."""

    def __init__(self, primary_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("stageActionFooter")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(10)

        self.primary_button = QPushButton(primary_text)
        self.primary_button.setObjectName("stagePrimaryButton")
        self.primary_button.setFixedHeight(36)
        layout.addWidget(self.primary_button)

        self.validate_button = QPushButton("Validate Settings")
        self.validate_button.setObjectName("stageSecondaryButton")
        self.validate_button.setFixedHeight(36)
        layout.addWidget(self.validate_button)

        layout.addStretch()


class FormRow(QFrame):
    """Standardized label-control row with optional hint text."""

    def __init__(self, label_text: str, control: QWidget, hint: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("formRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("formRowLabel")
        row.addWidget(label)
        row.addWidget(control, 1)
        layout.addLayout(row)

        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("formRowHint")
            hint_label.setWordWrap(True)
            layout.addWidget(hint_label)


class StagePageScaffold(QWidget):
    """Page scaffold with content stack and action footer region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(16)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(16)
        root.addLayout(self.content_layout)

        self.footer_layout = QVBoxLayout()
        self.footer_layout.setSpacing(8)
        root.addLayout(self.footer_layout)

        root.addStretch()
