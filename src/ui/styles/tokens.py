"""Design tokens for 360toolkit UI themes."""

from __future__ import annotations

TOKENS = {
    "spacing": {
        "xs": 4,
        "sm": 8,
        "md": 12,
        "lg": 16,
        "xl": 24,
        "xxl": 32,
    },
    "radius": {
        "sm": 4,
        "md": 8,
        "lg": 12,
    },
    "font": {
        "family": "Segoe UI",
        "size_xs": 8,
        "size_sm": 9,
        "size_md": 9,
        "size_lg": 11,
        "size_xl": 14,
        "weight_regular": 400,
        "weight_semibold": 600,
        "weight_bold": 700,
    },
    "control": {
        "height_sm": 24,
        "height_md": 30,
        "height_lg": 34,
    },
    "color": {
        "dark": {
            "bg_app": "#1B1F24",
            "bg_surface": "#23282F",
            "bg_card": "#23282F",
            "bg_card_hover": "#2A3038",
            "bg_input": "#1F242B",
            "bg_log": "#1C2127",
            "border": "#31363C",
            "border_soft": "#3C434B",
            "text_primary": "#E8EAED",
            "text_secondary": "#B8BDC5",
            "text_muted": "#6B7280",
            "accent": "#2EA8FF",
            "accent_hover": "#57BBFF",
            "accent_pressed": "#148EE6",
            "success": "#10B981",
            "success_hover": "#34D399",
            "success_pressed": "#059669",
            "warning": "#F59E0B",
            "warning_hover": "#FBBF24",
            "danger": "#F43F5E",
            "danger_hover": "#FB7185",
            "danger_pressed": "#E11D48",
            "focus": "#2EA8FF",
            "sidebar_bg": "#1A1E23",
            "sidebar_active": "#2A313A",
            "sidebar_hover": "#242A31",
        },
        "light": {
            "bg_app": "#F3F4F6",
            "bg_surface": "#FAFAFA",
            "bg_card": "#F7F7F8",
            "bg_card_hover": "#F0F1F3",
            "bg_input": "#FDFDFD",
            "bg_log": "#F4F5F7",
            "border": "#D6D8DC",
            "border_soft": "#E2E4E8",
            "text_primary": "#1F2328",
            "text_secondary": "#4A4F57",
            "text_muted": "#6B7280",
            "accent": "#1976D2",
            "accent_hover": "#2E88E5",
            "accent_pressed": "#0E66BE",
            "success": "#10B981",
            "success_hover": "#34D399",
            "success_pressed": "#059669",
            "warning": "#F59E0B",
            "warning_hover": "#FBBF24",
            "danger": "#F43F5E",
            "danger_hover": "#FB7185",
            "danger_pressed": "#E11D48",
            "focus": "#1976D2",
            "sidebar_bg": "#ECEDEF",
            "sidebar_active": "#E1E3E7",
            "sidebar_hover": "#E6E8EB",
        },
    },
}


def get_theme_tokens(theme: str = "dark") -> dict:
    """Return theme tokens merged for convenient access."""
    theme = (theme or "dark").lower()
    if theme not in TOKENS["color"]:
        theme = "dark"

    return {
        "spacing": TOKENS["spacing"],
        "radius": TOKENS["radius"],
        "font": TOKENS["font"],
        "control": TOKENS["control"],
        "color": TOKENS["color"][theme],
    }
