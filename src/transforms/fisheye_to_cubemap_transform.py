"""Generic per-lens fisheye to five cubemap faces transform."""

from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np


class FisheyeToCubemapTransform:
    """Approximate non-calibrated fisheye-to-cubemap transform."""

    FACE_ORIENTATIONS: tuple[tuple[str, float, float], ...] = (
        ('dir_plusZ', 0.0, 0.0),
        ('dir_plusX', 90.0, 0.0),
        ('dir_minusX', -90.0, 0.0),
        ('dir_plusY', 0.0, 90.0),
        ('dir_minusY', 0.0, -90.0),
    )

    def __init__(self) -> None:
        self._map_cache: dict[tuple[int, int, int, int, int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def fisheye_to_cubemap(
        self,
        fisheye_img: np.ndarray,
        face_size: int = 1024,
        lens_fov: float = 190.0,
        projection_model: str = 'equidistant',
    ) -> Dict[str, np.ndarray]:
        if projection_model not in {'equidistant', 'equisolid'}:
            raise ValueError(f'Unknown projection model: {projection_model}')

        outputs: Dict[str, np.ndarray] = {}
        for face_name, yaw, pitch in self.FACE_ORIENTATIONS:
            outputs[face_name] = self._render_face(
                fisheye_img,
                face_name=face_name,
                face_size=face_size,
                yaw=yaw,
                pitch=pitch,
                lens_fov=lens_fov,
                projection_model=projection_model,
            )
        return outputs

    def _render_face(
        self,
        fisheye_img: np.ndarray,
        *,
        face_name: str,
        face_size: int,
        yaw: float,
        pitch: float,
        lens_fov: float,
        projection_model: str,
    ) -> np.ndarray:
        height, width = fisheye_img.shape[:2]
        map_x, map_y, valid_mask = self._build_maps(
            width=width,
            height=height,
            face_size=face_size,
            yaw=yaw,
            pitch=pitch,
            lens_fov=lens_fov,
            projection_model=projection_model,
            face_name=face_name,
        )

        rendered = cv2.remap(
            fisheye_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        rendered = rendered.copy()
        if rendered.ndim == 2:
            rendered[~valid_mask] = 0
        elif rendered.shape[2] == 4:
            rendered[~valid_mask] = (0, 0, 0, 0)
        else:
            rendered[~valid_mask] = (0, 0, 0)
        return rendered

    def _build_maps(
        self,
        *,
        width: int,
        height: int,
        face_size: int,
        yaw: float,
        pitch: float,
        lens_fov: float,
        projection_model: str,
        face_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (width, height, face_size, int(round(yaw)), int(round(pitch)), int(round(lens_fov * 10)), f'{projection_model}:{face_name}')
        cached = self._map_cache.get(key)
        if cached is not None:
            return cached

        coords = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        rays = np.stack([xx, -yy, np.ones_like(xx)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-8

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)

        pitch_rot = np.empty_like(rays)
        pitch_rot[..., 0] = rays[..., 0]
        pitch_rot[..., 1] = rays[..., 1] * cos_pitch - rays[..., 2] * sin_pitch
        pitch_rot[..., 2] = rays[..., 1] * sin_pitch + rays[..., 2] * cos_pitch

        rotated = np.empty_like(rays)
        rotated[..., 0] = pitch_rot[..., 0] * cos_yaw + pitch_rot[..., 2] * sin_yaw
        rotated[..., 1] = pitch_rot[..., 1]
        rotated[..., 2] = -pitch_rot[..., 0] * sin_yaw + pitch_rot[..., 2] * cos_yaw

        theta = np.arccos(np.clip(rotated[..., 2], -1.0, 1.0))
        phi = np.arctan2(rotated[..., 1], rotated[..., 0])

        half_fov = math.radians(max(1.0, lens_fov) / 2.0)
        max_radius = min(width, height) * 0.5

        if projection_model == 'equidistant':
            radius = (theta / half_fov) * max_radius
        else:
            denom = math.sin(half_fov / 2.0)
            radius = np.zeros_like(theta)
            if denom > 1e-8:
                radius = max_radius * (np.sin(theta / 2.0) / denom)

        center_x = (width - 1) * 0.5
        center_y = (height - 1) * 0.5
        map_x = center_x + radius * np.cos(phi)
        map_y = center_y + radius * np.sin(phi)

        valid_mask = theta <= half_fov
        valid_mask &= radius <= max_radius
        valid_mask &= map_x >= 0
        valid_mask &= map_x <= (width - 1)
        valid_mask &= map_y >= 0
        valid_mask &= map_y <= (height - 1)

        result = (
            np.clip(map_x, 0, width - 1).astype(np.float32),
            np.clip(map_y, 0, height - 1).astype(np.float32),
            valid_mask,
        )
        self._map_cache[key] = result
        return result