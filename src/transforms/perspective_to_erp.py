from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


class PerspectiveToERPProjector:
    """Project perspective pixels or masks back into ERP coordinates with cached maps."""

    def __init__(self):
        self.cache: Dict[Tuple[float, float, float, float, float, int, int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _resolve_v_fov(h_fov: float, output_width: int, output_height: int, v_fov: float | None) -> float:
        if v_fov is not None:
            return float(v_fov)
        return float(h_fov) * float(output_height) / float(output_width)

    def generate_projection_map(
        self,
        erp_width: int,
        erp_height: int,
        yaw: float = 0,
        pitch: float = 0,
        roll: float = 0,
        h_fov: float = 90,
        v_fov: float | None = None,
        output_width: int = 1920,
        output_height: int = 1080,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v_fov = self._resolve_v_fov(h_fov, output_width, output_height, v_fov)
        cache_key = (yaw, pitch, roll, h_fov, v_fov, output_width, output_height, erp_width, erp_height)
        if cache_key in self.cache:
            return self.cache[cache_key]

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        h_fov_rad = math.radians(h_fov)
        v_fov_rad = math.radians(v_fov)

        x_coords, y_coords = np.meshgrid(
            np.arange(output_width, dtype=np.float32),
            np.arange(output_height, dtype=np.float32),
        )

        x_norm = (x_coords - output_width / 2) / (output_width / 2)
        y_norm = (y_coords - output_height / 2) / (output_height / 2)

        focal_length_x = 1.0 / math.tan(h_fov_rad / 2)
        focal_length_y = 1.0 / math.tan(v_fov_rad / 2)

        x_3d = x_norm / focal_length_x
        y_3d = -y_norm / focal_length_y
        z_3d = np.ones_like(x_3d)

        norm = np.sqrt(x_3d ** 2 + y_3d ** 2 + z_3d ** 2)
        x_3d /= norm
        y_3d /= norm
        z_3d /= norm

        if roll_rad != 0:
            cos_roll = math.cos(roll_rad)
            sin_roll = math.sin(roll_rad)
            x_rot = x_3d * cos_roll - y_3d * sin_roll
            y_rot = x_3d * sin_roll + y_3d * cos_roll
            x_3d, y_3d = x_rot, y_rot

        if pitch_rad != 0:
            cos_pitch = math.cos(pitch_rad)
            sin_pitch = math.sin(pitch_rad)
            y_rot = y_3d * cos_pitch - z_3d * sin_pitch
            z_rot = y_3d * sin_pitch + z_3d * cos_pitch
            y_3d, z_3d = y_rot, z_rot

        if yaw_rad != 0:
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            x_rot = x_3d * cos_yaw + z_3d * sin_yaw
            z_rot = -x_3d * sin_yaw + z_3d * cos_yaw
            x_3d, z_3d = x_rot, z_rot

        phi = np.arctan2(x_3d, z_3d)
        theta = np.arcsin(np.clip(y_3d, -1.0, 1.0))

        map_x = (phi + math.pi) * erp_width / (2 * math.pi)
        map_y = (math.pi / 2 - theta) * erp_height / math.pi

        map_x = np.clip(map_x, 0, erp_width - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, erp_height - 1).astype(np.float32)
        self.cache[cache_key] = (map_x, map_y)
        return map_x, map_y

    def reproject_mask(
        self,
        perspective_mask: np.ndarray,
        erp_width: int,
        erp_height: int,
        yaw: float = 0,
        pitch: float = 0,
        roll: float = 0,
        h_fov: float = 90,
        v_fov: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if perspective_mask.ndim != 2:
            raise ValueError('perspective_mask must be a 2D array')

        output_height, output_width = perspective_mask.shape[:2]
        map_x, map_y = self.generate_projection_map(
            erp_width=erp_width,
            erp_height=erp_height,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            h_fov=h_fov,
            v_fov=v_fov,
            output_width=output_width,
            output_height=output_height,
        )

        erp_mask = np.zeros((erp_height, erp_width), dtype=np.float32)
        weight_map = np.zeros((erp_height, erp_width), dtype=np.float32)

        dest_x = np.rint(map_x).astype(np.int32)
        dest_y = np.rint(map_y).astype(np.int32)
        source = perspective_mask.astype(np.float32)

        np.maximum.at(erp_mask, (dest_y.ravel(), dest_x.ravel()), source.ravel())
        np.add.at(weight_map, (dest_y.ravel(), dest_x.ravel()), 1.0)

        return erp_mask, weight_map

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_cache_size(self) -> int:
        return len(self.cache)