from __future__ import annotations

import math
from typing import List, Optional
from src.gps.gps_extractor import GpsSample

def latlon_to_local(lat_rad: float, lon_rad: float, lat0_rad: float, lon0_rad: float) -> tuple[float, float]:
    """Convert geographic coordinates to a local tangent plane (approximate ENU)."""
    R = 6371000.0  # Earth radius in meters
    x = (lon_rad - lon0_rad) * math.cos((lat_rad + lat0_rad) / 2.0) * R
    y = (lat_rad - lat0_rad) * R
    return x, y

def local_to_latlon(x_s: float, y_s: float, lat0_rad: float, lon0_rad: float) -> tuple[float, float]:
    """Invert the local metric projection back to geodetic coordinates."""
    R = 6371000.0
    lat = y_s / R + lat0_rad
    lon = x_s / (R * math.cos(lat0_rad)) + lon0_rad
    return math.degrees(lat), math.degrees(lon)

def angle_between(ax: float, ay: float, bx: float, by: float) -> float:
    """Compute the angle in radians between vectors a and b."""
    dot = ax * bx + ay * by
    na  = math.sqrt(ax * ax + ay * ay)
    nb  = math.sqrt(bx * bx + by * by)
    if na == 0.0 or nb == 0.0:
        return 0.0
    cos_theta = dot / (na * nb)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)

def _compute_median(values: List[float]) -> float:
    """Compute the median of a list of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def clean_gps_samples(
    samples: List[GpsSample],
    max_speed_m_s: float = 4.0,
    angle_min_deg: float = 20.0,
    length_min_m: float = 5.0,
    smooth_window: int = 7,
    elevation_mode: str = 'flat',
    elevation_smooth_window: int = 31,
) -> List[GpsSample]:
    """
    Clean and smooth a track of GpsSamples by filtering outliers (speed and V-turns)
    and applying a moving average smoothing filter in local metric coordinates.

    Args:
        elevation_mode: How to handle altitude values.
            'raw'    – keep original ExifTool altitudes unchanged.
            'flat'   – replace all altitudes with the robust median (ideal for flat floors).
            'smooth' – apply a moving average low-pass filter to altitude.
        elevation_smooth_window: Window size for elevation smoothing (only used when
            elevation_mode='smooth').
    """
    if not samples:
        return []
    if len(samples) < 3:
        return list(samples)

    # 1. Sort points by time
    samples_sorted = sorted(samples, key=lambda s: s.timestamp)

    # 2. Origin point for local ENU metric coordinate conversion
    lat0 = math.radians(samples_sorted[0].latitude)
    lon0 = math.radians(samples_sorted[0].longitude)
    t0 = samples_sorted[0].timestamp.timestamp()

    working_pts = []
    for s in samples_sorted:
        x, y = latlon_to_local(math.radians(s.latitude), math.radians(s.longitude), lat0, lon0)
        t_s = s.timestamp.timestamp() - t0
        working_pts.append({
            'sample': s,
            'x': x,
            'y': y,
            't_s': t_s
        })

    # 3. Speed-based spike detection
    # We compare each point to the last kept point to avoid spike deletion cascades.
    keep_speed = [True] * len(working_pts)
    last_kept_idx = 0
    for i in range(1, len(working_pts)):
        dx = working_pts[i]['x'] - working_pts[last_kept_idx]['x']
        dy = working_pts[i]['y'] - working_pts[last_kept_idx]['y']
        dt = working_pts[i]['t_s'] - working_pts[last_kept_idx]['t_s']
        
        distance = math.sqrt(dx*dx + dy*dy)
        speed = distance / dt if dt > 0.0 else 0.0
        
        if speed <= max_speed_m_s:
            last_kept_idx = i
        else:
            keep_speed[i] = False

    pts_after_speed = [working_pts[i] for i in range(len(working_pts)) if keep_speed[i]]
    if len(pts_after_speed) < 3:
        # Fallback to unsmoothed track if filtering is too aggressive
        return samples_sorted

    # 4. Turn angle-based V-turn spike detection
    keep_angle = [True] * len(pts_after_speed)
    for i in range(1, len(pts_after_speed) - 1):
        ax = pts_after_speed[i]['x'] - pts_after_speed[i-1]['x']
        ay = pts_after_speed[i]['y'] - pts_after_speed[i-1]['y']
        bx = pts_after_speed[i+1]['x'] - pts_after_speed[i]['x']
        by = pts_after_speed[i+1]['y'] - pts_after_speed[i]['y']

        len_a = math.sqrt(ax * ax + ay * ay)
        len_b = math.sqrt(bx * bx + by * by)

        # For a sharp V-turn, the interior vertex angle is very small (< 20 degrees).
        # interior angle = pi - angle_between_vectors
        angle_rad = angle_between(ax, ay, bx, by)
        interior_angle_deg = math.degrees(math.pi - angle_rad)

        if interior_angle_deg < angle_min_deg and max(len_a, len_b) > length_min_m:
            keep_angle[i] = False

    pts_clean = [pts_after_speed[i] for i in range(len(pts_after_speed)) if keep_angle[i]]
    if len(pts_clean) < 3:
        # Fallback if too few points remain after filtering
        pts_clean = pts_after_speed

    # 5. Smoothing filter in metric space (Moving Average)
    xs = [p['x'] for p in pts_clean]
    ys = [p['y'] for p in pts_clean]

    def moving_average(values: List[float], window: int) -> List[float]:
        # Ensure window is odd
        if window % 2 == 0:
            window = max(1, window - 1)
        half = window // 2
        result = []
        for j in range(len(values)):
            left = max(0, j - half)
            right = min(len(values), j + half + 1)
            window_vals = values[left:right]
            result.append(sum(window_vals) / len(window_vals))
        return result

    x_smooth = moving_average(xs, smooth_window)
    y_smooth = moving_average(ys, smooth_window)

    # 6. Compute elevation values based on the selected elevation mode
    raw_elevations = [p['sample'].elevation for p in pts_clean]
    valid_elevations = [e for e in raw_elevations if e is not None]

    if elevation_mode == 'flat' and valid_elevations:
        # Replace all altitudes with the robust track median
        median_ele = _compute_median(valid_elevations)
        final_elevations = [median_ele] * len(pts_clean)
    elif elevation_mode == 'smooth' and valid_elevations:
        # Apply a wide-window moving average to the altitude channel
        # Fill None values with median before smoothing
        median_ele = _compute_median(valid_elevations)
        filled_elevations = [e if e is not None else median_ele for e in raw_elevations]
        final_elevations = moving_average(filled_elevations, elevation_smooth_window)
    else:
        # 'raw' mode or no valid elevation data — keep originals
        final_elevations = raw_elevations

    # 7. Reproject smoothed metric coordinates back to lat/lon
    cleaned_samples = []
    for i, p in enumerate(pts_clean):
        lat_deg, lon_deg = local_to_latlon(x_smooth[i], y_smooth[i], lat0, lon0)
        cleaned_samples.append(GpsSample(
            latitude=lat_deg,
            longitude=lon_deg,
            elevation=final_elevations[i],
            timestamp=p['sample'].timestamp
        ))

    return cleaned_samples
