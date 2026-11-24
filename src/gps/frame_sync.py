from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .gps_extractor import GpsSample


@dataclass(frozen=True)
class SyncedFrameGps:
    frame_index: int
    image_path: str
    image_name: str
    source_time_seconds: float
    gps_query_time_seconds: float
    gps_time_iso: str
    latitude: float
    longitude: float
    altitude: float
    east_m: float
    north_m: float
    up_m: float
    interpolation: str


def _natural_sort_key(path: Path) -> tuple:
    import re

    parts = re.split(r'(\d+)', path.name.lower())
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def _sample_seconds(sample: GpsSample) -> float:
    return sample.timestamp.timestamp()


def _ecef_from_geodetic(latitude_deg: float, longitude_deg: float, altitude_m: float) -> tuple[float, float, float]:
    a = 6378137.0
    e_sq = 6.69437999014e-3

    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)

    n = a / math.sqrt(1.0 - e_sq * sin_lat * sin_lat)
    x = (n + altitude_m) * cos_lat * cos_lon
    y = (n + altitude_m) * cos_lat * sin_lon
    z = (n * (1.0 - e_sq) + altitude_m) * sin_lat
    return x, y, z


def _enu_from_reference(latitude_deg: float, longitude_deg: float, altitude_m: float, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> tuple[float, float, float]:
    x, y, z = _ecef_from_geodetic(latitude_deg, longitude_deg, altitude_m)
    xr, yr, zr = _ecef_from_geodetic(ref_lat_deg, ref_lon_deg, ref_alt_m)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    ref_lat = math.radians(ref_lat_deg)
    ref_lon = math.radians(ref_lon_deg)
    sin_lat = math.sin(ref_lat)
    cos_lat = math.cos(ref_lat)
    sin_lon = math.sin(ref_lon)
    cos_lon = math.cos(ref_lon)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def _infer_frame_time_seconds(
    frame_path: Path,
    frame_position: int,
    *,
    source_frame_index: Optional[int],
    stage1_method: str,
    requested_fps: float,
    start_time_seconds: float,
    source_video_fps: Optional[float],
    capture_timebase_fps: Optional[float],
) -> float:
    numeric_frame_index = source_frame_index
    if numeric_frame_index is None and frame_path.stem.isdigit():
        numeric_frame_index = int(frame_path.stem)

    if capture_timebase_fps and capture_timebase_fps > 0 and numeric_frame_index is not None:
        return numeric_frame_index / float(capture_timebase_fps)

    if stage1_method == 'sdk_stitching' and source_video_fps and numeric_frame_index is not None:
        return numeric_frame_index / float(source_video_fps)

    if requested_fps > 0:
        return float(start_time_seconds) + (frame_position / float(requested_fps))

    return float(start_time_seconds)


def _interpolate_gps_sample(samples: List[GpsSample], target_epoch_seconds: float) -> Dict[str, Any]:
    if not samples:
        raise ValueError('No GPS samples available for synchronization')

    sample_times = [_sample_seconds(sample) for sample in samples]
    insert_at = bisect_left(sample_times, target_epoch_seconds)

    if insert_at <= 0:
        sample = samples[0]
        return {
            'latitude': sample.latitude,
            'longitude': sample.longitude,
            'altitude': float(sample.elevation or 0.0),
            'gps_time_iso': sample.timestamp.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'interpolation': 'clamped-start',
        }
    if insert_at >= len(samples):
        sample = samples[-1]
        return {
            'latitude': sample.latitude,
            'longitude': sample.longitude,
            'altitude': float(sample.elevation or 0.0),
            'gps_time_iso': sample.timestamp.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'interpolation': 'clamped-end',
        }

    prev_sample = samples[insert_at - 1]
    next_sample = samples[insert_at]
    prev_time = _sample_seconds(prev_sample)
    next_time = _sample_seconds(next_sample)
    if next_time <= prev_time:
        ratio = 0.0
    else:
        ratio = (target_epoch_seconds - prev_time) / (next_time - prev_time)

    altitude_prev = float(prev_sample.elevation or 0.0)
    altitude_next = float(next_sample.elevation or altitude_prev)
    gps_time_iso = prev_sample.timestamp.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    return {
        'latitude': prev_sample.latitude + (next_sample.latitude - prev_sample.latitude) * ratio,
        'longitude': prev_sample.longitude + (next_sample.longitude - prev_sample.longitude) * ratio,
        'altitude': altitude_prev + (altitude_next - altitude_prev) * ratio,
        'gps_time_iso': gps_time_iso,
        'interpolation': 'linear',
    }


def synchronize_frames_with_gps(
    frame_paths: Iterable[str | Path],
    gps_samples: List[GpsSample],
    *,
    stage1_method: str,
    requested_fps: float,
    start_time_seconds: float,
    sync_offset_seconds: float = 0.0,
    drift_scale: float = 1.0,
    source_video_fps: Optional[float] = None,
    capture_timebase_fps: Optional[float] = None,
    source_frame_indices: Optional[Iterable[int]] = None,
) -> List[SyncedFrameGps]:
    indexed_paths = [Path(path) for path in frame_paths]
    source_indices_list = list(source_frame_indices) if source_frame_indices is not None else []
    path_entries = [
        (
            path,
            source_indices_list[index] if index < len(source_indices_list) else None,
        )
        for index, path in enumerate(indexed_paths)
    ]
    ordered_entries = sorted(path_entries, key=lambda item: _natural_sort_key(item[0]))
    if not ordered_entries or not gps_samples:
        return []

    gps_origin = gps_samples[0]
    synced_records: List[SyncedFrameGps] = []
    video_origin_epoch = _sample_seconds(gps_samples[0])

    for index, (frame_path, source_frame_index) in enumerate(ordered_entries):
        frame_time_seconds = _infer_frame_time_seconds(
            frame_path,
            index,
            source_frame_index=source_frame_index,
            stage1_method=stage1_method,
            requested_fps=requested_fps,
            start_time_seconds=start_time_seconds,
            source_video_fps=source_video_fps,
            capture_timebase_fps=capture_timebase_fps,
        )
        gps_query_seconds = (frame_time_seconds * float(drift_scale)) + float(sync_offset_seconds)
        interpolated = _interpolate_gps_sample(gps_samples, video_origin_epoch + gps_query_seconds)
        east, north, up = _enu_from_reference(
            interpolated['latitude'],
            interpolated['longitude'],
            interpolated['altitude'],
            gps_origin.latitude,
            gps_origin.longitude,
            float(gps_origin.elevation or 0.0),
        )
        synced_records.append(
            SyncedFrameGps(
                frame_index=index,
                image_path=str(frame_path),
                image_name=frame_path.name,
                source_time_seconds=frame_time_seconds,
                gps_query_time_seconds=gps_query_seconds,
                gps_time_iso=interpolated['gps_time_iso'],
                latitude=interpolated['latitude'],
                longitude=interpolated['longitude'],
                altitude=interpolated['altitude'],
                east_m=east,
                north_m=north,
                up_m=up,
                interpolation=interpolated['interpolation'],
            )
        )

    return synced_records


def export_synced_gps_json(records: List[SyncedFrameGps], output_path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> Path:
    payload = {
        'metadata': metadata or {},
        'frames': [asdict(record) for record in records],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return output


def export_synced_gps_csv(records: List[SyncedFrameGps], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'Name', 'X', 'Y', 'Z', 'Latitude', 'Longitude', 'Altitude',
            'FrameTimeSeconds', 'GpsQueryTimeSeconds', 'GpsTime', 'Interpolation',
        ])
        for record in records:
            writer.writerow([
                record.image_name,
                f'{record.east_m:.6f}',
                f'{record.north_m:.6f}',
                f'{record.up_m:.6f}',
                f'{record.latitude:.9f}',
                f'{record.longitude:.9f}',
                f'{record.altitude:.6f}',
                f'{record.source_time_seconds:.6f}',
                f'{record.gps_query_time_seconds:.6f}',
                record.gps_time_iso,
                record.interpolation,
            ])
    return output


def export_synced_gps_xmp(records: List[SyncedFrameGps], creator_tool: str = '360toolkit GPS Sync') -> int:
    count = 0
    for record in records:
        image_path = Path(record.image_path)
        xmp_path = image_path.with_suffix('.xmp')
        xmp_content = f'''<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xcr:Coordinates>relative</xcr:Coordinates>
      <xcr:Position>{record.east_m:.6f} {record.north_m:.6f} {record.up_m:.6f}</xcr:Position>
      <xcr:PosePrior>initial</xcr:PosePrior>
      <xmp:CreatorTool>{creator_tool}</xmp:CreatorTool>
      <xmp:MetadataDate>{record.gps_time_iso}</xmp:MetadataDate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
        xmp_path.write_text(xmp_content, encoding='utf-8')
        count += 1
    return count