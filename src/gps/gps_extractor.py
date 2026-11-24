from __future__ import annotations

import json
import csv
import logging
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpsSample:
    latitude: float
    longitude: float
    elevation: Optional[float]
    timestamp: datetime


class GPSExtractor:
    """Extract a GPS track from an Insta360 INSV file via ExifTool JSON output."""

    def __init__(self, exiftool_path: str | Path | None = None):
        resolved = Path(exiftool_path).expanduser() if exiftool_path else None
        self.exiftool_path = resolved if resolved and resolved.exists() else self._auto_detect_exiftool()

    @staticmethod
    def _auto_detect_exiftool() -> Optional[Path]:
        detected = shutil.which('exiftool') or shutil.which('exiftool.exe')
        return Path(detected) if detected else None

    @staticmethod
    def _normalized_key(value: str) -> str:
        return ''.join(ch for ch in value.lower() if ch.isalnum())

    @classmethod
    def _lookup_value(cls, record: Dict[str, Any], candidate_keys: Iterable[str]) -> Any:
        normalized = {
            cls._normalized_key(str(key)): value
            for key, value in record.items()
        }
        for candidate in candidate_keys:
            candidate_norm = cls._normalized_key(candidate)
            if candidate_norm in normalized:
                return normalized[candidate_norm]
        for key_norm, value in normalized.items():
            for candidate in candidate_keys:
                candidate_norm = cls._normalized_key(candidate)
                if key_norm.endswith(candidate_norm):
                    return value
        return None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_timestamp(cls, record: Dict[str, Any]) -> Optional[datetime]:
        direct_value = cls._lookup_value(
            record,
            ('GPSDateTime', 'GPSDate/Time', 'GPSDateTimeUTC', 'DateTimeOriginal', 'SubSecDateTimeOriginal'),
        )
        if isinstance(direct_value, str):
            parsed = cls._parse_time_string(direct_value)
            if parsed is not None:
                return parsed

        gps_date = cls._lookup_value(record, ('GPSDateStamp', 'GPSDate'))
        gps_time = cls._lookup_value(record, ('GPSTimeStamp', 'GPSTime'))
        if gps_date and gps_time:
            parsed = cls._parse_time_string(f'{gps_date} {gps_time}')
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_time_string(value: str) -> Optional[datetime]:
        cleaned = str(value).strip()
        if not cleaned:
            return None

        if len(cleaned) >= 10 and cleaned[4] == ':' and cleaned[7] == ':':
            cleaned = f'{cleaned[:4]}-{cleaned[5:7]}-{cleaned[8:]}'
        cleaned = cleaned.replace('/', '-')
        if cleaned.endswith(' UTC'):
            cleaned = cleaned[:-4] + 'Z'
        if cleaned.endswith('Z'):
            cleaned = cleaned[:-1] + '+00:00'

        formats = [
            None,
            '%Y-%m-%d %H:%M:%S.%f%z',
            '%Y-%m-%d %H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
        ]
        for fmt in formats:
            try:
                if fmt is None:
                    parsed = datetime.fromisoformat(cleaned)
                else:
                    parsed = datetime.strptime(cleaned, fmt)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @classmethod
    def _expand_prefixed_records(cls, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        base_record: Dict[str, Any] = {}

        for key, value in record.items():
            if not isinstance(key, str):
                continue
            if ':' not in key:
                base_record[key] = value
                continue
            prefix, field_name = key.split(':', 1)
            if not field_name:
                continue
            grouped.setdefault(prefix, {})[field_name] = value

        if not grouped:
            return [record]

        records: List[Dict[str, Any]] = []
        has_base_gps = (
            cls._parse_float(cls._lookup_value(base_record, ('GPSLatitude', 'Latitude'))) is not None
            and cls._parse_float(cls._lookup_value(base_record, ('GPSLongitude', 'Longitude'))) is not None
        )
        if has_base_gps:
            records.append(base_record)

        def _group_sort_key(prefix: str) -> tuple[int, str]:
            match = re.search(r'(\d+)$', prefix)
            return (int(match.group(1)) if match else -1, prefix)

        for prefix in sorted(grouped.keys(), key=_group_sort_key):
            candidate = dict(base_record)
            candidate.update(grouped[prefix])
            records.append(candidate)

        return records or [record]

    @classmethod
    def parse_exiftool_json(cls, payload: str) -> List[GpsSample]:
        raw = json.loads(payload)
        if isinstance(raw, dict):
            records = [raw]
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError('ExifTool JSON payload must be a JSON object or a list of records')

        samples: List[GpsSample] = []
        seen: set[tuple[str, float, float, Optional[float]]] = set()
        for raw_record in records:
            if not isinstance(raw_record, dict):
                continue

            for record in cls._expand_prefixed_records(raw_record):
                latitude = cls._parse_float(cls._lookup_value(record, ('GPSLatitude', 'Latitude')))
                longitude = cls._parse_float(cls._lookup_value(record, ('GPSLongitude', 'Longitude')))
                if latitude is None or longitude is None:
                    continue

                elevation = cls._parse_float(cls._lookup_value(record, ('GPSAltitude', 'Altitude', 'Elevation')))
                timestamp = cls._parse_timestamp(record)
                if timestamp is None:
                    continue

                dedupe_key = (timestamp.isoformat(), latitude, longitude, elevation)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                samples.append(GpsSample(latitude=latitude, longitude=longitude, elevation=elevation, timestamp=timestamp))

        samples.sort(key=lambda item: item.timestamp)
        return samples

    @staticmethod
    def write_gpx(samples: List[GpsSample], output_path: Path, creator: str = '360toolkit / ExifTool') -> Path:
        ns = 'http://www.topografix.com/GPX/1/1'
        ET.register_namespace('', ns)

        gpx = ET.Element(f'{{{ns}}}gpx', attrib={'version': '1.1', 'creator': creator})
        metadata = ET.SubElement(gpx, f'{{{ns}}}metadata')
        ET.SubElement(metadata, f'{{{ns}}}time').text = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        track = ET.SubElement(gpx, f'{{{ns}}}trk')
        ET.SubElement(track, f'{{{ns}}}name').text = 'Insta360 GPS Data'
        segment = ET.SubElement(track, f'{{{ns}}}trkseg')

        for sample in samples:
            trkpt = ET.SubElement(
                segment,
                f'{{{ns}}}trkpt',
                attrib={
                    'lat': f'{sample.latitude:.15f}'.rstrip('0').rstrip('.'),
                    'lon': f'{sample.longitude:.15f}'.rstrip('0').rstrip('.'),
                },
            )
            if sample.elevation is not None:
                ET.SubElement(trkpt, f'{{{ns}}}ele').text = f'{sample.elevation:.5f}'.rstrip('0').rstrip('.')
            ET.SubElement(trkpt, f'{{{ns}}}time').text = sample.timestamp.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(gpx)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return output_path

    @staticmethod
    def write_csv(samples: List[GpsSample], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Timestamp', 'Latitude', 'Longitude', 'Elevation'])
            for s in samples:
                ele_val = f'{s.elevation:.6f}' if s.elevation is not None else ''
                writer.writerow([
                    s.timestamp.isoformat(),
                    f'{s.latitude:.9f}',
                    f'{s.longitude:.9f}',
                    ele_val,
                ])
        return output_path

    @staticmethod
    def write_json(samples: List[GpsSample], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for s in samples:
            data.append({
                'timestamp': s.timestamp.isoformat(),
                'latitude': s.latitude,
                'longitude': s.longitude,
                'elevation': s.elevation,
            })
        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return output_path

    @staticmethod
    def write_kml(samples: List[GpsSample], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kml_content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '  <Document>',
            '    <name>Insta360 GPS Track</name>',
            '    <Placemark>',
            '      <name>Path</name>',
            '      <LineString>',
            '        <altitudeMode>relativeToGround</altitudeMode>',
            '        <coordinates>',
        ]
        for s in samples:
            ele_val = f'{s.elevation or 0.0:.2f}'
            kml_content.append(f'          {s.longitude:.9f},{s.latitude:.9f},{ele_val}')
        kml_content.extend([
            '        </coordinates>',
            '      </LineString>',
            '    </Placemark>',
            '  </Document>',
            '</kml>',
        ])
        output_path.write_text('\n'.join(kml_content), encoding='utf-8')
        return output_path

    def extract_track(
        self,
        input_file: str | Path,
        output_path: str | Path,
        clean_path: bool = True,
        max_speed_m_s: float = 4.0,
        angle_min_deg: float = 20.0,
        length_min_m: float = 5.0,
        smooth_window: int = 7,
        elevation_mode: str = 'flat',
        elevation_smooth_window: int = 31,
    ) -> tuple[Dict[str, Any], List[GpsSample]]:
        if self.exiftool_path is None:
            return ({
                'success': False,
                'error': 'ExifTool executable not found',
                'sample_count': 0,
            }, [])

        input_path = Path(input_file)
        output_path = Path(output_path)

        cmd = [
            str(self.exiftool_path),
            '-ee',
            '-G3',
            '-a',
            '-u',
            '-api', 'requestall=3',
            '-n',
            '-j',
            str(input_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:
            return ({
                'success': False,
                'error': f'ExifTool invocation failed: {exc}',
                'sample_count': 0,
            }, [])

        if result.returncode != 0:
            stderr_text = (result.stderr or '').strip()
            return ({
                'success': False,
                'error': f'ExifTool exited {result.returncode}: {stderr_text or "unknown error"}',
                'sample_count': 0,
            }, [])

        try:
            samples = self.parse_exiftool_json(result.stdout)
        except Exception as exc:
            logger.error('[GPS] Failed to parse ExifTool JSON: %s', exc, exc_info=True)
            return ({
                'success': False,
                'error': f'Failed to parse ExifTool JSON: {exc}',
                'sample_count': 0,
            }, [])

        if not samples:
            return ({
                'success': False,
                'error': 'No GPS samples were found in the INSV metadata',
                'sample_count': 0,
            }, [])

        # Integrate path cleaning and smoothing if requested
        if clean_path:
            try:
                from src.gps.path_cleaning import clean_gps_samples
                samples = clean_gps_samples(
                    samples,
                    max_speed_m_s=max_speed_m_s,
                    angle_min_deg=angle_min_deg,
                    length_min_m=length_min_m,
                    smooth_window=smooth_window,
                    elevation_mode=elevation_mode,
                    elevation_smooth_window=elevation_smooth_window,
                )
                logger.info('[GPS] Track cleaned successfully. Remaining samples: %d', len(samples))
            except Exception as e:
                logger.error('[GPS] Failed to clean GPS track, keeping raw: %s', e, exc_info=True)

        if not samples:
            return ({
                'success': False,
                'error': 'All GPS samples were filtered out as outliers',
                'sample_count': 0,
            }, [])

        self.write_gpx(samples, output_path)
        return ({
            'success': True,
            'output_path': str(output_path),
            'sample_count': len(samples),
            'start_time': samples[0].timestamp.isoformat(),
            'end_time': samples[-1].timestamp.isoformat(),
            'source': 'exiftool',
        }, samples)

    def extract_to_gpx(self, input_file: str | Path, output_path: str | Path) -> Dict[str, Any]:
        result, _samples = self.extract_track(input_file, output_path)
        return result