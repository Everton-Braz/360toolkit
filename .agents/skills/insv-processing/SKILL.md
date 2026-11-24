---
name: insv-processing
description: >-
  Comprehensive guide, extraction runbook, and technical reference for Insta360 INSV
  (Insta360 Video) files. Covers container architecture, binary trailer inspection,
  MediaSDK GPU stitching, raw dual-fisheye extraction, ExifTool GPS/telemetry decoding,
  track cleaning/smoothing, frame-to-GPS synchronization, and photogrammetry export
  (GPX, CSV, JSON, KML, RealityScan/RealityCapture XMP sidecars, and COLMAP priors).
---

# Insta360 INSV File Processing & Management Guide

This skill provides an exhaustive technical reference, practical runbooks, and automation scripts for managing, inspecting, extracting frames from, and extracting telemetry/GPS from **Insta360 INSV (`.insv`)** video files.

---

## 1. Overview of INSV Architecture & Internals

### 1.1 Container Structure
An `.insv` file is an **ISO Base Media File Format (MPEG-4 Part 14 / QuickTime)** container with proprietary metadata streams and a binary trailer appended at the very end of the file:
- **Video Tracks**: Contains one or two high-resolution H.264/H.265 video streams representing unstitched fisheye lens recordings.
- **Audio Track**: Standard AAC stereo or multi-channel spatial audio.
- **Proprietary Metadata Tracks**: Embedded per-frame IMU (gyroscope + accelerometer), exposure data, and timecode.
- **Binary Trailer**: Appended at the end of the file (EOF), containing camera serial number, lens calibration parameters, offset matrices, camtype tokens, and stitching models.

```
+-------------------------------------------------------------+
| ISO BMFF Container (ftyp, moov, mdat)                       |
|  - Video Stream 0:v:0 (Lens 1 / Front / Zenith Fisheye)     |
|  - Video Stream 0:v:1 (Lens 2 / Back / Nadir Fisheye)       |
|  - Audio Stream & Embedded Telemetry Tracks                 |
+-------------------------------------------------------------+
| INSV Binary Trailer (Appended at EOF)                       |
|  - Protobuf ExtraMetadata & Calibration String              |
|  - Gyroscope & Accelerometer Log Blocks                     |
|  - Trailer Header Table (72 bytes)                          |
|  - Magic Footer: 8db42d694ccc418790edff439fe026bf (32 B)   |
+-------------------------------------------------------------+
```

### 1.2 File Naming & Multi-Stream Configurations

1. **Dual-File Track Pairs (Older / 5.7K Cameras: ONE X, ONE R, ONE X2, X3)**:
   - File 1 (Front Lens): `VID_YYYYMMDD_HHMMSS_00_XXX.insv` (`_00_` marker)
   - File 2 (Back Lens): `VID_YYYYMMDD_HHMMSS_10_XXX.insv` (`_10_` marker)
   - *Rule*: Both files must remain in the same directory for MediaSDK stitching.

2. **Single-File Multi-Stream Containers (X4, Antigravity A1, Newer Rigs)**:
   - Single `.insv` file with two embedded video streams:
     - `0:v:0` = Lens 1 (Front / Top)
     - `0:v:1` = Lens 2 (Back / Bottom)
   - Stream geometries:
     - **Dual Independent Tracks**: Two separate `3840×3840` streams.
     - **Vertically Stacked**: Single `3840×3840` stream containing two `3840×1920` halves.
     - **Side-by-Side**: Single `3840×1920` stream containing two `1920×1920` halves.

### 1.3 Binary Trailer Anatomy & Calibration Strings

- **Trailer Magic Footer (Last 32 bytes)**:
  ```python
  MAGIC = b"8db42d694ccc418790edff439fe026bf"
  ```
- **Trailer Header Size**: `72` bytes (`32 + 4 + 4 + 32`).
- **Calibration String (`ExtraMetadata.offset` / `original_offset`)**:
  ASCII string starting with `N_` (where `N` is lens count, typically `2_`):
  ```text
  2_focal_cx_cy_tiltcx_tiltcy_yaw_pitch_roll_p0_p1_p2_d0_d1_d2_d3_d4_tilew_tileh_camtype_...
  ```
  - **19 parameters per lens block**:
    1. `focal`: Focal length
    2. `cx`, `cy`: Optical center coordinates
    3. `tilt_cx`, `tilt_cy`: Lens tilt center
    4. `yaw`, `pitch`, `roll`: Relative lens rotation angles
    5. `p0`, `p1`, `p2`: Tangential distortion parameters
    6. `d0`, `d1`, `d2`, `d3`, `d4`: Radial distortion coefficients
    7. `tile_w`, `tile_h`: Sub-tile dimensions
    8. `camtype`: Camera lens type identifier token (e.g. `_112_`, `_155_`, `_113_`, `_156_`)

### 1.4 Camera Model Detection Signatures
Scanning the trailing 1 MB (`INSV_TRAILER_SCAN_BYTES = 1048576`) yields known camera markers:
- `b'antigravity a1'` -> **Antigravity A1**
- `b'insta360 x5'` -> **Insta360 X5**
- `b'insta360 x4'` -> **Insta360 X4**
- `b'insta360 x3'` -> **Insta360 X3**
- `b'insta360 one x2'` -> **Insta360 ONE X2**
- `b'insta360 one x'` -> **Insta360 ONE X**
- Camtype tokens `_112_` / `_155_` -> **Antigravity A1**

---

## 2. Frame Extraction Methods

### Method 1: Official Insta360 MediaSDK (Primary / Highest Quality)

Uses the official `MediaSDKTest.exe` / `MediaSDK-Demo.exe` with GPU acceleration.

#### Stitching Algorithms
| Mode | Quality | Speed | Best For |
| :--- | :--- | :--- | :--- |
| `dynamicstitch` | **Perfect** (Dynamic optical flow + blending) | ~1.4s / 8K frame | **Recommended Default** |
| `aistitch` | **Best** (AI model v2 neural blending) | ~2.2s / 8K frame | Highest detail, textured scenes |
| `optflow` | High (Traditional optical flow) | ~1.4s / 8K frame | General stitching |
| `template` | Low / Draft (Static geometric warp) | ~0.2s / frame | CPU fallback / fast previews |

#### MediaSDK Command-Line Interface (CLI)
```bash
MediaSDKTest.exe \
  -inputs "C:\Path\VID_00_001.insv" "C:\Path\VID_10_001.insv" \
  -image_sequence_dir "C:\Output\Frames" \
  -image_type jpg \
  -export_frame_index 0-24-48-72-96 \
  -model_dir "C:\MediaSDK\models" \
  -stitch_type dynamicstitch \
  -enable_stitchfusion \
  -enable_flowstate \
  -enable_directionlock \
  -enable_colorplus \
  -output_size 7680x3840 \
  -disable_cuda false \
  -enable_soft_encode false \
  -enable_soft_decode false \
  -image_processing_accel auto
```

#### Key MediaSDK Flags & Rules
1. **Frame Sequence Format**: Must be **dash-separated** (`-export_frame_index 0-24-48`), NOT comma-separated. The flag name is singular.
2. **Stitch Fusion (`-enable_stitchfusion`)**: Chromatic and exposure auto-calibration between lenses. **Critical** for eliminating seam lines.
3. **Direction Lock (`-enable_directionlock`)**: Stabilizes the horizon and locks heading. **Requires** `-enable_flowstate`.
4. **Resolution (`-output_size`)**: Must maintain a **2:1 aspect ratio** (e.g. `7680x3840` for 8K, `5760x2880` for 5.7K, `3840x1920` for 4K).
5. **Native Color Grading Flags** (integer `[-100, 100]`):
   - `-exposure`, `-highlights`, `-shadows`, `-contrast`, `-brightness`, `-blackpoint`, `-saturation`, `-vibrance`, `-warmth`, `-tint`, `-definition` (`[0, 100]`).
6. **Path Handling**: MediaSDK fails on non-ASCII / Unicode directory paths. Always stage extraction in `%TEMP%` if the destination path contains special characters, then move finalized files.

#### Handling Unsupported Camtypes (e.g., Antigravity A1 / Custom Sensors)
If MediaSDK outputs `CameraLensType:155 / no implemention!`, patch the trailer in a temporary copy:
- Replace `_112_` with `_113_` (offset field)
- Replace `_155_` with `_156_` (original_offset field)
- Apply 180° image rotation post-extraction if required by sensor orientation.

---

### Method 2: FFmpeg Raw Dual-Fisheye Stream Extraction

Extracts individual fisheye lens images losslessly without stitching (for custom calibration or fisheye photogrammetry).

```bash
# Extract Lens 1 (Front / Stream 0:v:0)
ffmpeg -ss 0.0 -i "input.insv" -map 0:v:0 -vf "fps=2.0" -q:v 2 -y "output/frame_%05d_lens1.png"

# Extract Lens 2 (Back / Stream 0:v:1)
ffmpeg -ss 0.0 -i "input.insv" -map 0:v:1 -vf "fps=2.0" -q:v 2 -y "output/frame_%05d_lens2.png"
```

---

### Method 3: FFmpeg v360 Dual-Fisheye Stitching (Zenith/Nadir Rigs)

For dual-fisheye cameras with top/bottom (zenith/nadir) lens orientation:

```bash
ffmpeg -ss 0.0 -i "input.insv" \
  -filter_complex "[0:v:0][0:v:1]hstack=inputs=2[st];[st]v360=dfisheye:equirect:ih_fov=185:iv_fov=185:pitch=-90:yaw=0:roll=0,fps=2.0" \
  -q:v 2 -y "output/frame_%05d.jpg"
```

---

## 3. GPS & Telemetry Extraction

Insta360 cameras record GNSS data when paired with the Insta360 GPS Remote, smartphone app, or compatible smartwatch/action remotes.

> [!NOTE]
> The official Insta360 MediaSDK exposes only IMU/gyroscope data, **not** decoded GNSS/GPS tracks. **ExifTool** is the standard tool for extracting embedded GPS telemetry from `.insv` containers.

### 3.1 ExifTool Extraction Command

```bash
exiftool -ee -G3 -a -u -api requestall=3 -n -j "input.insv" > "telemetry.json"
```
- `-ee`: Extract embedded data from all metadata tracks/blocks.
- `-G3`: Print group names showing metadata source stream.
- `-api requestall=3`: Process all private binary MakerNotes and camera data.
- `-n`: Numeric output for coordinates (decimal degrees) and timestamps.
- `-j`: JSON formatted output.

### 3.2 GPS Sample Structure
Insta360 records GPS at **up to 10 Hz** (10 samples per second):
- `GPSLatitude` / `Latitude`: Decimal degrees (WGS84).
- `GPSLongitude` / `Longitude`: Decimal degrees (WGS84).
- `GPSAltitude` / `Altitude`: Meters above sea level.
- `GPSDateTime` / `GPSDateTimeUTC` / `SubSecDateTimeOriginal`: ISO 8601 timestamp with fractional sub-second precision.

---

## 4. GPS Path Cleaning & Outlier Filtering

Raw GPS tracks from action cameras often contain multi-path reflections, stationary jitter, and speed spikes.

### 4.1 Coordinate Projection to Local Metric Space
Convert geodetic coordinates $(lat, lon)$ to a local East-North-Up (ENU) tangent plane:
$$x = (\lambda - \lambda_0) \cos\left(\frac{\phi + \phi_0}{2}\right) R$$
$$y = (\phi - \phi_0) R$$
where $R = 6,371,000\text{ m}$.

### 4.2 Outlier Filtering Algorithm
1. **Speed Spike Filter**: Compute velocity $v = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t}$. If $v > v_{\max}$ (default $4.0\text{ m/s}$ for walking/scanning), reject sample.
2. **V-Turn (Angle) Filter**: Detect acute angular spikes where the camera trajectory suddenly doubles back ($\theta_{\text{interior}} < 20^\circ$ with displacement $> 5.0\text{ m}$).
3. **Metric Moving Average Smoothing**: Apply windowed moving average across local $(x, y)$ coordinates ($w = 7$).
4. **Elevation Stabilization**:
   - `flat`: Replace elevation with the robust track **median** (recommended for indoor/level photogrammetry).
   - `smooth`: Apply moving average filter ($w = 31$).
   - `raw`: Retain original GNSS altitude.
5. **Reprojection**: Transform smoothed $(x, y)$ back to $(lat, lon)$.

---

## 5. Frame-to-GPS Synchronization & Photogrammetry Export

### 5.1 Timebase Alignment Model
Given:
- Video start time: $t_0 = 0.0\text{ s}$
- Frame index $f$ and nominal video frame rate $fps_{\text{video}}$
- Time offset $\Delta t$ and drift scale factor $\alpha$ (default $1.0$):

$$t_f = \frac{f}{fps_{\text{video}}}$$
$$t_{\text{query}}^{\text{gps}} = \alpha \cdot t_f + \Delta t$$

### 5.2 Sub-Second Linear Interpolation
For query timestamp $t_{\text{query}}^{\text{gps}}$ between consecutive GPS samples $(t_1, p_1)$ and $(t_2, p_2)$:
$$r = \frac{t_{\text{query}}^{\text{gps}} - t_1}{t_2 - t_1}$$
$$lat_f = lat_1 + r \cdot (lat_2 - lat_1)$$
$$lon_f = lon_1 + r \cdot (lon_2 - lon_1)$$
$$alt_f = alt_1 + r \cdot (alt_2 - alt_1)$$

### 5.3 Export Formats

#### 1. GPX 1.1 Track (`.gpx`)
Standard GPX track with sub-second timestamps:
```xml
<?xml version="1.0" encoding="utf-8"?>
<gpx version="1.1" creator="360toolkit" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Insta360 GPS Track</name>
    <trkseg>
      <trkpt lat="-3.728123" lon="-38.524987">
        <ele>32.40000</ele>
        <time>2026-03-27T16:27:28.100Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>
```

#### 2. RealityScan / RealityCapture XMP Sidecar (`.xmp`)
Saved alongside each frame (e.g. `frame_00001.xmp` next to `frame_00001.jpg`):
```xml
<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xcr:Coordinates>relative</xcr:Coordinates>
      <xcr:Position>0.262425 -2.263975 7.038791</xcr:Position>
      <xcr:PosePrior>initial</xcr:PosePrior>
      <xmp:CreatorTool>360toolkit GPS Sync</xmp:CreatorTool>
      <xmp:MetadataDate>2026-03-27T16:27:28.100Z</xmp:MetadataDate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
```

#### 3. COLMAP GPS Prior CSV
```csv
Name,X,Y,Z,Latitude,Longitude,Altitude,FrameTimeSeconds,GpsQueryTimeSeconds,GpsTime,Interpolation
frame_00001.jpg,0.000000,0.000000,0.000000,-3.728123000,-38.524987000,32.400000,0.000000,0.000000,2026-03-27T16:27:28.000Z,linear
frame_00002.jpg,0.521300,1.241000,0.012000,-3.728118000,-38.524982000,32.412000,0.500000,0.500000,2026-03-27T16:27:28.500Z,linear
```

---

## 6. Python Automation & Inspection Tools

### 6.1 INSV Trailer Inspector & Diagnostics Script

Save as `inspect_insv.py`:
```python
import os, sys, re, struct

MAGIC = b"8db42d694ccc418790edff439fe026bf"

def inspect_insv(file_path: str):
    print(f"\n=== Inspecting INSV: {file_path} ===")
    file_size = os.path.getsize(file_path)
    print(f"File Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    with open(file_path, "rb") as f:
        # Check magic at EOF
        f.seek(-32, 2)
        tail_magic = f.read(32)
        if tail_magic == MAGIC:
            print("[OK] Valid Insta360 Trailer Magic detected.")
        else:
            print("[WARN] Magic NOT found at EOF — file may be corrupted or truncated.")

        # Read last 1MB for trailer inspection
        scan_size = min(file_size, 1024 * 1024)
        f.seek(-scan_size, 2)
        trailer_data = f.read(scan_size)

    # Detect Camera Model
    camera_signatures = [
        (b"antigravity a1", "Antigravity A1"),
        (b"insta360 x5", "Insta360 X5"),
        (b"insta360 x4", "Insta360 X4"),
        (b"insta360 x3", "Insta360 X3"),
        (b"insta360 one x2", "Insta360 ONE X2"),
        (b"insta360 one x", "Insta360 ONE X"),
    ]
    detected_model = "Unknown"
    trailer_lower = trailer_data.lower()
    for sig, model in camera_signatures:
        if sig in trailer_lower:
            detected_model = model
            break
    print(f"Detected Camera Model: {detected_model}")

    # Parse Calibration Offset String
    calib_match = re.search(rb"2_[\d.]+_[\d.]+_[\d.]+_[^_]", trailer_data)
    if calib_match:
        start_idx = calib_match.start()
        end_idx = start_idx
        while end_idx < len(trailer_data) and 32 <= trailer_data[end_idx] < 127:
            end_idx += 1
        calib_str = trailer_data[start_idx:end_idx].decode("ascii", errors="replace")
        fields = calib_str.split("_")
        print(f"\nCalibration String Found ({len(calib_str)} chars):")
        print(f"  Raw: {calib_str[:120]}...")
        if len(fields) >= 20:
            print(f"  Lens 1 Camtype Token: _{fields[19]}_")
        if len(fields) >= 39:
            print(f"  Lens 2 Camtype Token: _{fields[38]}_")
    else:
        print("\n[INFO] Calibration string pattern not found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_insv(sys.argv[1])
    else:
        print("Usage: python inspect_insv.py <path_to_file.insv>")
```

---

## 7. Troubleshooting & Best Practices

| Issue / Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| `0xC0000135` (Missing DLL) | Missing `nvcuda.dll` or VC++ Redistributable | Install latest NVIDIA GPU drivers and Microsoft Visual C++ 2015–2022 Redistributable (x64). |
| `0xC0000005` (Access Violation) | CUDA DLL version mismatch between MediaSDK (CUDA 10/11) and PyTorch (CUDA 12) | Build clean `PATH` for MediaSDK subprocess excluding `_internal/torch` directories. |
| MediaSDK exits with code 1 / stalls | Output directory contains non-ASCII/Unicode characters | Stage extraction in ASCII `%TEMP%` directory and move files after completion. |
| `CameraLensType:155 / no implemention!` | Lens camtype `155` not in MediaSDK dispatch table | Patch trailer copies `_112_` -> `_113_` and `_155_` -> `_156_` before SDK invocation. |
| Black / empty output frames | GPU out of memory or stitcher failure | Reduce `-output_size` from 8K to 5.7K/4K, or switch quality preset from `aistitch` to `dynamicstitch`. |
| Missing frames in sparse export | SDK keyframe seek drops single target indices | Use dense overlap window extraction or single-frame retry strategy (`_recover_single_frame_with_retry`). |
| GPS samples not found | Video recorded without GPS accessory or connection | Verify recording in Insta360 app; check if phone/remote had active GNSS lock during recording. |
| GPS drift over long clips | Clock drift between camera oscillator and GNSS | Apply linear drift correction $\alpha \cdot t_f + \Delta t$ via two reference anchor points. |
