"""
Metashape to LichtFeld Export Script

Exports Metashape camera poses and sparse point cloud to LichtFeld-compatible
transforms.json format, running directly inside Metashape.

Fully self-contained - no external dependencies required.

Author: Claude Code
Date: January 2026
Version: 3.0
"""

import Metashape
import os
import json
import numpy as np
import sys
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path
from PySide2 import QtWidgets, QtCore

# Try to import open3d for better PLY handling
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

LOG_FILE = None

def init_logging(output_dir):
    """Initialize logging to file"""
    global LOG_FILE
    log_path = os.path.join(output_dir, "lichtfeld_export.log")
    LOG_FILE = open(log_path, 'w', encoding='utf-8')
    log(f"Log file created: {log_path}")

def log(message):
    """Print message to console and log file"""
    print(message)
    sys.stdout.flush()
    if LOG_FILE:
        LOG_FILE.write(message + '\n')
        LOG_FILE.flush()

def close_logging():
    """Close log file"""
    global LOG_FILE
    if LOG_FILE:
        LOG_FILE.close()
        LOG_FILE = None


# ============================================================================
# CONFIGURATION
# ============================================================================

class ExportParams:
    """Parameters for LichtFeld export"""
    def __init__(self):
        self.output_dir = self._get_default_output_dir()
        self.fix_upside_down = True  # Apply 90° X rotation fix
        self.export_pointcloud = True
        self.all_cameras = True  # True = all aligned cameras, False = selected only
        self.save_log = True

    def _get_default_output_dir(self):
        """Get default output directory based on current project location"""
        try:
            doc = Metashape.app.document
            if doc.path:
                project_dir = os.path.dirname(doc.path)
                path = os.path.join(project_dir, "LichtFeld_Export")
                return path.replace("\\", "/")
        except:
            pass
        return os.path.expanduser("~/LichtFeld_Export").replace("\\", "/")


# ============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# ============================================================================

def transform_camera_matrix(transform_np, fix_upside_down=True):
    """
    Convert Metashape camera transform to LichtFeld/nerfstudio convention.

    Args:
        transform_np: 4x4 numpy array camera-to-world matrix from Metashape
        fix_upside_down: If True, apply additional 90° rotation to fix upside-down scene

    Returns:
        Transformed 4x4 numpy array
    """
    # Step 1: Rotate scene according to nerfstudio convention (row swap)
    transform = transform_np[[2, 0, 1, 3], :]

    # Step 2: Convert from OpenCV to OpenGL (flip Y and Z columns)
    transform[:, 1:3] *= -1

    # Step 3: Fix orientation with +90° rotation around X-axis
    if fix_upside_down:
        cos_90 = 0.0
        sin_90 = 1.0
        rot_x_pos90 = np.array([
            [1, 0, 0, 0],
            [0, cos_90, -sin_90, 0],
            [0, sin_90, cos_90, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        transform = rot_x_pos90 @ transform

    # Step 4: Pre-compensate for LichtFeld's 180° Y-rotation
    cos_pi = -1.0
    sin_pi = 0.0
    y_rot_180 = np.array([
        [cos_pi, 0, sin_pi, 0],
        [0, 1, 0, 0],
        [-sin_pi, 0, cos_pi, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    transform = y_rot_180 @ transform

    return transform


def get_applied_transform(fix_upside_down=True):
    """
    Get the 3x4 transformation matrix applied to point cloud.

    Note: LichtFeld only applies Y-rotation to cameras, not point clouds.
    So we don't need Y-rotation pre-compensation here.
    """
    # Base transform: row swap [2, 0, 1]
    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([2, 0, 1]), :]

    # Add orientation fix: +90° rotation around X
    if fix_upside_down:
        cos_90 = 0.0
        sin_90 = 1.0
        rot_x_pos90 = np.array([
            [1, 0, 0],
            [0, cos_90, -sin_90],
            [0, sin_90, cos_90]
        ], dtype=np.float64)
        applied_transform[:3, :3] = rot_x_pos90 @ applied_transform[:3, :3]

    return applied_transform


def metashape_matrix_to_numpy(ms_matrix):
    """Convert Metashape.Matrix to numpy array"""
    rows = []
    for i in range(4):
        row = [ms_matrix[i, j] for j in range(4)]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


# ============================================================================
# XML PARSING FOR COMPONENT TRANSFORMS
# ============================================================================

def parse_component_transforms_from_xml(xml_path):
    """
    Parse component transforms from Metashape XML.

    Component transforms are needed because:
    - Camera transforms in XML are in component-local coordinates
    - Component transform converts from component-local to chunk coordinates

    Args:
        xml_path: Path to Metashape cameras.xml

    Returns:
        dict: component_id -> 4x4 numpy transform matrix
    """
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    chunk = root[0]

    components = chunk.find("components")
    component_dict = {}

    if components is not None:
        for component in components.iter("component"):
            transform = component.find("transform")
            if transform is not None:
                rotation = transform.find("rotation")
                if rotation is None or rotation.text is None:
                    r = np.eye(3)
                else:
                    r = np.array([float(x) for x in rotation.text.split()]).reshape((3, 3))

                translation = transform.find("translation")
                if translation is None or translation.text is None:
                    t = np.zeros(3)
                else:
                    t = np.array([float(x) for x in translation.text.split()])

                scale = transform.find("scale")
                if scale is None or scale.text is None:
                    s = 1.0
                else:
                    s = float(scale.text)

                m = np.eye(4)
                m[:3, :3] = r
                m[:3, 3] = t / s
                component_dict[component.get("id")] = m

    return component_dict


def parse_camera_component_ids_from_xml(xml_path):
    """
    Parse camera-to-component mapping from Metashape XML.

    Args:
        xml_path: Path to Metashape cameras.xml

    Returns:
        dict: camera_label -> component_id
    """
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    chunk = root[0]
    cameras = chunk.find("cameras")

    camera_components = {}
    if cameras is not None:
        for camera in cameras.iter("camera"):
            label = camera.get("label")
            component_id = camera.get("component_id")
            if label and component_id:
                camera_components[label] = component_id

    return camera_components


def get_single_component_transform(component_dict):
    """
    Get component transform for single-component projects.

    Args:
        component_dict: Dictionary of component_id -> 4x4 transform matrix

    Returns:
        4x4 numpy array if single component with transform, None otherwise
    """
    if len(component_dict) == 0:
        return None
    elif len(component_dict) == 1:
        return list(component_dict.values())[0]
    else:
        log("WARNING: Multiple components detected. Using first component transform.")
        return list(component_dict.values())[0]


# ============================================================================
# CAMERA MODEL MAPPING
# ============================================================================

def get_camera_model_string(sensor_type):
    """Map Metashape sensor type to camera model string"""
    if sensor_type == Metashape.Sensor.Type.Frame:
        return "PINHOLE"
    elif sensor_type == Metashape.Sensor.Type.Fisheye:
        return "OPENCV_FISHEYE"
    elif sensor_type == Metashape.Sensor.Type.Spherical:
        return "EQUIRECTANGULAR"
    else:
        raise ValueError(f"Unsupported sensor type: {sensor_type}")


def get_sensor_params(sensor):
    """
    Extract sensor calibration parameters in LichtFeld format.

    Args:
        sensor: Metashape.Sensor object

    Returns:
        dict with sensor parameters
    """
    calib = sensor.calibration

    # Get image dimensions
    w = int(calib.width)
    h = int(calib.height)

    params = {
        "w": w,
        "h": h,
    }

    if sensor.type == Metashape.Sensor.Type.Spherical:
        # Spherical sensors have special handling
        params["fl_x"] = w / 2.0
        params["fl_y"] = h
        params["cx"] = w / 2.0
        params["cy"] = h / 2.0
    else:
        # Frame and Fisheye sensors
        params["fl_x"] = calib.f
        params["fl_y"] = calib.f
        # Metashape cx/cy are OFFSETS from center, convert to absolute
        params["cx"] = calib.cx + w / 2.0
        params["cy"] = calib.cy + h / 2.0

        # Distortion parameters
        params["k1"] = calib.k1
        params["k2"] = calib.k2
        params["k3"] = calib.k3
        params["k4"] = calib.k4
        params["p1"] = calib.p1
        params["p2"] = calib.p2

    return params


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_transforms_json(chunk, output_dir, params, component_dict, camera_components, progress_callback=None):
    """
    Export camera transforms to LichtFeld-compatible transforms.json.

    Args:
        chunk: Metashape.Chunk object
        output_dir: Output directory path
        params: ExportParams object
        component_dict: Dictionary of component transforms from XML
        camera_components: Dictionary of camera_label -> component_id
        progress_callback: Optional callback(value, message)

    Returns:
        dict with export statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get aligned cameras
    if params.all_cameras:
        cameras = [cam for cam in chunk.cameras if cam.transform]
    else:
        cameras = [cam for cam in chunk.cameras if cam.transform and cam.selected]

    if not cameras:
        raise Exception("No aligned cameras found")

    # Check sensor types are consistent
    sensor_types = list(set(cam.sensor.type for cam in cameras))
    if len(sensor_types) > 1:
        raise Exception("Mixed camera types are not supported. All cameras must have the same sensor type.")

    camera_model = get_camera_model_string(sensor_types[0])
    log(f"Camera model: {camera_model}")
    log(f"Processing {len(cameras)} cameras...")

    if component_dict:
        log(f"Found {len(component_dict)} component transform(s)")

    # Build frames list
    frames = []
    num_skipped = 0

    for i, camera in enumerate(cameras):
        if progress_callback and i % 10 == 0:
            progress = 10 + int((i / len(cameras)) * 50)
            progress_callback(progress, f"Processing camera {i+1}/{len(cameras)}")

        # Check for calibration
        if not camera.sensor.calibration:
            log(f"  Skipping {camera.label}: no calibration")
            num_skipped += 1
            continue

        # Get camera transform (in component-local coordinates)
        transform_np = metashape_matrix_to_numpy(camera.transform)

        # Apply component transform if present
        component_id = camera_components.get(camera.label)
        if component_id and component_id in component_dict:
            transform_np = component_dict[component_id] @ transform_np

        # Apply LichtFeld transformation
        transform_lf = transform_camera_matrix(transform_np, params.fix_upside_down)

        # Get image path
        if camera.photo and camera.photo.path:
            file_path = f"images/{os.path.basename(camera.photo.path)}"
        else:
            file_path = f"images/{camera.label}.jpg"

        # Build frame data
        frame = {
            "file_path": file_path,
            "transform_matrix": transform_lf.tolist()
        }

        # Add sensor parameters
        sensor_params = get_sensor_params(camera.sensor)
        frame.update(sensor_params)

        frames.append(frame)

    log(f"Processed {len(frames)} camera frames")
    if num_skipped > 0:
        log(f"Skipped {num_skipped} cameras")

    # Build output data
    data = {
        "camera_model": camera_model,
        "frames": frames
    }

    # Store applied transform for reference
    applied_transform = get_applied_transform(params.fix_upside_down)
    data["applied_transform"] = applied_transform.tolist()

    # Export point cloud if requested
    if params.export_pointcloud:
        if progress_callback:
            progress_callback(65, "Exporting point cloud")

        pointcloud_written = export_pointcloud(chunk, output_dir, applied_transform, component_dict)
        if pointcloud_written:
            data["ply_file_path"] = "pointcloud.ply"
    else:
        pointcloud_written = False

    # Write transforms.json
    output_json = os.path.join(output_dir, "transforms.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    log(f"\nWrote {output_json}")
    log(f"Dataset ready: {len(frames)} frames, camera_model={camera_model}")

    return {
        "num_frames": len(frames),
        "num_skipped": num_skipped,
        "camera_model": camera_model,
        "has_pointcloud": pointcloud_written
    }


def export_pointcloud(chunk, output_dir, applied_transform, component_dict):
    """
    Export and transform sparse point cloud (tie points).

    Args:
        chunk: Metashape.Chunk object
        output_dir: Output directory
        applied_transform: 3x4 LichtFeld transformation matrix
        component_dict: Dictionary of component transforms

    Returns:
        bool: True if point cloud was written successfully
    """
    if not chunk.tie_points:
        log("  No tie points available")
        return False

    tie_points = chunk.tie_points

    # Get valid points
    valid_points = [pt for pt in tie_points.points if pt.valid]
    log(f"  Processing {len(valid_points)} tie points...")

    if len(valid_points) == 0:
        log("  No valid tie points")
        return False

    # Get component transform for point cloud
    component_transform = get_single_component_transform(component_dict)
    if component_transform is not None:
        log("  Applying component transform to point cloud")

    # Extract coordinates and colors
    points = []
    colors = []

    for pt in valid_points:
        # Get point coordinates (in component-local coordinates)
        points.append([pt.coord.x, pt.coord.y, pt.coord.z])

        # Get color from track
        track = tie_points.tracks[pt.track_id]
        colors.append([track.color[0] / 255.0, track.color[1] / 255.0, track.color[2] / 255.0])

    points_np = np.array(points, dtype=np.float64)
    colors_np = np.array(colors, dtype=np.float64)

    # Step 1: Apply component transform (if present)
    if component_transform is not None:
        R_comp = component_transform[:3, :3]
        t_comp = component_transform[:3, 3]
        points_np = (R_comp @ points_np.T).T + t_comp

    # Step 2: Apply LichtFeld coordinate transform
    R = applied_transform[:3, :3]
    t = applied_transform[:3, 3]
    points_transformed = (R @ points_np.T).T + t

    # Write PLY file
    output_ply = os.path.join(output_dir, "pointcloud.ply")

    if HAS_OPEN3D:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_transformed)
        pc.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.io.write_point_cloud(output_ply, pc)
        log(f"  Wrote point cloud with {len(points_transformed)} points to {output_ply}")
    else:
        write_ply_simple(output_ply, points_transformed, colors_np)
        log(f"  Wrote point cloud with {len(points_transformed)} points")

    return True


def write_ply_simple(path, points, colors):
    """Write a simple PLY file without external dependencies"""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for pt, col in zip(points, colors):
            r = int(col[0] * 255)
            g = int(col[1] * 255)
            b = int(col[2] * 255)
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")


# ============================================================================
# UI DIALOG
# ============================================================================

class ExportDialog(QtWidgets.QDialog):
    """Dialog for configuring LichtFeld export"""

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Export to LichtFeld Format")
        self.params = None

        defaults = ExportParams()
        self._create_output_group(defaults)
        self._create_options_group(defaults)
        self._create_buttons()
        self._setup_layout()

        self.exec_()

    def _create_output_group(self, defaults):
        self.output_group = QtWidgets.QGroupBox("Output Settings")
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(8)

        self.folder_label = QtWidgets.QLabel("Output folder:")
        self.folder_path = QtWidgets.QLineEdit()
        self.folder_path.setText(defaults.output_dir)
        self.folder_path.setReadOnly(True)
        self.folder_path.setMinimumWidth(350)
        self.folder_btn = QtWidgets.QPushButton("...")
        self.folder_btn.setFixedWidth(30)
        self.folder_btn.clicked.connect(self._browse_folder)

        layout.addWidget(self.folder_label, 0, 0)
        layout.addWidget(self.folder_path, 0, 1)
        layout.addWidget(self.folder_btn, 0, 2)

        self.output_group.setLayout(layout)

    def _create_options_group(self, defaults):
        self.options_group = QtWidgets.QGroupBox("Export Options")
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(8)

        # Camera scope
        self.cameras_label = QtWidgets.QLabel("Cameras:")
        self.cameras_group = QtWidgets.QButtonGroup(self)
        self.radio_all = QtWidgets.QRadioButton("All aligned")
        self.radio_selected = QtWidgets.QRadioButton("Selected only")
        self.cameras_group.addButton(self.radio_all)
        self.cameras_group.addButton(self.radio_selected)
        self.radio_all.setChecked(defaults.all_cameras)
        self.radio_selected.setChecked(not defaults.all_cameras)

        layout.addWidget(self.cameras_label, 0, 0)
        layout.addWidget(self.radio_all, 0, 1)
        layout.addWidget(self.radio_selected, 0, 2)

        # Fix orientation checkbox
        self.fix_orientation_check = QtWidgets.QCheckBox("Fix upside-down orientation (+90° X rotation)")
        self.fix_orientation_check.setChecked(defaults.fix_upside_down)
        self.fix_orientation_check.setToolTip(
            "Apply additional rotation to fix upside-down scene.\n"
            "Usually needed for 360° captures."
        )
        layout.addWidget(self.fix_orientation_check, 1, 0, 1, 3)

        # Export point cloud checkbox
        self.pointcloud_check = QtWidgets.QCheckBox("Export sparse point cloud")
        self.pointcloud_check.setChecked(defaults.export_pointcloud)
        self.pointcloud_check.setToolTip(
            "Export tie points as transformed PLY file.\n"
            "The point cloud will have the same coordinate transform applied as the cameras."
        )
        layout.addWidget(self.pointcloud_check, 2, 0, 1, 3)

        # Save log checkbox
        self.log_check = QtWidgets.QCheckBox("Save log file")
        self.log_check.setChecked(defaults.save_log)
        layout.addWidget(self.log_check, 3, 0, 1, 3)

        self.options_group.setLayout(layout)

    def _create_buttons(self):
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)

        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_export.setFixedWidth(100)
        self.btn_export.clicked.connect(self._run_export)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.clicked.connect(self.reject)

    def _setup_layout(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(12)

        layout.addWidget(self.output_group)
        layout.addWidget(self.options_group)

        # Info label
        info_label = QtWidgets.QLabel(
            "Exports camera poses and point cloud to LichtFeld-compatible transforms.json format.\n"
            "Supports Frame (pinhole), Fisheye, and Spherical (equirectangular) camera models."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info_label)

        # Button row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.progress_bar)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _browse_folder(self):
        folder = Metashape.app.getExistingDirectory("Select output folder")
        if folder:
            self.folder_path.setText(folder.replace("\\", "/"))

    def _collect_params(self):
        params = ExportParams()
        params.output_dir = self.folder_path.text()
        params.all_cameras = self.radio_all.isChecked()
        params.fix_upside_down = self.fix_orientation_check.isChecked()
        params.export_pointcloud = self.pointcloud_check.isChecked()
        params.save_log = self.log_check.isChecked()
        return params

    def _update_progress(self, value, message=""):
        self.progress_bar.setValue(int(value))
        if message:
            self.progress_bar.setFormat(f"{message} - %p%")
        QtWidgets.QApplication.processEvents()

    def _run_export(self):
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.params = self._collect_params()

        try:
            run_export(self.params, self._update_progress)
            self.accept()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"{str(e)}\n\n{error_details}")
            self.btn_export.setEnabled(True)
            self.btn_cancel.setEnabled(True)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_export(params, progress_callback=None):
    """Main export execution flow"""

    def update_progress(value, message=""):
        if progress_callback:
            progress_callback(value, message)

    os.makedirs(params.output_dir, exist_ok=True)

    if params.save_log:
        init_logging(params.output_dir)

    log("=" * 60)
    log("Metashape to LichtFeld Export")
    log("=" * 60)

    doc = Metashape.app.document
    chunk = doc.chunk

    if not chunk:
        raise Exception("No active chunk")

    update_progress(5, "Validating")

    # Validate
    aligned_cameras = [cam for cam in chunk.cameras if cam.transform]
    if not aligned_cameras:
        raise Exception("No aligned cameras found in chunk")

    log(f"Chunk: {chunk.label}")
    log(f"Aligned cameras: {len(aligned_cameras)}")
    log(f"Fix upside-down: {params.fix_upside_down}")
    log(f"Export point cloud: {params.export_pointcloud}")

    # Export XML temporarily to get component transforms
    update_progress(8, "Reading component transforms")
    temp_xml = os.path.join(params.output_dir, "_temp_cameras.xml")
    try:
        chunk.exportCameras(
            path=temp_xml,
            format=Metashape.CamerasFormatXML
        )
        component_dict = parse_component_transforms_from_xml(temp_xml)
        camera_components = parse_camera_component_ids_from_xml(temp_xml)
    finally:
        if os.path.exists(temp_xml):
            os.remove(temp_xml)

    update_progress(10, "Exporting")

    # Export
    result = export_transforms_json(
        chunk, params.output_dir, params,
        component_dict, camera_components,
        update_progress
    )

    update_progress(95, "Finalizing")

    # Summary
    log("\n" + "=" * 60)
    log("EXPORT COMPLETE")
    log("=" * 60)
    log(f"Frames exported:    {result['num_frames']}")
    log(f"Frames skipped:     {result['num_skipped']}")
    log(f"Camera model:       {result['camera_model']}")
    log(f"Point cloud:        {'Yes' if result['has_pointcloud'] else 'No'}")
    log(f"Output directory:   {params.output_dir}")
    log("=" * 60)

    update_progress(100, "Complete")

    if params.save_log:
        close_logging()

    Metashape.app.messageBox(
        f"Export complete!\n\n"
        f"Frames: {result['num_frames']}\n"
        f"Camera model: {result['camera_model']}\n"
        f"Point cloud: {'Yes' if result['has_pointcloud'] else 'No'}\n\n"
        f"Output: {params.output_dir}"
    )


def show_export_dialog():
    """Entry point for menu"""
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()

    try:
        dlg = ExportDialog(parent)
    except Exception as e:
        import traceback
        Metashape.app.messageBox(f"Error: {str(e)}\n\n{traceback.format_exc()}")


# Register menu item
label = "Scripts/Export to LichtFeld Format"
Metashape.app.addMenuItem(label, show_export_dialog)
