"""
Export COLMAP data to RealityScan/RealityCapture/Lichtfeld Studio compatible formats.

Supports:
- COLMAP text format (images.txt, cameras.txt)
- XMP sidecar files (camera pose metadata)
- CSV trajectory format (x, y, z positions)
- Lichtfeld Studio / NerfStudio format (transforms.json)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET
import numpy as np
import shutil

logger = logging.getLogger(__name__)


class LichtfeldExporter:
    """Export COLMAP data to Lichtfeld Studio format (transforms.json)."""
    
    def __init__(self, colmap_dir: str, output_dir: str):
        self.colmap_dir = Path(colmap_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse COLMAP data (reuse RealityScanExporter logic or reimplement)
        self.exporter = RealityScanExporter(colmap_dir, output_dir)
        self.exporter.parse_colmap_text()
        
    def export(self, images_dir: str, fix_rotation: bool = True) -> bool:
        """
        Export to Lichtfeld Studio format.
        
        Args:
            images_dir: Path to source images
            fix_rotation: Apply rotation fix for 360 images (standard for Lichtfeld)
            
        Returns:
            True if successful
        """
        try:
            # Prepare output structure
            images_out_dir = self.output_dir / "images"
            images_out_dir.mkdir(exist_ok=True)
            
            frames = []
            
            # 1. Process Frames
            for image_id, image_data in self.exporter.images.items():
                name = image_data['name']
                camera_id = image_data['camera_id']
                camera = self.exporter.cameras.get(camera_id)
                
                if not camera:
                    continue
                
                # Copy image
                src_image = Path(images_dir) / name
                if not src_image.exists():
                    logger.warning(f"Image not found: {src_image}")
                    continue
                    
                dst_image = images_out_dir / name
                if not dst_image.exists():
                    try:
                        shutil.copy2(src_image, dst_image)
                    except Exception as e:
                        logger.warning(f"Could not copy image {src_image}: {e}")
                        continue
                
                # Calculate transformation matrix
                c2w = self._calculate_transform_matrix(image_data, fix_rotation)
                
                # Build frame entry
                frame = {
                    "file_path": f"images/{name}",
                    "transform_matrix": c2w.tolist()
                }
                
                # Intrinsics
                self._add_intrinsics(frame, camera)
                frames.append(frame)
            
            # 2. Process Point Cloud
            ply_path = self._export_point_cloud(fix_rotation)
            
            # 3. Write transforms.json
            transforms_data = {
                "camera_model": self._get_camera_model_string(self.exporter.cameras[1]['model']),
                "frames": frames
            }
            if ply_path:
                transforms_data["ply_file_path"] = ply_path
                
            output_json = self.output_dir / "transforms.json"
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(transforms_data, f, indent=4)
                
            logger.info(f"Lichtfeld Studio export successful: {output_json}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Lichtfeld Studio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _calculate_transform_matrix(self, image_data: Dict, fix_rotation: bool) -> np.ndarray:
        """Convert COLMAP pose to Lichtfeld Studio convention."""
        qw, qx, qy, qz = image_data['qvec']
        tx, ty, tz = image_data['tvec']
        
        # Quaternion to Rotation Matrix
        R = self._qvec2rotmat((qw, qx, qy, qz))
        t = np.array([tx, ty, tz])
        
        # Camera to World
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        
        # Apply transformations (Row swap [2, 0, 1])
        c2w = c2w[[2, 0, 1, 3], :]
        
        # OpenCV to OpenGL (y, z flip)
        c2w[:, 1:3] *= -1
        
        # Fix orientation (+90 around X)
        if fix_rotation:
            rot_x_90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
            c2w = rot_x_90 @ c2w
            
        # Pre-compensate 180 Y-rotation
        y_rot_180 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
        c2w = y_rot_180 @ c2w
        
        return c2w

    def _qvec2rotmat(self, qvec):
        """Convert quaternion to rotation matrix."""
        qw, qx, qy, qz = qvec
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
    def _add_intrinsics(self, frame: Dict, camera: Dict):
        """Add intrinsic parameters."""
        model = camera['model']
        width = camera['width']
        height = camera['height']
        params = camera['params']
        
        frame["w"] = width
        frame["h"] = height
        
        # Mapping logic from colmap_converter.py
        if model == "SPHERE" or model == "EQUIRECTANGULAR":
            frame["fl_x"] = params[0] if params else width / np.pi
            frame["fl_y"] = frame["fl_x"]
            frame["cx"] = params[1] if len(params) > 1 else width / 2.0
            frame["cy"] = params[2] if len(params) > 2 else height / 2.0
        else:
            # Assuming PINHOLE-like or OPENCV
            # PARAMS for PINHOLE: f, cx, cy
            # PARAMS for OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            # PARAMS for SIMPLE_RADIAL: f, cx, cy, k
            if len(params) >= 3:
                frame["fl_x"] = params[0]
                frame["fl_y"] = params[1] if len(params) > 3 else params[0] # Approximate for simple models
                frame["cx"] = params[2] if len(params) > 2 else width / 2.0
                frame["cy"] = params[3] if len(params) > 3 else height / 2.0
            else:
                 # Fallback
                 frame["fl_x"] = params[0] if params else width
                 frame["fl_y"] = params[0] if params else width
                 frame["cx"] = width / 2.0
                 frame["cy"] = height / 2.0

    def _get_camera_model_string(self, model: str) -> str:
        """Map COLMAP model to Lichtfeld string."""
        if model == "SPHERE": return "EQUIRECTANGULAR"
        if model in ["SIMPLE_PINHOLE", "PINHOLE"]: return "PINHOLE"
        if "FISHEYE" in model: return "OPENCV_FISHEYE"
        return "OPENCV"

    def _export_point_cloud(self, fix_rotation: bool) -> Optional[str]:
        """Export point cloud to PLY."""
        if not self.exporter.points3d:
            return None
            
        points_np = []
        colors_np = []
        
        for p in self.exporter.points3d.values():
            points_np.append(p['xyz'])
            colors_np.append(p['rgb'])
            
        points_np = np.array(points_np, dtype=np.float64)
        colors_np = np.array(colors_np, dtype=np.uint8)
        
        # Apply transformation to points
        # 1. Row swap
        points_np = points_np[:, [2, 0, 1]]
        
        # 2. Fix rotation (+90 around X)
        if fix_rotation:
             rot_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
             points_np = points_np @ rot_x_90.T # Right multiply for row vectors
             
        # 3. Y-axis inversion? Code says "applied_transform[:3, :3] = rot_x_90 @ applied_transform[:3, :3]"
        # The logic in colmap_converter.py:
        # applied_transform = np.eye(4)[:3, :]
        # applied_transform = applied_transform[np.array([2, 0, 1]), :]
        # if fix_rotation: rot_x_90... applied_transform...
        # points = einsum(T, points) + t
        # Basically transforms the coordinate system (basis vectors)
        
        ply_file = self.output_dir / "pointcloud.ply"
        
        with open(ply_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_np)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points_np)):
                x, y, z = points_np[i]
                r, g, b = colors_np[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
                
        return "pointcloud.ply"


class RealityScanExporter:
    """Export COLMAP data to RealityScan/RealityCapture compatible formats."""
    
    def __init__(self, colmap_dir: str, output_dir: str, perspective_dir: Optional[str] = None, 
                 camera_config: Optional[List[Dict]] = None):
        """
        Initialize exporter.
        
        Args:
            colmap_dir: Path to COLMAP sparse reconstruction (contains images.txt)
            output_dir: Path to output directory for export files
            perspective_dir: Path to perspective images directory (if exporting for perspective images)
            camera_config: List of camera configurations (yaw, pitch, roll, fov)
        """
        self.colmap_dir = Path(colmap_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perspective images directory (for RealityScan export)
        self.perspective_dir = Path(perspective_dir) if perspective_dir else None
        
        # Camera configuration for splits
        self.camera_config = camera_config
        
        # Parse COLMAP data
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        
        # Mapping from equirect frames to perspective images
        self.perspective_mapping = {}
        
    def parse_colmap_text(self) -> bool:
        """
        Parse COLMAP text format files (images.txt, cameras.txt, points3D.txt).
        
        Returns:
            True if parsing successful, False otherwise
        """
        try:
            # Parse cameras.txt
            cameras_file = self.colmap_dir / "cameras.txt"
            if cameras_file.exists():
                self.cameras = self._parse_cameras_txt(cameras_file)
                logger.info(f"Parsed {len(self.cameras)} cameras from {cameras_file}")
            else:
                logger.warning(f"cameras.txt not found: {cameras_file}")
            
            # Parse images.txt
            images_file = self.colmap_dir / "images.txt"
            if images_file.exists():
                self.images = self._parse_images_txt(images_file)
                logger.info(f"Parsed {len(self.images)} images from {images_file}")
            else:
                logger.error(f"images.txt not found: {images_file}")
                return False
            
            # Parse points3D.txt (optional)
            points_file = self.colmap_dir / "points3D.txt"
            if points_file.exists():
                self.points3d = self._parse_points3d_txt(points_file)
                logger.info(f"Parsed {len(self.points3d)} 3D points from {points_file}")
            
            return len(self.images) > 0
            
        except Exception as e:
            logger.error(f"Error parsing COLMAP files: {e}")
            return False
    
    def build_perspective_mapping(self, perspective_dir: Optional[str] = None) -> bool:
        """
        Build mapping from equirect frames to perspective images.
        
        Example mapping:
          0.png → frame_00000_cam_00.png, frame_00000_cam_01.png, ..., frame_00000_cam_07.png
          4.png → frame_00001_cam_00.png, frame_00001_cam_01.png, ..., frame_00001_cam_07.png
          
        Args:
            perspective_dir: Path to perspective images directory
            
        Returns:
            True if mapping successful, False otherwise
        """
        if perspective_dir:
            self.perspective_dir = Path(perspective_dir)
        
        if not self.perspective_dir or not self.perspective_dir.exists():
            logger.warning(f"Perspective directory not found: {self.perspective_dir}")
            return False
        
        try:
            # Get all perspective images
            perspective_images = sorted(self.perspective_dir.glob('frame_*_cam_*.png'))
            if not perspective_images:
                logger.warning(f"No perspective images found in {self.perspective_dir}")
                return False
            
            # Get equirect frames in order
            equirect_frames = sorted([img['name'] for img in self.images.values()])
            
            # Build mapping: frame index → perspective images
            for idx, equirect_name in enumerate(equirect_frames):
                # Find all perspective images for this frame index
                pattern = f"frame_{idx:05d}_cam_*.png"
                matching_images = sorted(self.perspective_dir.glob(pattern))
                
                if matching_images:
                    self.perspective_mapping[equirect_name] = [img.name for img in matching_images]
                    logger.debug(f"Mapped {equirect_name} → {len(matching_images)} perspective images")
            
            logger.info(f"Built perspective mapping: {len(equirect_frames)} equirect → "
                       f"{len(perspective_images)} perspective images")
            return len(self.perspective_mapping) > 0
            
        except Exception as e:
            logger.error(f"Error building perspective mapping: {e}")
            return False
    
    def _parse_cameras_txt(self, file_path: Path) -> Dict:
        """Parse COLMAP cameras.txt format."""
        cameras = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 4:
                    continue
                
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
        
        return cameras
    
    def _parse_images_txt(self, file_path: Path) -> Dict:
        """
        Parse COLMAP images.txt format.
        
        Format:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] as (X, Y, POINT3D_ID)
        """
        images = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 10:
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Convert quaternion + translation to camera pose
            images[image_id] = {
                'name': name,
                'camera_id': camera_id,
                'qvec': (qw, qx, qy, qz),
                'tvec': (tx, ty, tz),
                'position': self._compute_camera_center(qw, qx, qy, qz, tx, ty, tz)
            }
            
            # Skip next line (POINTS2D)
            if i < len(lines):
                i += 1
        
        return images
    
    def _parse_points3d_txt(self, file_path: Path) -> Dict:
        """Parse COLMAP points3D.txt format."""
        points = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    continue
                
                point_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                error = float(parts[7])
                
                points[point_id] = {
                    'xyz': (x, y, z),
                    'rgb': (r, g, b),
                    'error': error
                }
        
        return points
    
    def _compute_camera_center(self, qw: float, qx: float, qy: float, qz: float, 
                               tx: float, ty: float, tz: float) -> Tuple[float, float, float]:
        """
        Compute camera center from quaternion rotation and translation.
        
        Camera center C = -R^T * t
        where R is rotation matrix from quaternion (qw, qx, qy, qz).
        """
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
        
        # Compute camera center: C = -R^T * t
        cx = -(R[0][0] * tx + R[1][0] * ty + R[2][0] * tz)
        cy = -(R[0][1] * tx + R[1][1] * ty + R[2][1] * tz)
        cz = -(R[0][2] * tx + R[1][2] * ty + R[2][2] * tz)
        
        return (cx, cy, cz)
    
    def _quaternion_to_rotation_matrix(self, qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
        """Convert quaternion to 3x3 rotation matrix."""
        R = [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy)
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx)
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2)
            ]
        ]
        return R
    
    def _quaternion_to_euler(self, qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw) in degrees.
        
        Returns:
            (roll, pitch, yaw) in degrees
        """
        import math
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
    def export_colmap_text(self, include_points3d: bool = True) -> bool:
        """
        Export COLMAP text format (images.txt, cameras.txt, points3D.txt).
        
        This is a direct copy of existing COLMAP files for compatibility.
        
        Args:
            include_points3d: Whether to copy points3D.txt
            
        Returns:
            True if export successful
        """
        try:
            import shutil
            
            # Special handling for perspective export
            if self.perspective_dir and self.camera_config and self.perspective_mapping:
                logger.info("Generating COLMAP files for perspective images...")
                return self._export_perspective_colmap_text(include_points3d)
            
            # Copy cameras.txt
            src_cameras = self.colmap_dir / "cameras.txt"
            dst_cameras = self.output_dir / "cameras.txt"
            if src_cameras.exists():
                shutil.copy2(src_cameras, dst_cameras)
                logger.info(f"Exported: {dst_cameras}")
            
            # Copy images.txt
            src_images = self.colmap_dir / "images.txt"
            dst_images = self.output_dir / "images.txt"
            if src_images.exists():
                shutil.copy2(src_images, dst_images)
                logger.info(f"Exported: {dst_images}")
            
            # Copy points3D.txt (optional)
            if include_points3d:
                src_points = self.colmap_dir / "points3D.txt"
                dst_points = self.output_dir / "points3D.txt"
                if src_points.exists():
                    shutil.copy2(src_points, dst_points)
                    logger.info(f"Exported: {dst_points}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting COLMAP text format: {e}")
            return False
    
    def export_csv_trajectory(self, images_dir: Optional[str] = None, use_perspective: bool = False) -> bool:
        """
        Export CSV trajectory format: Name, X, Y, Z
        
        Compatible with RealityScan/RealityCapture import.
        
        Args:
            images_dir: Optional path to images directory (for absolute paths)
            use_perspective: If True, export for perspective images (RealityScan compatible)
            
        Returns:
            True if export successful
        """
        try:
            csv_file = self.output_dir / "trajectory.csv"
            
            with open(csv_file, 'w') as f:
                # Header
                f.write("Name,X,Y,Z\n")
                
                # Export for perspective images
                if use_perspective and self.perspective_mapping:
                    count = 0
                    for image_id in sorted(self.images.keys()):
                        image_data = self.images[image_id]
                        equirect_name = image_data['name']
                        cx, cy, cz = image_data['position']
                        
                        # Get all perspective images for this equirect frame
                        perspective_images = self.perspective_mapping.get(equirect_name, [])
                        
                        # Write same pose for all perspective images
                        for persp_name in perspective_images:
                            # Use perspective directory if available, otherwise fall back
                            if self.perspective_dir:
                                name = str(self.perspective_dir / persp_name)
                            elif images_dir:
                                # WARNING: This uses the provided images_dir which might be Stage 1 dir.
                                # But if perspective_dir is None, we might not have a choice.
                                # However, usually perspective_dir IS set if use_perspective is True.
                                name = str(Path(images_dir) / persp_name)
                            else:
                                name = persp_name
                            
                            f.write(f"{name},{cx:.6f},{cy:.6f},{cz:.6f}\n")
                            count += 1
                    
                    logger.info(f"Exported CSV trajectory: {csv_file} ({count} perspective cameras)")
                
                # Export for equirect images (original behavior)
                else:
                    for image_id in sorted(self.images.keys()):
                        image_data = self.images[image_id]
                        name = image_data['name']
                        cx, cy, cz = image_data['position']
                        
                        # Optionally use absolute path
                        if images_dir:
                            name = str(Path(images_dir) / name)
                        
                        f.write(f"{name},{cx:.6f},{cy:.6f},{cz:.6f}\n")
                    
                    logger.info(f"Exported CSV trajectory: {csv_file} ({len(self.images)} cameras)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting CSV trajectory: {e}")
            return False
    
    def export_xmp_sidecars(self, images_dir: str, use_perspective: bool = False) -> bool:
        """
        Export XMP sidecar files with camera pose metadata.
        
        Creates .xmp files alongside images with:
        - Camera position (x, y, z)
        - Camera orientation (quaternion or Euler angles)
        - Camera model metadata
        
        Args:
            images_dir: Path to images directory
            use_perspective: If True, export for perspective images (RealityScan compatible)
            
        Returns:
            True if export successful
        """
        try:
            images_path = Path(images_dir)
            if not images_path.exists():
                logger.error(f"Images directory not found: {images_dir}")
                return False
            
            xmp_count = 0
            
            # Export for perspective images
            if use_perspective and self.perspective_mapping:
                for image_id in sorted(self.images.keys()):
                    image_data = self.images[image_id]
                    equirect_name = image_data['name']
                    
                    # Get all perspective images for this equirect frame
                    perspective_images = self.perspective_mapping.get(equirect_name, [])
                    
                    # Create XMP for each perspective image with same pose
                    for persp_name in perspective_images:
                        persp_file = images_path / persp_name
                        
                        if not persp_file.exists():
                            logger.debug(f"Perspective image not found: {persp_file}")
                            continue
                        
                        # Create XMP sidecar
                        xmp_file = persp_file.with_suffix('.xmp')
                        if self._create_xmp_sidecar(xmp_file, image_data):
                            xmp_count += 1
                
                logger.info(f"Exported {xmp_count} XMP sidecar files to {images_dir} (perspective images)")
            
            # Export for equirect images (original behavior)
            else:
                for image_id in sorted(self.images.keys()):
                    image_data = self.images[image_id]
                    image_name = image_data['name']
                    image_file = images_path / image_name
                    
                    if not image_file.exists():
                        logger.warning(f"Image file not found: {image_file}")
                        continue
                    
                    # Create XMP sidecar
                    xmp_file = image_file.with_suffix('.xmp')
                    if self._create_xmp_sidecar(xmp_file, image_data):
                        xmp_count += 1
                
                logger.info(f"Exported {xmp_count} XMP sidecar files to {images_dir}")
            
            return xmp_count > 0
            
        except Exception as e:
            logger.error(f"Error exporting XMP sidecars: {e}")
            return False
    
    def _create_xmp_sidecar(self, xmp_file: Path, image_data: Dict) -> bool:
        """
        Create XMP sidecar file with camera pose metadata.
        
        Uses RealityCapture schema (xcr) for compatibility.
        """
        try:
            # Extract data
            qw, qx, qy, qz = image_data['qvec']
            cx, cy, cz = image_data['position']
            
            # Convert quaternion to rotation matrix (row-major)
            # R = 
            # [ 1-2y^2-2z^2,   2xy-2zw,    2xz+2yw ]
            # [ 2xy+2zw,       1-2x^2-2z^2, 2yz-2xw ]
            # [ 2xz-2yw,       2yz+2xw,    1-2x^2-2y^2 ]
            
            r00 = 1 - 2*qy*qy - 2*qz*qz
            r01 = 2*qx*qy - 2*qz*qw
            r02 = 2*qx*qz + 2*qy*qw
            
            r10 = 2*qx*qy + 2*qz*qw
            r11 = 1 - 2*qx*qx - 2*qz*qz
            r12 = 2*qy*qz - 2*qx*qw
            
            r20 = 2*qx*qz - 2*qy*qw
            r21 = 2*qy*qz + 2*qx*qw
            r22 = 1 - 2*qx*qx - 2*qy*qy
            
            r_str = f"{r00:.6f} {r01:.6f} {r02:.6f} {r10:.6f} {r11:.6f} {r12:.6f} {r20:.6f} {r21:.6f} {r22:.6f}"
            
            # Create XMP structure (RealityCapture format)
            xmp_template = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      
      <!-- RealityCapture Schema -->
      <xcr:Position>{cx:.6f} {cy:.6f} {cz:.6f}</xcr:Position>
      <xcr:Rotation>{r_str}</xcr:Rotation>
      <xcr:PosePrior>locked</xcr:PosePrior>
      
      <!-- Source -->
      <xmp:CreatorTool>360FrameTools RealityScan Output</xmp:CreatorTool>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
            
            with open(xmp_file, 'w', encoding='utf-8') as f:
                f.write(xmp_template)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating XMP sidecar {xmp_file}: {e}")
            return False
    
    def export_json_metadata(self) -> bool:
        """
        Export camera poses as JSON (custom format for easy parsing).
        
        Returns:
            True if export successful
        """
        try:
            json_file = self.output_dir / "camera_poses.json"
            
            # Convert to JSON-serializable format
            export_data = {
                'cameras': self.cameras,
                'images': {}
            }
            
            for image_id, image_data in self.images.items():
                qw, qx, qy, qz = image_data['qvec']
                roll, pitch, yaw = self._quaternion_to_euler(qw, qx, qy, qz)
                cx, cy, cz = image_data['position']
                
                export_data['images'][image_data['name']] = {
                    'image_id': image_id,
                    'camera_id': image_data['camera_id'],
                    'position': {'x': cx, 'y': cy, 'z': cz},
                    'quaternion': {'w': qw, 'x': qx, 'y': qy, 'z': qz},
                    'euler_angles': {'roll': roll, 'pitch': pitch, 'yaw': yaw}
                }
            
            with open(json_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported JSON metadata: {json_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting JSON metadata: {e}")
            return False
    
    def _export_perspective_colmap_text(self, include_points3d: bool = True) -> bool:
        """
        Generate COLMAP text files for perspective images.
        """
        try:
            # 1. Generate cameras.txt (PINHOLE model)
            # We assume all perspective images share the same intrinsics (FOV)
            # Use the first camera config to determine FOV/Width/Height
            # Wait, width/height might vary if not checked, but usually batch uses same output size.
            
            # Use default or configured output size if available, otherwise guess? 
            # We don't have output_width/height stored in Exporter...
            # But we can check one image file if needed, or assume defaults.
            # Ideally BatchOrchestrator passed this info, but current interface doesn't.
            # Let's peek at the first perspective file to get dimensions.
            
            first_persp = next(self.perspective_dir.glob('*.png'), None)
            if not first_persp:
                logger.error("No perspective images found to determine dimensions")
                return False
                
            try:
                import cv2
                img = cv2.imread(str(first_persp))
                height, width = img.shape[:2]
            except:
                width, height = 1920, 1080 # Fallback
                
            # Get FOV from config (assume homogenous)
            fov = 90
            if self.camera_config:
                fov = self.camera_config[0].get('fov', 90)
                
            # Calculate focal length: f = (w/2) / tan(fov/2)
            f = (width / 2.0) / np.tan(np.radians(fov / 2.0))
            cx = width / 2.0
            cy = height / 2.0
            
            # Write cameras.txt
            # ID MODEL WIDTH HEIGHT PARAMS
            # ID=1, MODEL=PINHOLE, PARAMS=f, cx, cy, k (SIMPLE_RADIAL?) or f, cx, cy (PINHOLE)
            # PINHOLE: fx, fy, cx, cy
            dst_cameras = self.output_dir / "cameras.txt"
            with open(dst_cameras, 'w') as f_cam:
                f_cam.write("# Camera list with one line of data per camera.\n")
                f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                f_cam.write(f"1 PINHOLE {width} {height} {f} {f} {cx} {cy}\n")
                
            logger.info(f"Generated cameras.txt for perspective (PINHOLE, {width}x{height}, f={f:.1f})")
            
            # 2. Generate images.txt
            dst_images = self.output_dir / "images.txt"
            
            # Pre-calculate rotation matrices for each split camera
            # relative to the equirectangular frame.
            split_rotations = {} # cam_idx -> Matrix (3x3)
            
            for idx, cam_conf in enumerate(self.camera_config):
                y = cam_conf.get('yaw', 0)
                p = cam_conf.get('pitch', 0)
                r = cam_conf.get('roll', 0)
                
                # Convert to rotation matrix
                # Order of operations matters: usually R = Rz(yaw) * Ry(pitch) * Rx(roll)
                R_split = self._euler_to_rotation_matrix(y, p, r)
                split_rotations[idx] = R_split

            new_image_id = 1
            with open(dst_images, 'w') as f_img:
                f_img.write("# Image list with two lines of data per image.\n")
                f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
                
                # Iterate equirect images
                for eq_id in sorted(self.images.keys()):
                    eq_data = self.images[eq_id]
                    eq_name = eq_data['name']
                    eq_qvec = eq_data['qvec'] # (qw, qx, qy, qz)
                    eq_tvec = eq_data['tvec']
                    
                    # Convert Equirect quaternion to Rotation Matrix
                    R_eq = self._quaternion_to_rotation_matrix(*eq_qvec)
                    R_eq = np.array(R_eq)
                    t_eq = np.array(eq_tvec)

                    # Get mapped perspective images
                    persp_images = self.perspective_mapping.get(eq_name, [])
                    
                    for p_name in persp_images:
                        # Extract cam_idx from filename "frame_XXXXX_cam_YY.png"
                        try:
                            # Assume standard naming convention from Orchestrator
                            parts = p_name.split('_cam_')
                            if len(parts) < 2:
                                continue # Unknown format
                            cam_idx = int(parts[1].split('.')[0])
                        except:
                            logger.warning(f"Could not parse camera index from {p_name}")
                            continue
                            
                        if cam_idx not in split_rotations:
                            continue
                            
                        R_rel = split_rotations[cam_idx] # Rotation of virtual camera relative to equirect center
                        
                        # Composition:
                        # The virtual camera defines a view direction.
                        # We want the World-To-Camera transform for this new camera.
                        # R_persp_to_world = R_eq_to_world * R_persp_to_eq
                        # But we store R_world_to_cam.
                        # R_world_to_persp = R_eq_to_persp * R_world_to_eq
                        # R_world_to_persp = (R_persp_to_eq)^T * R_world_to_eq
                        
                        # WAIT. R_rel we computed from Euler angles is typically "Object Rotation" (Active).
                        # Or it's the rotation of the camera frame axes.
                        # If I rotate camera by +90 Yaw, its axes change.
                        
                        # Let's assume R_rel transforms a vector from Perspective Frame to Equirect Frame.
                        # v_eq = R_rel * v_persp
                        # We have v_cam = R_world_to_cam * v_world
                        # v_eq = R_world_to_eq * v_world
                        # v_eq = R_rel * v_persp => v_persp = R_rel^T * v_eq
                        # v_persp = R_rel^T * (R_world_to_eq * v_world)
                        # So R_world_to_persp = R_rel^T * R_world_to_eq
                        
                        # Let's verify rotation matrix transpose behavior.
                        # _euler_to_rotation_matrix should return the matrix that rotates the point?
                        # Or rotates the frame?
                        # Usually euler angles build a rotation matrix R.
                        # If we rotate the camera frame by R_rel (yaw,pitch,roll),
                        # then R_rel represents the orientation of the camera frame w.r.t the parent frame.
                        # So Vectors in Camera Frame: v_c = R_rel^T * v_p (if p is parent)
                        # Yes, R_world_to_persp = R_rel^T * R_world_to_eq.
                        
                        R_persp = R_rel.T @ R_eq
                        
                        # Translation:
                        # The virtual cameras are at the center of the sphere (nodal point).
                        # So T is derived from Rot center?
                        # C = -R^T * T  => T = -R * C
                        # The Center C is the same for all splits (Position in World).
                        # C_eq = -R_eq^T * t_eq
                        # C_persp = C_eq (Virtual splits, 0 baseline)
                        # T_persp = -R_persp * C_persp
                        #         = -R_persp * (-R_eq^T * t_eq)
                        #         = R_persp * R_eq^T * t_eq
                        #         = (R_rel^T * R_eq) * R_eq^T * t_eq
                        #         = R_rel^T * (R_eq * R_eq^T) * t_eq
                        #         = R_rel^T * t_eq
                        
                        t_persp = R_rel.T @ t_eq
                        
                        # Convert back to quaternion
                        q_persp = self._matrix_to_quaternion(R_persp) # (qw, qx, qy, qz)
                        
                        # Write line
                        f_img.write(f"{new_image_id} {q_persp[0]:.7f} {q_persp[1]:.7f} {q_persp[2]:.7f} {q_persp[3]:.7f} "
                                   f"{t_persp[0]:.7f} {t_persp[1]:.7f} {t_persp[2]:.7f} 1 {p_name}\n")
                        f_img.write("\n") # Empty POINTS2D line
                        
                        new_image_id += 1
                        
            # 3. Copy points3D.txt (optional)
            if include_points3d:
                src_points = self.colmap_dir / "points3D.txt"
                dst_points = self.output_dir / "points3D.txt"
                if src_points.exists():
                    shutil.copy2(src_points, dst_points)
                    
            logger.info("Generated perspective COLMAP files successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating perspective COLMAP files: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _euler_to_rotation_matrix(self, yaw_deg, pitch_deg, roll_deg):
        """
        Convert Euler angles (degrees) to 3x3 Rotation Matrix.
        Order: Yaw(Y) -> Pitch(X) -> Roll(Z)?? 
        Common 360 viewer convention:
        Yaw rotates around Y (vertical).
        Pitch rotates around X (right).
        Roll rotates around Z (forward).
        """
        y = np.radians(yaw_deg)
        p = np.radians(pitch_deg)
        r = np.radians(roll_deg)
        
        # Ry (Yaw)
        Ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])
        
        # Rx (Pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p), np.cos(p)]
        ])
        
        # Rz (Roll)
        Rz = np.array([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ])
        
        # Combined Rotation: R = Ry * Rx * Rz (Order depends on convention!)
        # Testing:
        # If I look right (Yaw 90), then point (0,0,1) should map to (1,0,0) in world?
        # If camera rotates Right, the world rotates Left relative to camera.
        # This matrix represents "Rotation of the Frame".
        # Let's stick with Ry @ Rx @ Rz for now.
        return Ry @ Rx @ Rz

    def _matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # Robust implementation
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2,1] - R[1,2]) * s
            qy = (R[0,2] - R[2,0]) * s
            qz = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                qw = (R[2,1] - R[1,2]) / s
                qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s
                qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                qw = (R[0,2] - R[2,0]) / s
                qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s
                qz = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                qw = (R[1,0] - R[0,1]) / s
                qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s
                qz = 0.25 * s
        return np.array([qw, qx, qy, qz])

    def export_all(self, images_dir: str, perspective_dir: Optional[str] = None,
                   include_xmp: bool = True, include_csv: bool = True, 
                   include_json: bool = True) -> Dict[str, bool]:
        """
        Export all formats at once.
        
        Args:
            images_dir: Path to images directory (equirect or perspective)
            perspective_dir: Path to perspective images directory (if exporting for RealityScan)
            include_xmp: Whether to export XMP sidecars
            include_csv: Whether to export CSV trajectory
            include_json: Whether to export JSON metadata
            
        Returns:
            Dictionary with export results for each format
        """
        results = {}
        
        # Parse COLMAP data first
        if not self.parse_colmap_text():
            logger.error("Failed to parse COLMAP data")
            return {'error': 'Failed to parse COLMAP data'}
        
        # Build perspective mapping if perspective directory provided
        use_perspective = False
        if perspective_dir:
            if self.build_perspective_mapping(perspective_dir):
                use_perspective = True
                logger.info("Using perspective images for export (RealityScan compatible)")
            else:
                logger.warning("Could not build perspective mapping, falling back to equirect export")
        
        # Export COLMAP text format (always)
        results['colmap_txt'] = self.export_colmap_text()
        
        # Export XMP sidecars (to perspective directory if available)
        if include_xmp:
            target_dir = perspective_dir if use_perspective else images_dir
            results['xmp_sidecars'] = self.export_xmp_sidecars(target_dir, use_perspective=use_perspective)
        
        # Export CSV trajectory
        if include_csv:
            target_dir = perspective_dir if use_perspective else images_dir
            results['csv_trajectory'] = self.export_csv_trajectory(target_dir, use_perspective=use_perspective)
        
        # Export JSON metadata
        if include_json:
            results['json_metadata'] = self.export_json_metadata()
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info(f"Export complete: {successful}/{total} formats successful")
        
        return results


def export_realityscan_formats(colmap_dir: str, images_dir: str, output_dir: str,
                               perspective_dir: Optional[str] = None,
                               include_xmp: bool = True, include_csv: bool = True,
                               include_json: bool = True,
                               camera_config: Optional[List[Dict]] = None) -> Dict[str, bool]:
    """
    Convenience function to export COLMAP data to RealityScan/RealityCapture formats.
    
    Args:
        colmap_dir: Path to COLMAP sparse reconstruction directory
        images_dir: Path to images directory (equirect frames)
        output_dir: Path to output directory for export files
        perspective_dir: Path to perspective images directory (for RealityScan)
        include_xmp: Whether to export XMP sidecars
        include_csv: Whether to export CSV trajectory
        include_json: Whether to export JSON metadata
        camera_config: List of camera configurations (yaw, pitch, roll, fov) used for splitting
        
    Returns:
        Dictionary with export results
    """
    exporter = RealityScanExporter(colmap_dir, output_dir, perspective_dir, camera_config)
    return exporter.export_all(images_dir, perspective_dir, include_xmp, include_csv, include_json)
