"""
Equirectangular to Pinhole (E2P) Transformation Engine
Converts 360-degree equirectangular images to perspective pinhole camera views.
Based on spherical coordinate mapping and camera projection mathematics.

OPTIMIZATION NOTES:
- OpenCV cv2.remap() is ESSENTIAL and CANNOT be replaced
- It performs fast bilinear interpolation for geometric transforms
- No pure Python/NumPy alternative exists with comparable performance
- This is the CORE function for perspective projection in Stage 2

Ported from 360toFrame for 360FrameTools unified application.
"""

import numpy as np
import cv2  # REQUIRED: cv2.remap() for fast geometric transformation
import math
import logging

logger = logging.getLogger(__name__)


class E2PTransform:
    """Equirectangular to Pinhole transformation class with caching"""
    
    def __init__(self):
        self.cache = {}  # Cache for transformation maps
        
    def equirect_to_pinhole(self, equirect_img, yaw=0, pitch=0, roll=0, 
                           h_fov=90, v_fov=None, output_width=1920, output_height=1080):
        """
        Convert equirectangular image to pinhole perspective view
        
        Args:
            equirect_img: Input equirectangular image (numpy array)
            yaw: Horizontal rotation in degrees (-180 to 180)
            pitch: Vertical tilt in degrees (-90 to 90)
            roll: Roll rotation in degrees (-180 to 180)
            h_fov: Horizontal field of view in degrees
            v_fov: Vertical field of view in degrees (auto-calculated if None)
            output_width: Output image width
            output_height: Output image height
            
        Returns:
            Perspective view image (numpy array)
        """
        
        # Auto-calculate vertical FOV if not provided
        if v_fov is None:
            v_fov = h_fov * output_height / output_width
            
        # Create cache key for transformation map
        cache_key = (yaw, pitch, roll, h_fov, v_fov, output_width, output_height, 
                    equirect_img.shape[0], equirect_img.shape[1])
        
        # Check if transformation map is cached
        if cache_key in self.cache:
            map_x, map_y = self.cache[cache_key]
            logger.debug(f"Using cached transformation map for yaw={yaw}, pitch={pitch}")
        else:
            # Generate transformation map
            logger.debug(f"Generating new transformation map for yaw={yaw}, pitch={pitch}")
            map_x, map_y = self._generate_transform_map(
                equirect_img.shape, yaw, pitch, roll, h_fov, v_fov, 
                output_width, output_height
            )
            self.cache[cache_key] = (map_x, map_y)
        
        # Apply transformation using OpenCV remap
        perspective_img = cv2.remap(equirect_img, map_x, map_y, 
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        return perspective_img
    
    def _generate_transform_map(self, equirect_shape, yaw, pitch, roll, 
                               h_fov, v_fov, output_width, output_height):
        """
        Generate transformation mapping from perspective to equirectangular coordinates
        
        Returns:
            map_x, map_y: OpenCV remap coordinate arrays
        """
        
        equirect_height, equirect_width = equirect_shape[:2]
        
        # Convert angles to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        h_fov_rad = math.radians(h_fov)
        v_fov_rad = math.radians(v_fov)
        
        # Create output coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(output_width, dtype=np.float32),
            np.arange(output_height, dtype=np.float32)
        )
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x_coords - output_width / 2) / (output_width / 2)
        y_norm = (y_coords - output_height / 2) / (output_height / 2)
        
        # Calculate 3D ray directions for each pixel
        # Perspective projection
        focal_length_x = 1.0 / math.tan(h_fov_rad / 2)
        focal_length_y = 1.0 / math.tan(v_fov_rad / 2)
        
        # 3D coordinates on the image plane
        x_3d = x_norm / focal_length_x
        y_3d = y_norm / focal_length_y
        z_3d = np.ones_like(x_3d)
        
        # Normalize to unit vectors
        norm = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
        x_3d /= norm
        y_3d /= norm
        z_3d /= norm
        
        # Apply rotation transformations
        # Roll rotation (around Z-axis)
        if roll_rad != 0:
            cos_roll = math.cos(roll_rad)
            sin_roll = math.sin(roll_rad)
            x_rot = x_3d * cos_roll - y_3d * sin_roll
            y_rot = x_3d * sin_roll + y_3d * cos_roll
            x_3d, y_3d = x_rot, y_rot
        
        # Pitch rotation (around X-axis)
        if pitch_rad != 0:
            cos_pitch = math.cos(pitch_rad)
            sin_pitch = math.sin(pitch_rad)
            y_rot = y_3d * cos_pitch - z_3d * sin_pitch
            z_rot = y_3d * sin_pitch + z_3d * cos_pitch
            y_3d, z_3d = y_rot, z_rot
        
        # Yaw rotation (around Y-axis)
        if yaw_rad != 0:
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            x_rot = x_3d * cos_yaw + z_3d * sin_yaw
            z_rot = -x_3d * sin_yaw + z_3d * cos_yaw
            x_3d, z_3d = x_rot, z_rot
        
        # Convert 3D coordinates to spherical coordinates
        # Longitude (phi): -π to π
        phi = np.arctan2(x_3d, z_3d)
        
        # Latitude (theta): -π/2 to π/2
        theta = np.arcsin(np.clip(y_3d, -1.0, 1.0))
        
        # Convert spherical coordinates to equirectangular pixel coordinates
        # Longitude maps to x-coordinate
        map_x = (phi + math.pi) * equirect_width / (2 * math.pi)
        
        # Latitude maps to y-coordinate (corrected orientation)
        map_y = (math.pi/2 + theta) * equirect_height / math.pi
        
        # Ensure coordinates are within bounds
        map_x = np.clip(map_x, 0, equirect_width - 1)
        map_y = np.clip(map_y, 0, equirect_height - 1)
        
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    def clear_cache(self):
        """Clear the transformation map cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared transform cache ({cache_size} entries)")
    
    def get_cache_size(self):
        """Return number of cached transformation maps"""
        return len(self.cache)
    
    def get_camera_matrix(self, h_fov, v_fov, output_width, output_height):
        """
        Calculate camera intrinsic matrix for the given parameters
        
        Returns:
            3x3 camera matrix for COLMAP compatibility
        """
        focal_length_x = output_width / (2 * math.tan(math.radians(h_fov) / 2))
        focal_length_y = output_height / (2 * math.tan(math.radians(v_fov) / 2))
        
        cx = output_width / 2
        cy = output_height / 2
        
        camera_matrix = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ])
        
        return camera_matrix
    
    def get_rotation_matrix(self, yaw, pitch, roll):
        """
        Calculate rotation matrix from Euler angles
        
        Returns:
            3x3 rotation matrix
        """
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # Rotation matrices for each axis
        R_yaw = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])
        
        R_pitch = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])
        
        R_roll = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (order: yaw, pitch, roll)
        R = R_roll @ R_pitch @ R_yaw
        
        return R
