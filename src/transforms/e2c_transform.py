"""
Equirectangular to Cube Map (E2C) Transformation Engine
Converts 360-degree equirectangular images to cube map faces with adjustable overlap.

OPTIMIZATION NOTES:
- OpenCV cv2.remap() is ESSENTIAL for cubemap generation
- Used for fast perspective projection of each cube face
- No lightweight alternative exists with comparable performance

Supports TWO distinct modes:
1. **6-Face Cubemap** (for VR/rendering):
   - Standard cubemap: Front, Back, Left, Right, Top, Bottom (90° FOV each)
   - Layouts: Cross Horizontal (4:3), Cross Vertical (3:4), Strip (6:1), Separate files
   - Use case: Unity, Unreal, WebGL, environment maps

2. **8-Tile Grid** (for photogrammetry):
   - 8 perspective views arranged in 4×2 grid
   - Configurable FOV (90-110°) and overlap (0-50%)
   - Use case: COLMAP, Meshroom, Gaussian Splatting, NeRF

Based on GUIDE-Cubemap-8Tiles.md

Ported from 360toFrame for 360FrameTools unified application.
"""

import numpy as np
import cv2  # REQUIRED: cv2.remap() for cubemap face generation
import math
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CubemapLayout(Enum):
    """Cubemap output layout formats (for 6-face cubemaps)"""
    CROSS_HORIZONTAL = "cross_horizontal"  # 4:3 aspect (Unity/Unreal standard)
    CROSS_VERTICAL = "cross_vertical"      # 3:4 aspect
    STRIP_HORIZONTAL = "strip_horizontal"  # 6:1 aspect (compact storage)
    SEPARATE = "separate"                  # 6 individual face images


class E2CTransform:
    """Equirectangular to Cube Map transformation class"""
    
    def __init__(self):
        self.cache = {}  # Cache for transformation maps
        
    def equirect_to_cubemap(self, equirect_img, face_size=1024, overlap_percent=10, mode='6-face'):
        """
        Convert equirectangular image to cube map faces with overlap
        
        Args:
            equirect_img: Input equirectangular image (numpy array)
            face_size: Size of each cube face (square)
            overlap_percent: Overlap percentage (0-50) for seamless stitching
            mode: '6-face' for standard cubemap, '8-tile' for 8-tile variant
            
        Returns:
            Dictionary with cube faces (6 or 8 depending on mode)
        """
        
        if mode == '8-tile':
            return self._generate_8tile_cubemap(equirect_img, face_size, overlap_percent)
        else:
            return self._generate_6face_cubemap(equirect_img, face_size, overlap_percent)
    
    def _generate_6face_cubemap(self, equirect_img, face_size, overlap_percent):
        """Generate standard 6-face cubemap"""
        
        # Define cube face parameters (yaw, pitch)
        faces = {
            'front': (0, 0),
            'right': (90, 0),
            'back': (180, 0),
            'left': (270, 0),
            'top': (0, 90),
            'bottom': (0, -90)
        }
        
        # Calculate effective FOV based on overlap
        # With overlap, we need to capture more than 90 degrees per face
        overlap_factor = 1 + (overlap_percent / 100.0)
        fov = 90 * overlap_factor
        
        cube_faces = {}
        
        for face_name, (yaw, pitch) in faces.items():
            # Generate cube face using perspective projection
            face_img = self._generate_cube_face(
                equirect_img, 
                yaw=yaw, 
                pitch=pitch, 
                fov=fov,
                face_size=face_size
            )
            cube_faces[face_name] = face_img
            logger.debug(f"Generated {face_name} face ({face_size}x{face_size}, FOV={fov:.1f}°)")
        
        return cube_faces
    
    def _generate_8tile_cubemap(self, equirect_img, face_size, overlap_percent):
        """
        Generate 8-tile cubemap variant
        
        8-tile layout includes:
        - 4 horizontal faces (front, right, back, left)
        - 4 corner/edge tiles for better coverage
        
        Returns:
            Dictionary with 8 tiles
        """
        
        overlap_factor = 1 + (overlap_percent / 100.0)
        fov = 90 * overlap_factor
        
        # Standard 4 horizontal faces
        tiles = {
            'front': (0, 0),
            'right': (90, 0),
            'back': (180, 0),
            'left': (270, 0),
        }
        
        # Add 4 additional corner tiles (diagonal views)
        tiles.update({
            'front_right': (45, 0),
            'back_right': (135, 0),
            'back_left': (-135, 0),
            'front_left': (-45, 0)
        })
        
        cube_faces = {}
        
        for tile_name, (yaw, pitch) in tiles.items():
            face_img = self._generate_cube_face(
                equirect_img,
                yaw=yaw,
                pitch=pitch,
                fov=fov,
                face_size=face_size
            )
            cube_faces[tile_name] = face_img
            logger.debug(f"Generated {tile_name} tile ({face_size}x{face_size})")
        
        return cube_faces
    
    def _generate_cube_face(self, equirect_img, yaw=0, pitch=0, fov=90, face_size=1024):
        """
        Generate a single cube face from equirectangular image
        
        Args:
            equirect_img: Input equirectangular image
            yaw: Horizontal rotation in degrees
            pitch: Vertical tilt in degrees
            fov: Field of view in degrees
            face_size: Output face size
            
        Returns:
            Cube face image (numpy array)
        """
        
        # Create output image
        face_img = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        
        # Get equirectangular dimensions
        eq_height, eq_width = equirect_img.shape[:2]
        
        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        fov_rad = np.radians(fov)
        
        # Half FOV for calculations
        half_fov = fov_rad / 2
        
        # Create coordinate grids for the face
        x = np.linspace(-1, 1, face_size)
        y = np.linspace(-1, 1, face_size)
        xx, yy = np.meshgrid(x, y)
        
        # Convert face coordinates to 3D sphere coordinates
        # This creates a perspective projection on the cube face
        z = np.ones_like(xx)
        
        # Scale by FOV
        xx_scaled = xx * np.tan(half_fov)
        yy_scaled = yy * np.tan(half_fov)
        
        # Create 3D vectors for each pixel
        vectors = np.stack([xx_scaled, yy_scaled, z], axis=-1)
        
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors = vectors / (norms + 1e-8)
        
        # Apply rotation (yaw and pitch)
        # Rotation matrix for yaw
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        # Rotation matrix for pitch
        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        
        # Apply rotations to vectors
        x_rot = vectors[..., 0] * cos_yaw - vectors[..., 2] * sin_yaw
        z_rot = vectors[..., 0] * sin_yaw + vectors[..., 2] * cos_yaw
        
        y_rot = vectors[..., 1] * cos_pitch - z_rot * sin_pitch
        z_final = vectors[..., 1] * sin_pitch + z_rot * cos_pitch
        
        # Convert 3D coordinates to spherical coordinates
        # Longitude (u) and latitude (v)
        longitude = np.arctan2(x_rot, z_final)
        latitude = np.arcsin(np.clip(y_rot, -1, 1))
        
        # Convert to equirectangular image coordinates
        u = (longitude + np.pi) / (2 * np.pi) * (eq_width - 1)
        # Fixed: latitude should be inverted for correct vertical mapping
        v = (latitude + np.pi / 2) / np.pi * (eq_height - 1)
        
        # Ensure coordinates are within bounds
        u = np.clip(u, 0, eq_width - 1).astype(np.float32)
        v = np.clip(v, 0, eq_height - 1).astype(np.float32)
        
        # Use OpenCV remap for faster interpolation
        face_img = cv2.remap(equirect_img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        return face_img
    
    def get_cube_face_names(self, mode='6-face'):
        """
        Get list of cube face names
        
        Args:
            mode: '6-face' or '8-tile'
            
        Returns:
            List of face/tile names
        """
        if mode == '8-tile':
            return ['front', 'right', 'back', 'left', 
                   'front_right', 'back_right', 'back_left', 'front_left']
        else:
            return ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    def create_cubemap_cross_horizontal(self, faces: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Arrange 6 faces into horizontal cross layout (4:3 aspect).
        
        Layout:
                [Top]
        [Left] [Front] [Right] [Back]
               [Bottom]
        
        Args:
            faces: Dict with keys 'front', 'back', 'left', 'right', 'top', 'bottom'
        
        Returns:
            Combined cubemap image (3h × 4w)
        """
        face_size = faces['front'].shape[0]
        cross = np.zeros((face_size * 3, face_size * 4, 3), dtype=np.uint8)
        
        # Row 1: Top (center column 2)
        cross[0:face_size, face_size:face_size*2] = faces['top']
        
        # Row 2: Left, Front, Right, Back (all 4 columns)
        cross[face_size:face_size*2, 0:face_size] = faces['left']
        cross[face_size:face_size*2, face_size:face_size*2] = faces['front']
        cross[face_size:face_size*2, face_size*2:face_size*3] = faces['right']
        cross[face_size:face_size*2, face_size*3:face_size*4] = faces['back']
        
        # Row 3: Bottom (center column 2)
        cross[face_size*2:face_size*3, face_size:face_size*2] = faces['bottom']
        
        return cross
    
    def create_cubemap_cross_vertical(self, faces: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Arrange 6 faces into vertical cross layout (3:4 aspect).
        
        Layout:
               [Top]
              [Front]
        [Left][Back][Right]
              [Bottom]
        
        Args:
            faces: Dict with keys 'front', 'back', 'left', 'right', 'top', 'bottom'
        
        Returns:
            Combined cubemap image (4h × 3w)
        """
        face_size = faces['front'].shape[0]
        cross = np.zeros((face_size * 4, face_size * 3, 3), dtype=np.uint8)
        
        # Row 1: Top (center)
        cross[0:face_size, face_size:face_size*2] = faces['top']
        
        # Row 2: Front (center)
        cross[face_size:face_size*2, face_size:face_size*2] = faces['front']
        
        # Row 3: Left, Back, Right
        cross[face_size*2:face_size*3, 0:face_size] = faces['left']
        cross[face_size*2:face_size*3, face_size:face_size*2] = faces['back']
        cross[face_size*2:face_size*3, face_size*2:face_size*3] = faces['right']
        
        # Row 4: Bottom (center)
        cross[face_size*3:face_size*4, face_size:face_size*2] = faces['bottom']
        
        return cross
    
    def create_cubemap_strip(self, faces: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Arrange 6 faces into horizontal strip (6:1 aspect).
        
        Layout: [Right][Left][Top][Bottom][Front][Back]
        
        Args:
            faces: Dict with keys 'front', 'back', 'left', 'right', 'top', 'bottom'
        
        Returns:
            Combined cubemap image (h × 6w)
        """
        face_size = faces['front'].shape[0]
        strip = np.zeros((face_size, face_size * 6, 3), dtype=np.uint8)
        
        # Standard strip order (Unity/WebGL convention)
        strip[:, 0:face_size] = faces['right']
        strip[:, face_size:face_size*2] = faces['left']
        strip[:, face_size*2:face_size*3] = faces['top']
        strip[:, face_size*3:face_size*4] = faces['bottom']
        strip[:, face_size*4:face_size*5] = faces['front']
        strip[:, face_size*5:face_size*6] = faces['back']
        
        return strip
    
    def clear_cache(self):
        """Clear the transformation map cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared cubemap cache ({cache_size} entries)")
