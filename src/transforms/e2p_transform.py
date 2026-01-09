"""
Equirectangular to Pinhole (E2P) Transformation Engine
Converts 360-degree equirectangular images to perspective pinhole camera views.
Based on spherical coordinate mapping and camera projection mathematics.

Ported from 360toFrame for 360FrameTools unified application.
"""

import numpy as np
import cv2
import math
import logging

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class TorchE2PTransform:
    """
    GPU-accelerated Equirectangular to Pinhole transformation using PyTorch.
    """
    def __init__(self, device=None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TorchE2PTransform")
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache = {}
        logger.info(f"Initialized TorchE2PTransform on {self.device}")

    def equirect_to_pinhole(self, equirect_tensor, yaw=0, pitch=0, roll=0, 
                           h_fov=90, v_fov=None, output_width=1920, output_height=1080):
        """
        Convert equirectangular image to pinhole perspective view using GPU.
        
        Args:
            equirect_tensor: Input image tensor (1, 3, H, W) or (3, H, W), normalized 0-1 or 0-255
            yaw, pitch, roll: Camera orientation in degrees
            h_fov, v_fov: Field of view
            output_width, output_height: Output dimensions
            
        Returns:
            Perspective view tensor (1, 3, H_out, W_out)
        """
        if v_fov is None:
            v_fov = h_fov * output_height / output_width

        # Ensure input is (N, C, H, W)
        if equirect_tensor.dim() == 3:
            equirect_tensor = equirect_tensor.unsqueeze(0)
            
        # Create cache key
        cache_key = (yaw, pitch, roll, h_fov, v_fov, output_width, output_height)
        
        if cache_key in self.cache:
            grid = self.cache[cache_key]
        else:
            grid = self._generate_grid(yaw, pitch, roll, h_fov, v_fov, output_width, output_height)
            self.cache[cache_key] = grid
            
        # Apply grid sample
        # grid is (N, H_out, W_out, 2)
        # input is (N, C, H_in, W_in)
        return F.grid_sample(equirect_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def batch_equirect_to_pinhole(self, equirect_batch, yaw=0, pitch=0, roll=0, 
                                   h_fov=90, v_fov=None, output_width=1920, output_height=1080):
        """
        Convert multiple equirectangular images to pinhole views (batch processing).
        Processes all frames simultaneously on GPU for massive speedup.
        
        Args:
            equirect_batch: Batch of images (N, 3, H, W) tensor, normalized 0-1
            yaw, pitch, roll: Camera orientation in degrees (same for all frames)
            h_fov, v_fov: Field of view
            output_width, output_height: Output dimensions
            
        Returns:
            Batch of perspective views (N, 3, H_out, W_out)
        """
        if v_fov is None:
            v_fov = h_fov * output_height / output_width

        # Ensure input is (N, C, H, W)
        if equirect_batch.dim() == 3:
            equirect_batch = equirect_batch.unsqueeze(0)
            
        batch_size = equirect_batch.shape[0]
        
        # Create cache key (camera params only)
        cache_key = (yaw, pitch, roll, h_fov, v_fov, output_width, output_height)
        
        if cache_key in self.cache:
            grid_single = self.cache[cache_key]  # (1, H, W, 2)
        else:
            grid_single = self._generate_grid(yaw, pitch, roll, h_fov, v_fov, output_width, output_height)
            self.cache[cache_key] = grid_single
            
        # Expand grid for batch: (1, H, W, 2) -> (N, H, W, 2)
        grid_batch = grid_single.expand(batch_size, -1, -1, -1)
        
        # Batch transform (all frames processed in parallel on GPU)
        return F.grid_sample(equirect_batch, grid_batch, mode='bilinear', padding_mode='border', align_corners=True)

    def get_optimal_batch_size(self, input_height, input_width, output_height, output_width):
        """
        Calculate optimal batch size based on available VRAM.
        
        Args:
            input_height, input_width: Input equirectangular dimensions
            output_height, output_width: Output perspective dimensions
            
        Returns:
            Optimal batch size (int)
        """
        if self.device == 'cpu':
            return 1  # No batching on CPU
            
        try:
            # Get GPU memory info
            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            free_vram = total_vram - reserved
            
            # Conservative estimate: use 50% of free VRAM
            available = free_vram * 0.5
            
            # Memory per frame (in bytes):
            # Input: H × W × 3 channels × 4 bytes (float32)
            input_size = input_height * input_width * 3 * 4
            # Output: H_out × W_out × 3 × 4
            output_size = output_height * output_width * 3 * 4
            # Grid: H_out × W_out × 2 × 4 (shared across batch, but count once)
            grid_size = output_height * output_width * 2 * 4
            # Overhead for intermediate calculations (~30% of input+output)
            overhead = (input_size + output_size) * 0.3
            
            per_frame_memory = input_size + output_size + overhead
            
            # Calculate batch size
            batch_size = int(available // per_frame_memory)
            
            # Clamp between 1 and 32 (practical limits)
            batch_size = max(1, min(batch_size, 32))
            
            logger.info(f"Auto-detected optimal batch size: {batch_size} (Free VRAM: {free_vram / 1024**3:.2f} GB)")
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to detect optimal batch size: {e}. Using batch_size=4")
            return 4

    def _generate_grid(self, yaw, pitch, roll, h_fov, v_fov, width, height):
        """Generate sampling grid for grid_sample"""
        # Create meshgrid
        y_coords = torch.linspace(-1, 1, height, device=self.device)
        x_coords = torch.linspace(-1, 1, width, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Perspective projection params
        h_fov_rad = math.radians(h_fov)
        v_fov_rad = math.radians(v_fov)
        f_x = 1.0 / math.tan(h_fov_rad / 2)
        f_y = 1.0 / math.tan(v_fov_rad / 2)
        
        # 3D coordinates on image plane
        x_3d = grid_x / f_x
        y_3d = grid_y / f_y
        z_3d = torch.ones_like(x_3d)
        
        # Normalize
        norm = torch.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
        x_3d /= norm
        y_3d /= norm
        z_3d /= norm
        
        # Apply rotations (inverse of camera rotation)
        # Note: The math in original E2PTransform applies rotation to the ray vectors.
        # We need to replicate that exact logic.
        
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # Roll
        if roll_rad != 0:
            cos_roll = math.cos(roll_rad)
            sin_roll = math.sin(roll_rad)
            x_rot = x_3d * cos_roll - y_3d * sin_roll
            y_rot = x_3d * sin_roll + y_3d * cos_roll
            x_3d, y_3d = x_rot, y_rot
            
        # Pitch
        if pitch_rad != 0:
            cos_pitch = math.cos(pitch_rad)
            sin_pitch = math.sin(pitch_rad)
            y_rot = y_3d * cos_pitch - z_3d * sin_pitch
            z_rot = y_3d * sin_pitch + z_3d * cos_pitch
            y_3d, z_3d = y_rot, z_rot
            
        # Yaw
        if yaw_rad != 0:
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            x_rot = x_3d * cos_yaw + z_3d * sin_yaw
            z_rot = -x_3d * sin_yaw + z_3d * cos_yaw
            x_3d, z_3d = x_rot, z_rot
            
        # Convert to spherical (phi, theta)
        # phi = atan2(x, z) -> longitude
        # theta = asin(y) -> latitude
        phi = torch.atan2(x_3d, z_3d)
        theta = torch.asin(torch.clamp(y_3d, -1.0, 1.0))
        
        # Map to normalized equirectangular coordinates [-1, 1]
        # Longitude: -pi to pi -> -1 to 1
        # Latitude: -pi/2 to pi/2 -> -1 to 1
        # Note: grid_sample expects (x, y) where x is width (long), y is height (lat)
        
        # Original code: map_x = (phi + pi) * W / (2pi)
        # Normalized: (phi + pi) / (2pi) * 2 - 1 = phi / pi
        grid_u = phi / math.pi
        
        # Original code: map_y = (pi/2 + theta) * H / pi
        # Normalized: (pi/2 + theta) / pi * 2 - 1 = (0.5 + theta/pi) * 2 - 1 = 1 + 2theta/pi - 1 = 2theta/pi
        # Wait, let's check coordinate system of grid_sample.
        # (-1, -1) is top-left. (1, 1) is bottom-right.
        # Image coordinates: y increases downwards.
        # Spherical: theta increases upwards (usually).
        # Original code: map_y = (pi/2 + theta) ...
        # If theta is -pi/2 (bottom), map_y = 0 (top). So original code flips Y.
        # Let's stick to the original mapping logic.
        
        # Normalized U (Longitude): -1 (left) to 1 (right)
        # phi ranges -pi to pi.
        # grid_u = phi / pi.
        
        # Normalized V (Latitude): -1 (top) to 1 (bottom)
        # theta ranges -pi/2 to pi/2.
        # Original map_y goes 0 to H.
        # map_y = (pi/2 + theta) * H / pi
        # Normalized V = map_y / H * 2 - 1
        # = (pi/2 + theta) / pi * 2 - 1
        # = (0.5 + theta/pi) * 2 - 1
        # = 1 + 2*theta/pi - 1
        # = 2 * theta / pi
        
        grid_v = 2 * theta / math.pi
        
        # Stack to (H, W, 2)
        grid = torch.stack((grid_u, grid_v), dim=-1)
        
        # Add batch dimension (1, H, W, 2)
        return grid.unsqueeze(0)


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
