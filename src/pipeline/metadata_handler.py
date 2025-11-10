"""
Metadata Handler for 360FrameTools
Handles camera metadata preservation and EXIF embedding.

IMPORTANT: Does NOT extract/preserve GPS or GYRO data (not needed for photogrammetry).
Only preserves camera info (make, model, lens, focal length, etc.) and embeds
camera orientation (yaw, pitch, roll) for Stage 2 outputs.
"""

import piexif
from PIL import Image
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataHandler:
    """
    Handle camera metadata preservation and orientation embedding.
    NO GPS/GYRO metadata (excluded by design).
    """
    
    def __init__(self):
        """Initialize MetadataHandler"""
        pass
    
    def extract_camera_metadata(self, image_path: str) -> Dict:
        """
        Extract camera metadata from image (NO GPS/GYRO).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with camera metadata
        """
        try:
            img = Image.open(image_path)
            
            # Try multiple methods to get EXIF data (handle empty bytes)
            exif_dict = {}
            try:
                if 'exif' in img.info and img.info['exif']:
                    exif_dict = piexif.load(img.info['exif'])
                elif hasattr(img, '_getexif') and img._getexif() is not None:
                    exif_bytes = img.info.get('exif', b'')
                    if exif_bytes:
                        exif_dict = piexif.load(exif_bytes)
                else:
                    # Try to load directly from file
                    exif_dict = piexif.load(str(image_path))
            except Exception as e:
                # No EXIF data available or invalid format
                logger.debug(f"No EXIF data in {Path(image_path).name}: {e}")
                exif_dict = {}
            
            metadata = {}
            
            # Extract camera info from EXIF
            if '0th' in exif_dict:
                ifd = exif_dict['0th']
                
                # Camera make and model
                if piexif.ImageIFD.Make in ifd:
                    metadata['Make'] = ifd[piexif.ImageIFD.Make].decode('utf-8', errors='ignore')
                if piexif.ImageIFD.Model in ifd:
                    metadata['Model'] = ifd[piexif.ImageIFD.Model].decode('utf-8', errors='ignore')
                if piexif.ImageIFD.Software in ifd:
                    metadata['Software'] = ifd[piexif.ImageIFD.Software].decode('utf-8', errors='ignore')
            
            # Extract EXIF data
            if 'Exif' in exif_dict:
                exif_ifd = exif_dict['Exif']
                
                # Focal length
                if piexif.ExifIFD.FocalLength in exif_ifd:
                    focal = exif_ifd[piexif.ExifIFD.FocalLength]
                    if isinstance(focal, tuple):
                        metadata['FocalLength'] = focal[0] / focal[1] if focal[1] != 0 else 0
                    else:
                        metadata['FocalLength'] = focal
                
                # Focal length in 35mm
                if piexif.ExifIFD.FocalLengthIn35mmFilm in exif_ifd:
                    metadata['FocalLengthIn35mmFilm'] = exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm]
                
                # Aperture
                if piexif.ExifIFD.FNumber in exif_ifd:
                    fnum = exif_ifd[piexif.ExifIFD.FNumber]
                    if isinstance(fnum, tuple):
                        metadata['FNumber'] = fnum[0] / fnum[1] if fnum[1] != 0 else 0
                    else:
                        metadata['FNumber'] = fnum
                
                # ISO
                if piexif.ExifIFD.ISOSpeedRatings in exif_ifd:
                    metadata['ISO'] = exif_ifd[piexif.ExifIFD.ISOSpeedRatings]
                
                # Exposure time
                if piexif.ExifIFD.ExposureTime in exif_ifd:
                    exp = exif_ifd[piexif.ExifIFD.ExposureTime]
                    if isinstance(exp, tuple):
                        metadata['ExposureTime'] = f"{exp[0]}/{exp[1]}"
                    else:
                        metadata['ExposureTime'] = exp
                
                # Date/time
                if piexif.ExifIFD.DateTimeOriginal in exif_ifd:
                    metadata['DateTimeOriginal'] = exif_ifd[piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')
            
            logger.debug(f"Extracted camera metadata: {len(metadata)} fields")
            return metadata
        
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def embed_camera_orientation(self, image_path: str, yaw: float, pitch: float, 
                                 roll: float, h_fov: float, output_path: Optional[str] = None) -> bool:
        """
        Embed camera orientation and FOV into image EXIF.
        
        Args:
            image_path: Path to input image
            yaw: Camera yaw in degrees
            pitch: Camera pitch in degrees
            roll: Camera roll in degrees
            h_fov: Horizontal field of view in degrees
            output_path: Output path (overwrites input if None)
            
        Returns:
            True if successful
        """
        try:
            img = Image.open(image_path)
            
            # Load existing EXIF or create new
            try:
                exif_dict = piexif.load(img.info.get('exif', b''))
            except:
                exif_dict = {'0th': {}, 'Exif': {}, '1st': {}, 'GPS': {}}
            
            # Embed orientation in UserComment (JSON format)
            orientation_data = {
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'h_fov': h_fov,
                'generated_by': '360FrameTools',
                'timestamp': datetime.now().isoformat()
            }
            
            orientation_json = json.dumps(orientation_data)
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = orientation_json.encode('utf-8')
            
            # Also embed in ImageDescription for compatibility
            exif_dict['0th'][piexif.ImageIFD.ImageDescription] = f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f} FOV:{h_fov:.1f}".encode('utf-8')
            
            # Update software tag
            exif_dict['0th'][piexif.ImageIFD.Software] = b'360FrameTools v1.0'
            
            # Convert to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Save image with new EXIF
            output = output_path or image_path
            img.save(output, exif=exif_bytes, quality=95)
            
            logger.debug(f"Embedded camera orientation: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
            return True
        
        except Exception as e:
            logger.error(f"Error embedding orientation: {e}")
            return False
    
    def read_camera_orientation(self, image_path: str) -> Optional[Dict]:
        """
        Read camera orientation from image EXIF.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with orientation data or None
        """
        try:
            img = Image.open(image_path)
            exif_dict = piexif.load(img.info.get('exif', b''))
            
            if 'Exif' in exif_dict and piexif.ExifIFD.UserComment in exif_dict['Exif']:
                user_comment = exif_dict['Exif'][piexif.ExifIFD.UserComment].decode('utf-8', errors='ignore')
                
                # Try to parse JSON
                try:
                    orientation_data = json.loads(user_comment)
                    return orientation_data
                except:
                    pass
            
            return None
        
        except Exception as e:
            logger.error(f"Error reading orientation: {e}")
            return None
    
    def copy_metadata(self, source_path: str, dest_path: str, 
                     preserve_orientation: bool = True) -> bool:
        """
        Copy metadata from source to destination image.
        
        Args:
            source_path: Source image path
            dest_path: Destination image path
            preserve_orientation: Keep orientation data if exists
            
        Returns:
            True if successful
        """
        try:
            source_img = Image.open(source_path)
            dest_img = Image.open(dest_path)
            
            # Load source EXIF
            exif_dict = piexif.load(source_img.info.get('exif', b''))
            
            # Remove GPS/GYRO data if present (should not be there, but ensure)
            if 'GPS' in exif_dict:
                exif_dict['GPS'] = {}
            
            # Convert to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Save with copied EXIF
            dest_img.save(dest_path, exif=exif_bytes, quality=95)
            
            logger.debug(f"Copied metadata from {Path(source_path).name} to {Path(dest_path).name}")
            return True
        
        except Exception as e:
            logger.error(f"Error copying metadata: {e}")
            return False
    
    def save_metadata_json(self, metadata: Dict, output_path: str) -> bool:
        """
        Save metadata as JSON sidecar file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save JSON file
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved metadata JSON: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving metadata JSON: {e}")
            return False
    
    def load_metadata_json(self, json_path: str) -> Optional[Dict]:
        """
        Load metadata from JSON sidecar file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Metadata dictionary or None
        """
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            logger.debug(f"Loaded metadata JSON: {json_path}")
            return metadata
        
        except Exception as e:
            logger.error(f"Error loading metadata JSON: {e}")
            return None
