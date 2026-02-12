"""
Integration Test for 360FrameTools Pipeline
Tests the complete workflow: Extract → Split → Mask
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.extraction import FrameExtractor
from src.transforms import E2PTransform, E2CTransform
from src.masking import MultiCategoryMasker
from src.pipeline import MetadataHandler


class TestFullWorkflow:
    """Test complete pipeline workflow"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_equirect_image(self, temp_dir):
        """Create a sample equirectangular image for testing"""
        # Create a simple test image (equirectangular 2:1 ratio)
        img = np.zeros((1080, 2160, 3), dtype=np.uint8)
        
        # Add some colored regions
        img[:, :720] = [255, 0, 0]  # Blue
        img[:, 720:1440] = [0, 255, 0]  # Green
        img[:, 1440:] = [0, 0, 255]  # Red
        
        # Save
        image_path = temp_dir / "test_equirect.png"
        cv2.imwrite(str(image_path), img)
        
        return str(image_path)
    
    def test_e2p_transform(self, sample_equirect_image, temp_dir):
        """Test E2P transformation"""
        
        # Load image
        equirect_img = cv2.imread(sample_equirect_image)
        assert equirect_img is not None
        
        # Initialize transform
        e2p = E2PTransform()
        
        # Transform to perspective
        perspective_img = e2p.equirect_to_pinhole(
            equirect_img,
            yaw=0,
            pitch=0,
            roll=0,
            h_fov=90,
            output_width=1920,
            output_height=1080
        )
        
        assert perspective_img is not None
        assert perspective_img.shape == (1080, 1920, 3)
        
        # Save output
        output_path = temp_dir / "perspective_output.png"
        cv2.imwrite(str(output_path), perspective_img)
        
        assert output_path.exists()
        
        print(f"✓ E2P transform test passed")
    
    def test_e2c_transform(self, sample_equirect_image, temp_dir):
        """Test E2C cubemap transformation"""
        
        # Load image
        equirect_img = cv2.imread(sample_equirect_image)
        
        # Initialize transform
        e2c = E2CTransform()
        
        # Transform to cubemap
        cubemap_faces = e2c.equirect_to_cubemap(
            equirect_img,
            face_size=512,
            overlap_percent=10,
            mode='6-face'
        )
        
        assert len(cubemap_faces) == 6
        assert 'front' in cubemap_faces
        assert 'top' in cubemap_faces
        
        # Check face dimensions
        for face_name, face_img in cubemap_faces.items():
            assert face_img.shape[:2] == (512, 512)
        
        print(f"✓ E2C transform test passed")
    
    def test_metadata_handler(self, sample_equirect_image, temp_dir):
        """Test metadata handling"""
        
        handler = MetadataHandler()
        
        # Embed camera orientation
        success = handler.embed_camera_orientation(
            sample_equirect_image,
            yaw=45.0,
            pitch=30.0,
            roll=0.0,
            h_fov=90.0
        )
        
        assert success
        
        # Read back orientation
        orientation = handler.read_camera_orientation(sample_equirect_image)
        
        assert orientation is not None
        assert orientation['yaw'] == 45.0
        assert orientation['pitch'] == 30.0
        
        print(f"✓ Metadata handler test passed")
    
    def test_transform_cache(self, sample_equirect_image):
        """Test transformation caching"""
        
        equirect_img = cv2.imread(sample_equirect_image)
        e2p = E2PTransform()
        
        # First transform (should cache)
        img1 = e2p.equirect_to_pinhole(equirect_img, yaw=0, pitch=0, h_fov=90)
        cache_size_1 = e2p.get_cache_size()
        
        # Second transform with same params (should use cache)
        img2 = e2p.equirect_to_pinhole(equirect_img, yaw=0, pitch=0, h_fov=90)
        cache_size_2 = e2p.get_cache_size()
        
        # Cache should remain same size
        assert cache_size_1 == cache_size_2
        assert cache_size_1 > 0
        
        # Third transform with different params (should add to cache)
        img3 = e2p.equirect_to_pinhole(equirect_img, yaw=45, pitch=0, h_fov=90)
        cache_size_3 = e2p.get_cache_size()
        
        assert cache_size_3 > cache_size_2
        
        print(f"✓ Transform cache test passed (cache size: {cache_size_3})")
    
    def test_multiple_camera_generation(self, sample_equirect_image, temp_dir):
        """Test generating multiple camera views (simulates Stage 2)"""
        
        equirect_img = cv2.imread(sample_equirect_image)
        e2p = E2PTransform()
        
        # Generate 8 camera views
        camera_count = 8
        output_dir = temp_dir / "cameras"
        output_dir.mkdir()
        
        for i in range(camera_count):
            yaw = (360 / camera_count) * i
            
            perspective_img = e2p.equirect_to_pinhole(
                equirect_img,
                yaw=yaw,
                pitch=0,
                roll=0,
                h_fov=110,
                output_width=1920,
                output_height=1080
            )
            
            output_path = output_dir / f"camera_{i:02d}.png"
            cv2.imwrite(str(output_path), perspective_img)
        
        # Verify all cameras generated
        camera_files = list(output_dir.glob("camera_*.png"))
        assert len(camera_files) == camera_count
        
        print(f"✓ Multiple camera generation test passed ({camera_count} cameras)")


def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("360FrameTools - Integration Tests")
    print("="*60 + "\n")
    
    # Run pytest
    result = pytest.main([__file__, '-v', '--tb=short'])
    
    return result


if __name__ == '__main__':
    sys.exit(run_tests())
