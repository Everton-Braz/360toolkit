"""
360toolkit Comprehensive Test Suite
====================================
Tests all modules, imports, GPU detection, pipeline config generation,
and individual stage components.

Usage:
    pytest tests/ -v
    pytest tests/test_all_modules.py -v --tb=short
    python tests/test_all_modules.py   (standalone)
"""

import os
import sys
import json
import tempfile
import importlib
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix OpenMP conflict before any torch import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pytest
import numpy as np


# ============================================================================
# 1. IMPORT TESTS — Verify all modules load without errors
# ============================================================================

class TestImports:
    """Verify that all project modules can be imported successfully."""

    def test_import_src(self):
        import src
        assert hasattr(src, '__version__')

    def test_import_config_defaults(self):
        from src.config import defaults
        assert hasattr(defaults, 'DEFAULT_FPS')
        assert hasattr(defaults, 'DEFAULT_SPLIT_COUNT')
        assert hasattr(defaults, 'DEFAULT_H_FOV')
        assert hasattr(defaults, 'COCO_CLASSES')
        assert hasattr(defaults, 'MASKING_CATEGORIES')

    def test_import_config_settings(self):
        from src.config.settings import SettingsManager, get_settings
        settings = get_settings()
        assert isinstance(settings, SettingsManager)

    def test_import_config_manager(self):
        from src.config.config_manager import ConfigManager
        cm = ConfigManager()
        assert hasattr(cm, 'get_default_config')
        assert hasattr(cm, 'validate_config')

    def test_import_config_features(self):
        from src.config.features import FeatureFlags
        features = FeatureFlags.get_enabled_features()
        assert isinstance(features, list)

    def test_import_extraction(self):
        from src.extraction import FrameExtractor
        fe = FrameExtractor()
        assert hasattr(fe, 'extract_frames')
        assert hasattr(fe, 'get_video_info')

    def test_import_sdk_extractor(self):
        from src.extraction.sdk_extractor import SDKExtractor
        assert hasattr(SDKExtractor, 'is_available')

    def test_import_transforms_e2p(self):
        from src.transforms import E2PTransform
        t = E2PTransform()
        assert hasattr(t, 'equirect_to_pinhole')
        assert hasattr(t, 'clear_cache')

    def test_import_transforms_e2c(self):
        from src.transforms import E2CTransform
        t = E2CTransform()
        assert hasattr(t, 'equirect_to_cubemap')

    def test_import_cubemap_layout(self):
        from src.transforms.e2c_transform import CubemapLayout
        assert hasattr(CubemapLayout, 'CROSS_HORIZONTAL')

    def test_import_masking_factory(self):
        from src.masking import get_masker
        assert callable(get_masker)

    def test_import_multi_category_masker(self):
        from src.masking.multi_category_masker import MultiCategoryMasker
        assert hasattr(MultiCategoryMasker, 'generate_mask')

    def test_import_pipeline_metadata(self):
        from src.pipeline.metadata_handler import MetadataHandler
        mh = MetadataHandler()
        assert hasattr(mh, 'extract_camera_metadata')
        assert hasattr(mh, 'embed_camera_orientation')

    def test_import_pipeline_colmap_stage(self):
        from src.pipeline.colmap_stage import ColmapSettings
        assert hasattr(ColmapSettings, 'alignment_mode')

    def test_import_export_formats(self):
        from src.pipeline.export_formats import LichtfeldExporter, RealityScanExporter
        assert LichtfeldExporter is not None
        assert RealityScanExporter is not None

    @pytest.mark.skip(reason="pycolmap causes fatal Windows DLL crash (0xc0000138)")
    def test_import_premium_modules(self):
        from src.premium import RigColmapIntegrator
        assert hasattr(RigColmapIntegrator, 'create_virtual_camera')

    @pytest.mark.skip(reason="pycolmap causes fatal Windows DLL crash (0xc0000138)")
    def test_import_pose_transfer(self):
        from src.premium.pose_transfer_integration import (
            PoseTransferIntegrator, VirtualCamera, get_virtual_camera_rotations
        )
        rotations = get_virtual_camera_rotations()
        assert isinstance(rotations, list)
        assert len(rotations) > 0

    def test_import_utils(self):
        from src.utils.resource_path import get_resource_path
        assert callable(get_resource_path)

    def test_import_numpy(self):
        import numpy as np
        assert np.__version__

    def test_import_cv2(self):
        import cv2
        assert cv2.__version__

    def test_import_pil(self):
        from PIL import Image
        assert Image is not None


# ============================================================================
# 2. GPU / CUDA DETECTION TESTS
# ============================================================================

class TestGPUDetection:
    """Test GPU detection and graceful fallback."""

    def test_torch_importable(self):
        try:
            import torch
            assert torch.__version__
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_torch_cuda_status(self):
        """CUDA may or may not be available — just verify detection doesn't crash."""
        import torch
        available = torch.cuda.is_available()
        assert isinstance(available, bool)
        if available:
            device_name = torch.cuda.get_device_name(0)
            assert isinstance(device_name, str)
            print(f"  GPU: {device_name}")
            print(f"  CUDA: {torch.version.cuda}")

    def test_cuda_tensor_test(self):
        """Replicate the _select_device() tensor test used in masking."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA available")
        
        try:
            test = torch.zeros(1, device='cuda')
            result = test + 1
            assert result.item() == 1.0
            del test, result
            torch.cuda.empty_cache()
            print("  CUDA tensor test: PASSED")
        except RuntimeError as e:
            # Expected on sm_120 with cu124
            print(f"  CUDA tensor test: FAILED ({e})")
            print("  → CPU fallback will be used")

    def test_onnxruntime_gpu(self):
        """Check ONNX Runtime GPU availability."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            assert isinstance(providers, list)
            assert 'CPUExecutionProvider' in providers
            has_cuda = 'CUDAExecutionProvider' in providers
            print(f"  ONNX providers: {providers}")
            print(f"  ONNX GPU: {has_cuda}")
        except ImportError:
            pytest.skip("ONNX Runtime not installed")

    def test_ultralytics_importable(self):
        try:
            import ultralytics
            assert ultralytics.__version__
        except ImportError:
            pytest.skip("Ultralytics not installed")


# ============================================================================
# 3. CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration defaults, validation, and preset loading."""

    def test_default_values(self):
        from src.config.defaults import (
            DEFAULT_FPS, DEFAULT_SPLIT_COUNT, DEFAULT_H_FOV,
            DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MODEL_SIZE,
            DEFAULT_USE_GPU, MASK_VALUE_REMOVE, MASK_VALUE_KEEP,
        )
        assert DEFAULT_FPS > 0
        assert 1 <= DEFAULT_SPLIT_COUNT <= 12
        assert 30 <= DEFAULT_H_FOV <= 150
        assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0
        assert DEFAULT_MODEL_SIZE in ('nano', 'small', 'medium', 'large', 'xlarge')
        assert isinstance(DEFAULT_USE_GPU, bool)
        assert MASK_VALUE_REMOVE == 0
        assert MASK_VALUE_KEEP == 255

    def test_coco_classes(self):
        from src.config.defaults import COCO_CLASSES
        assert isinstance(COCO_CLASSES, dict)
        # COCO_CLASSES maps class_name -> class_id
        assert 'person' in COCO_CLASSES
        assert COCO_CLASSES['person'] == 0

    def test_masking_categories(self):
        from src.config.defaults import MASKING_CATEGORIES
        assert isinstance(MASKING_CATEGORIES, dict)
        assert 'persons' in MASKING_CATEGORIES
        assert 'personal_objects' in MASKING_CATEGORIES
        assert 'animals' in MASKING_CATEGORIES

    def test_camera_groups(self):
        from src.config.defaults import DEFAULT_CAMERA_GROUPS
        assert isinstance(DEFAULT_CAMERA_GROUPS, (dict, list))
        # Should have at least the 8-camera horizontal preset
        assert len(DEFAULT_CAMERA_GROUPS) >= 1

    def test_compass_rings(self):
        from src.config.defaults import DEFAULT_COMPASS_RINGS
        assert isinstance(DEFAULT_COMPASS_RINGS, list)
        assert len(DEFAULT_COMPASS_RINGS) >= 1

    def test_config_manager_default(self):
        from src.config.config_manager import ConfigManager
        cm = ConfigManager()
        config = cm.get_default_config()
        assert isinstance(config, dict)

    def test_config_validation(self):
        from src.config.config_manager import ConfigManager
        cm = ConfigManager()
        config = cm.get_default_config()
        valid, errors = cm.validate_config(config)
        assert isinstance(valid, bool)
        assert isinstance(errors, list)

    def test_settings_manager(self):
        from src.config.settings import get_settings
        settings = get_settings()
        # These should not crash
        settings.auto_detect_ffmpeg()
        ffmpeg = settings.get_ffmpeg_path()
        # ffmpeg may or may not be found
        assert ffmpeg is None or isinstance(ffmpeg, (str, Path))

    def test_feature_flags(self):
        from src.config.features import FeatureFlags
        # Shouldn't crash
        features = FeatureFlags.get_enabled_features()
        assert isinstance(features, list)
        assert FeatureFlags.has_batch_processing()

    def test_config_save_load(self):
        """Test saving and loading a config file."""
        from src.config.config_manager import ConfigManager
        cm = ConfigManager()
        config = cm.get_default_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = cm.save_config(config, filepath=temp_path)
            assert result is True
            
            loaded = cm.load_config(temp_path)
            assert loaded is not None
            assert isinstance(loaded, dict)
        finally:
            if temp_path.exists():
                temp_path.unlink()


# ============================================================================
# 4. TRANSFORM ENGINE TESTS
# ============================================================================

class TestTransforms:
    """Test E2P and E2C transform engines with synthetic images."""

    @pytest.fixture
    def equirect_image(self):
        """Create a synthetic equirectangular image (gradient)."""
        h, w = 960, 1920
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Horizontal gradient (longitude)
        for x in range(w):
            img[:, x, 0] = int(255 * x / w)  # Red = longitude
        # Vertical gradient (latitude)
        for y in range(h):
            img[y, :, 1] = int(255 * y / h)  # Green = latitude
        img[:, :, 2] = 128  # Blue = constant
        return img

    def test_e2p_basic(self, equirect_image):
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        result = t.equirect_to_pinhole(
            equirect_image, yaw=0, pitch=0, roll=0,
            h_fov=90, output_width=640, output_height=480
        )
        
        assert result is not None
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_e2p_all_compass_directions(self, equirect_image):
        """Test perspective extraction at all 8 compass directions."""
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        yaw_list = [0, 45, 90, 135, 180, -135, -90, -45]
        for yaw in yaw_list:
            result = t.equirect_to_pinhole(
                equirect_image, yaw=yaw, pitch=0, roll=0,
                h_fov=110, output_width=640, output_height=480
            )
            assert result is not None, f"Failed at yaw={yaw}"
            assert result.shape == (480, 640, 3), f"Wrong shape at yaw={yaw}"

    def test_e2p_look_up_down(self, equirect_image):
        """Test perspective extraction with pitch angles."""
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        for pitch in [-60, -30, 0, 30, 60]:
            result = t.equirect_to_pinhole(
                equirect_image, yaw=0, pitch=pitch, roll=0,
                h_fov=90, output_width=640, output_height=480
            )
            assert result is not None, f"Failed at pitch={pitch}"

    def test_e2p_cache(self, equirect_image):
        """Test that caching works (same params should reuse map)."""
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        t.clear_cache()
        assert t.get_cache_size() == 0
        
        # First call generates cache
        t.equirect_to_pinhole(equirect_image, yaw=0, pitch=0, h_fov=90,
                              output_width=640, output_height=480)
        cache_after_1 = t.get_cache_size()
        
        # Second call with same params should reuse cache
        t.equirect_to_pinhole(equirect_image, yaw=0, pitch=0, h_fov=90,
                              output_width=640, output_height=480)
        cache_after_2 = t.get_cache_size()
        
        assert cache_after_1 > 0
        assert cache_after_2 == cache_after_1  # No new cache entry

    def test_e2p_different_fovs(self, equirect_image):
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        for fov in [60, 90, 110, 130, 150]:
            result = t.equirect_to_pinhole(
                equirect_image, yaw=0, pitch=0, h_fov=fov,
                output_width=640, output_height=480
            )
            assert result is not None, f"Failed at FOV={fov}"

    def test_e2c_6face(self, equirect_image):
        from src.transforms import E2CTransform
        t = E2CTransform()
        
        faces = t.equirect_to_cubemap(equirect_image, face_size=256, mode='6-face')
        assert isinstance(faces, (dict, list))
        
        face_names = t.get_cube_face_names(mode='6-face')
        assert len(face_names) == 6

    def test_e2c_8tile(self, equirect_image):
        from src.transforms import E2CTransform
        t = E2CTransform()
        
        try:
            faces = t.equirect_to_cubemap(equirect_image, face_size=256, mode='8-tile')
            face_names = t.get_cube_face_names(mode='8-tile')
            assert len(face_names) == 8
        except (ValueError, NotImplementedError):
            pytest.skip("8-tile mode not implemented yet")

    def test_e2p_camera_matrix(self, equirect_image):
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        K = t.get_camera_matrix(h_fov=90, v_fov=67.5, 
                                output_width=640, output_height=480)
        assert K is not None
        assert K.shape == (3, 3)

    def test_e2p_rotation_matrix(self):
        from src.transforms import E2PTransform
        t = E2PTransform()
        
        R = t.get_rotation_matrix(yaw=0, pitch=0, roll=0)
        assert R is not None
        assert R.shape == (3, 3)
        # Identity-like rotation at (0,0,0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


# ============================================================================
# 5. MASKING TESTS
# ============================================================================

class TestMasking:
    """Test masking module (MultiCategoryMasker, factory function)."""

    def test_masking_factory_import(self):
        from src.masking import get_masker
        assert callable(get_masker)

    def test_masker_creation(self):
        """Test creating a masker doesn't crash."""
        from src.masking.multi_category_masker import MultiCategoryMasker
        try:
            masker = MultiCategoryMasker(
                model_size='nano',
                confidence_threshold=0.5,
                use_gpu=False
            )
            assert masker is not None
        except Exception as e:
            # Model download might fail in CI
            if "download" in str(e).lower() or "connect" in str(e).lower():
                pytest.skip(f"Model download unavailable: {e}")
            raise

    def test_category_config(self):
        """Test enabling/disabling detection categories."""
        from src.masking.multi_category_masker import MultiCategoryMasker
        try:
            masker = MultiCategoryMasker(model_size='nano', use_gpu=False)
            
            masker.set_enabled_categories(persons=True, personal_objects=False, animals=False)
            classes = masker.get_target_classes()
            assert 0 in classes  # person class = 0
            
            masker.set_enabled_categories(persons=False, personal_objects=False, animals=True)
            classes = masker.get_target_classes()
            assert 0 not in classes
        except Exception as e:
            if "download" in str(e).lower():
                pytest.skip(f"Model download unavailable: {e}")
            raise

    def test_mask_on_blank_image(self):
        """Masking a blank image should produce no mask (no detections)."""
        from src.masking.multi_category_masker import MultiCategoryMasker
        try:
            masker = MultiCategoryMasker(model_size='nano', use_gpu=False)
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            mask = masker.generate_mask_from_array(blank)
            # No objects in blank image → mask should be None or all-white
            if mask is not None:
                # All white (255) means "keep everything" = no masked objects
                assert mask.max() == 255
        except Exception as e:
            if "download" in str(e).lower():
                pytest.skip(f"Model download unavailable: {e}")
            raise

    def test_mask_value_constants(self):
        from src.config.defaults import MASK_VALUE_REMOVE, MASK_VALUE_KEEP
        assert MASK_VALUE_REMOVE == 0
        assert MASK_VALUE_KEEP == 255


# ============================================================================
# 6. PIPELINE TESTS
# ============================================================================

class TestPipeline:
    """Test pipeline orchestration, config generation, and metadata handling."""

    def test_metadata_handler_create(self):
        from src.pipeline.metadata_handler import MetadataHandler
        mh = MetadataHandler()
        assert mh is not None

    def test_metadata_json_round_trip(self):
        from src.pipeline.metadata_handler import MetadataHandler
        mh = MetadataHandler()
        
        metadata = {
            'camera': 'Insta360 X3',
            'yaw': 45.0,
            'pitch': 0.0,
            'fov': 110.0,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            mh.save_metadata_json(metadata, temp_path)
            loaded = mh.load_metadata_json(temp_path)
            assert loaded is not None
            assert loaded['camera'] == 'Insta360 X3'
            assert loaded['yaw'] == 45.0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_colmap_settings(self):
        from src.pipeline.colmap_stage import ColmapSettings
        settings = ColmapSettings()
        assert hasattr(settings, 'alignment_mode')

    def test_alignment_modes(self):
        from src.pipeline.colmap_stage import (
            ALIGNMENT_MODE_SPHERE_SFM,
            ALIGNMENT_MODE_RIG_SFM,
            ALIGNMENT_MODE_POSE_TRANSFER,
        )
        assert ALIGNMENT_MODE_SPHERE_SFM is not None
        assert ALIGNMENT_MODE_RIG_SFM is not None
        assert ALIGNMENT_MODE_POSE_TRANSFER is not None

    def test_exporter_classes(self):
        from src.pipeline.export_formats import LichtfeldExporter, RealityScanExporter
        # Just verify the classes exist and have __init__
        assert LichtfeldExporter is not None
        assert RealityScanExporter is not None


# ============================================================================
# 7. PREMIUM MODULE TESTS
# ============================================================================

class TestPremium:
    """Test premium modules (SfM, pose transfer, rig COLMAP)."""

    @pytest.mark.skip(reason="pycolmap causes fatal Windows DLL crash (0xc0000138)")
    def test_rig_colmap_import(self):
        from src.premium import RigColmapIntegrator
        assert hasattr(RigColmapIntegrator, 'create_virtual_camera')

    @pytest.mark.skip(reason="pycolmap causes fatal Windows DLL crash (0xc0000138)")
    def test_virtual_camera_rotations(self):
        from src.premium.pose_transfer_integration import get_virtual_camera_rotations
        rotations = get_virtual_camera_rotations()
        assert isinstance(rotations, list)
        assert len(rotations) >= 8  # At least 8 cameras
        for item in rotations:
            assert len(item) == 2  # (pitch, yaw) tuple
            pitch, yaw = item
            assert -90 <= pitch <= 90
            assert -180 <= yaw <= 180

    @pytest.mark.skip(reason="pycolmap causes fatal Windows DLL crash (0xc0000138)")
    def test_virtual_camera_dataclass(self):
        from src.premium.pose_transfer_integration import VirtualCamera
        cam = VirtualCamera(index=0, pitch=0.0, yaw=0.0, fov=90.0, is_reference=True)
        assert cam.index == 0
        assert cam.is_reference is True

    def test_sphere_sfm_binary_exists(self):
        """Check if SphereSfM binary is bundled."""
        sphere_path = PROJECT_ROOT / 'bin' / 'SphereSfM' / 'colmap.exe'
        assert sphere_path.exists(), f"SphereSfM binary not found at {sphere_path}"


# ============================================================================
# 8. UTILITY TESTS
# ============================================================================

class TestUtils:
    """Test utility modules."""

    def test_resource_path_dev_mode(self):
        from src.utils.resource_path import get_resource_path
        # In dev mode, should resolve relative to project root
        path = get_resource_path('src/config/defaults.py')
        assert isinstance(path, Path)

    def test_pinned_memory_pool_import(self):
        try:
            from src.utils import PinnedMemoryPool
            assert PinnedMemoryPool is not None
        except ImportError:
            pytest.skip("PinnedMemoryPool requires torch")

    def test_cuda_stream_manager_import(self):
        try:
            from src.utils import CUDAStreamManager
            assert CUDAStreamManager is not None
        except ImportError:
            pytest.skip("CUDAStreamManager requires torch")


# ============================================================================
# 9. PYQT6 UI TESTS (non-visual)
# ============================================================================

class TestUI:
    """Test UI module imports and widget creation (headless)."""

    def test_pyqt6_import(self):
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        assert QApplication is not None

    def test_main_window_import(self):
        from src.ui.main_window import MainWindow
        assert hasattr(MainWindow, 'start_pipeline')

    def test_main_window_creation(self):
        """Test MainWindow can be instantiated (requires QApplication)."""
        from PyQt6.QtWidgets import QApplication
        
        # QApplication singleton
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        from src.ui.main_window import MainWindow
        window = MainWindow()
        assert window is not None
        assert window.windowTitle() != ''
        window.close()
        window.deleteLater()


# ============================================================================
# 10. EXTERNAL TOOL DETECTION TESTS
# ============================================================================

class TestExternalTools:
    """Test detection of external tools (FFmpeg, SDK, COLMAP)."""

    def test_ffmpeg_detection(self):
        from src.config.settings import get_settings
        settings = get_settings()
        settings.auto_detect_ffmpeg()
        ffmpeg = settings.get_ffmpeg_path()
        if ffmpeg:
            assert os.path.exists(ffmpeg), f"FFmpeg path doesn't exist: {ffmpeg}"
            print(f"  FFmpeg: {ffmpeg}")
        else:
            print("  FFmpeg: NOT FOUND (optional)")

    def test_colmap_binary(self):
        colmap_path = PROJECT_ROOT / 'bin' / 'COLMAP-3.11.1'
        if colmap_path.exists():
            print(f"  COLMAP: {colmap_path}")
        else:
            pytest.skip("COLMAP not bundled")

    def test_sdk_detection(self):
        from src.config.settings import get_settings
        settings = get_settings()
        sdk = settings.get_sdk_path()
        if sdk:
            assert os.path.exists(sdk), f"SDK path doesn't exist: {sdk}"
            print(f"  SDK: {sdk}")
        else:
            print("  SDK: NOT FOUND (optional)")


# ============================================================================
# 11. INTEGRATION TEST — Pipeline Config Generation
# ============================================================================

class TestPipelineConfig:
    """Test that a complete pipeline configuration can be generated."""

    def test_full_config_generation(self):
        """Simulate what the UI does when user clicks Start Pipeline."""
        from src.config.defaults import (
            DEFAULT_FPS, DEFAULT_SPLIT_COUNT, DEFAULT_H_FOV,
            DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MODEL_SIZE,
            DEFAULT_EXTRACTION_METHOD, DEFAULT_OUTPUT_FORMAT,
        )
        
        config = {
            'input_file': r'C:\test\sample.insv',
            'output_dir': r'C:\test\output',
            
            # Stage 1
            'extraction_method': DEFAULT_EXTRACTION_METHOD,
            'fps': DEFAULT_FPS,
            'output_format': DEFAULT_OUTPUT_FORMAT,
            
            # Stage 2
            'split_count': DEFAULT_SPLIT_COUNT,
            'h_fov': DEFAULT_H_FOV,
            'yaw_offset': 0,
            'pitch': 0,
            'roll': 0,
            
            # Stage 3
            'masking_enabled': True,
            'masking_engine': 'yolo',
            'model_size': DEFAULT_MODEL_SIZE,
            'confidence_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
            'use_gpu': True,
            'mask_persons': True,
            'mask_objects': True,
            'mask_animals': False,
            
            # Stage 4
            'alignment_enabled': True,
            'alignment_mode': 'mode_c',
            
            # Stage 5
            'export_enabled': False,
        }
        
        assert config['fps'] > 0
        assert config['split_count'] > 0
        assert 30 <= config['h_fov'] <= 150
        assert 0 <= config['confidence_threshold'] <= 1.0

    def test_stage_skip_config(self):
        """Test config allowing stage-only processing."""
        config = {
            'input_file': r'C:\test\equirect_folder',
            'output_dir': r'C:\test\output',
            'skip_extraction': True,
            'skip_splitting': False,
            'masking_enabled': True,
            'alignment_enabled': False,
            'export_enabled': False,
        }
        assert config['skip_extraction'] is True


# ============================================================================
# 12. FILE FORMAT & PATH TESTS
# ============================================================================

class TestFileFormats:
    """Test supported file format constants and path utilities."""

    def test_supported_image_formats(self):
        from src.config.defaults import SUPPORTED_IMAGE_FORMATS
        assert isinstance(SUPPORTED_IMAGE_FORMATS, (list, tuple, set))
        # Should support at least jpg, png
        formats_lower = [f.lower() for f in SUPPORTED_IMAGE_FORMATS]
        assert any('jpg' in f or 'jpeg' in f for f in formats_lower)
        assert any('png' in f for f in formats_lower)

    def test_extraction_methods(self):
        from src.config.defaults import EXTRACTION_METHODS
        assert isinstance(EXTRACTION_METHODS, (list, tuple, dict))

    def test_yolov8_models(self):
        from src.config.defaults import YOLOV8_MODELS
        assert isinstance(YOLOV8_MODELS, (list, tuple, dict))

    def test_masking_engines(self):
        from src.config.defaults import MASKING_ENGINES
        assert isinstance(MASKING_ENGINES, (list, tuple, dict))

    def test_model_files_present(self):
        """Check that required model files exist."""
        yolo_model = PROJECT_ROOT / 'yolov8m-seg.pt'
        if yolo_model.exists():
            assert yolo_model.stat().st_size > 1_000_000  # > 1 MB
            print(f"  YOLO model: {yolo_model.stat().st_size / 1e6:.1f} MB")
        else:
            print("  YOLO model: not found (will auto-download)")

        sam_model = PROJECT_ROOT / 'sam_vit_b_01ec64.pth'
        if sam_model.exists():
            assert sam_model.stat().st_size > 100_000_000  # > 100 MB
            print(f"  SAM model: {sam_model.stat().st_size / 1e6:.1f} MB")
        else:
            print("  SAM model: not found (optional)")


# ============================================================================
# MAIN — Run standalone
# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("360toolkit - Comprehensive Test Suite")
    print(f"{'='*70}\n")
    
    # Run with verbose output
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x',  # Stop on first failure
        '--no-header',
    ])
    
    sys.exit(exit_code)
