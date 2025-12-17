r"""
Insta360 MediaSDK 3.0.5 Integration Module

PRIMARY EXTRACTION METHOD - Uses official Insta360 MediaSDK for highest quality stitching.

SDK Documentation: https://github.com/Insta360Develop/Desktop-MediaSDK-Cpp
SDK Location: C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1

Hardware Requirements:
- GPU with CUDA or Vulkan support (REQUIRED for v3.x)
- Windows 7+ (x64 only) or Ubuntu 22.04
- 8GB+ VRAM recommended for 8K output

TESTED STITCH TYPES (2024-12-04):
- dynamicstitch: PERFECT results, fast (~1.4s per frame) - RECOMMENDED DEFAULT
- aistitch + v2 model: PERFECT results, slower (~2.2s per frame) - HIGHEST QUALITY
- optflow: Good but can have noise in sky areas (~1.4s per frame)
- template: Fast but low quality - use for previews only

Key MediaSDK APIs:
- SetImageSequenceInfo(output_dir, IMAGE_TYPE): Export video frames as image sequence
- SetExportFrameSequence(frame_indices): Extract specific frames by index
- SetStitchType(STITCH_TYPE): dynamicstitch (recommended), aistitch, optflow, template
- EnableStitchFusion(True): Chromatic calibration for seamless blending (CRITICAL)
- EnableFlowState(True): FlowState stabilization
- EnableColorPlus(True, model_path): AI color enhancement
- SetAiStitchModelFile(model_path): ai_stitcher_model_v1.ins or v2.ins (v2 is better)
- SetOutputSize(width, height): Output resolution (must be 2:1 ratio)
- StartStitch(): Begin stitching process

Frame Extraction Workflow:
1. SetInputPath([video_file_1, video_file_2])  # Dual-track or single-track
2. SetImageSequenceInfo(output_dir, IMAGE_TYPE.JPEG)  # Or PNG
3. SetExportFrameSequence([0, 10, 20, 30, ...])  # Frame indices based on FPS
4. SetStitchType(STITCH_TYPE.DYNAMICSTITCH)  # PERFECT results (tested!)
5. SetAiStitchModelFile(ai_model_v2_path)  # Use v2 model for best quality
6. EnableStitchFusion(True)  # CRITICAL for seamless blending
7. SetOutputSize(7680, 3840)  # 8K output
8. StartStitch()  # Execute

Output Naming:
- With SetExportFrameSequence: {frame_index}.jpg (e.g., 10.jpg = frame #10)
- Without: {timestamp_ms}.jpg (e.g., 100.jpg = frame at 100ms)

Fallback Strategy:
If SDK executable not found or GPU unavailable -> FFmpeg Method (proven filter chain)
"""

import subprocess
import logging
import json
import cv2
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)


# Auto-detect SDK path (supports frozen PyInstaller apps)
def _get_default_sdk_path():
    """Get SDK path - checks environment variable first, then bundled location, then dev path."""
    # 1. Check environment variable (set by runtime hook)
    if 'INSTA360_SDK_PATH' in os.environ:
        logger.info(f"Using SDK from environment: {os.environ['INSTA360_SDK_PATH']}")
        return os.environ['INSTA360_SDK_PATH']
    
    # 2. Check if running as PyInstaller frozen app
    if hasattr(sys, '_MEIPASS'):
        # Try bundled SDK locations
        base_path = Path(sys._MEIPASS)
        bundled_locations = [
            base_path / 'sdk' / 'MediaSDK-3.0.5-20250619-win64',
            base_path / '_internal' / 'sdk' / 'MediaSDK-3.0.5-20250619-win64',
            base_path / 'sdk',
            base_path / '_internal' / 'sdk',
        ]
        for loc in bundled_locations:
            if loc.exists():
                logger.info(f"Found bundled SDK at: {loc}")
                return str(loc)
        logger.warning("Running as frozen app but SDK not found in bundle!")
    
    # 3. Fallback to dev machine path
    return r"C:\Users\User\Documents\Windows_CameraSDK-2.0.2-build1+MediaSDK-3.0.5-build1"

# SDK configuration
DEFAULT_SDK_PATH = _get_default_sdk_path()


# Stitch type presets (MediaSDK 3.0.5)
# NOTE: These values must match exactly what MediaSDKTest.exe expects!
# Run "MediaSDKTest.exe --help" to see valid stitch_type values.
# TESTED RESULTS:
#   - dynamicstitch: PERFECT results, fast (1.42s) - RECOMMENDED
#   - aistitch + v2 model: PERFECT results, slower (2.21s) - BEST QUALITY
#   - optflow: Good but noisy sky (1.42s)
#   - template: Fast but low quality
STITCH_TYPES = {
    'dynamic': 'dynamicstitch', # Dynamic Stitching - PERFECT results, fast (RECOMMENDED)
    'aistitch': 'aistitch',     # AI Stitching with v2 model - PERFECT results, slower
    'optflow': 'optflow',       # Optical Flow - good but can have noise
    'template': 'template'      # Template (FAST, lower quality)
}


# Quality presets
# TESTED: 'dynamicstitch' and 'aistitch' with v2 model produce PERFECT results
# The SDK CLI accepts: template, optflow, dynamicstitch, aistitch
QUALITY_PRESETS = {
    'best': {
        'stitch_type': 'dynamicstitch', # Dynamic Stitching - PERFECT results, recommended!
        'use_ai_model_v2': True,        # Use v2 model for best AI processing
        'enable_stitchfusion': True,    # Chromatic calibration (CRITICAL for seamless blending)
        'enable_flowstate': True,       # Stabilization
        'enable_colorplus': True,       # AI color enhancement
        'enable_denoise': True,         # AI denoising (reduces noise in sky/flat areas)
        'enable_defringe': True,        # Purple fringe removal
    },
    'good': {
        'stitch_type': 'dynamicstitch', # Dynamic Stitching - PERFECT results
        'use_ai_model_v2': False,       # Use v1 model (faster)
        'enable_stitchfusion': True,    # Keep chromatic calibration
        'enable_flowstate': True,       # Stabilization
        'enable_colorplus': False,
        'enable_denoise': False,
        'enable_defringe': False,
    },
    'balanced': {
        'stitch_type': 'optflow',       # Optical Flow - good quality, moderate speed
        'use_ai_model_v2': False,
        'enable_stitchfusion': True,    # Chromatic calibration
        'enable_flowstate': True,       # Stabilization
        'enable_colorplus': False,
        'enable_denoise': False,
        'enable_defringe': False,
    },
    'draft': {
        'stitch_type': 'template',      # Template - fastest, lowest quality
        'use_ai_model_v2': False,
        'enable_stitchfusion': False,
        'enable_flowstate': False,
        'enable_colorplus': False,
        'enable_denoise': False,
        'enable_defringe': False,
    }
}


class SDKExtractor:
    """
    Insta360 MediaSDK wrapper for frame extraction with stitching.
    
    Uses subprocess to call MediaSDK-Demo.exe with appropriate parameters.
    Implements SetImageSequenceInfo + SetExportFrameSequence workflow.
    """
    
    def __init__(self, sdk_path: Optional[str] = None):
        """
        Initialize SDK extractor.
        
        Args:
            sdk_path: Path to MediaSDK installation (auto-detects if None)
        """
        self.sdk_path = Path(sdk_path) if sdk_path else Path(DEFAULT_SDK_PATH)
        self.is_cancelled = False
        
        # Detect SDK executable (multiple possible locations)
        self.demo_exe = self._find_sdk_executable()
        
        # Find model files (search in multiple possible locations)
        self._locate_model_files()
        
        # Check SDK availability
        self.available = self._check_sdk_available()
        
        # Process handle for termination support
        self._current_process = None
        
        if self.available:
            logger.info(f"[OK] Insta360 MediaSDK detected: {self.demo_exe}")
            logger.info(f"[OK] AI Model V1: {self.ai_model_v1.exists()}")
            logger.info(f"[OK] AI Model V2: {self.ai_model_v2.exists()}")
        else:
            logger.warning("WARNING: Insta360 MediaSDK not found - will fallback to FFmpeg")
    
    def _find_sdk_executable(self) -> Optional[Path]:
        """Find MediaSDK executable in multiple possible locations."""
        possible_paths = [
            # Bundled structure - PREFER MediaSDKTest.exe as it supports CLI args
            self.sdk_path / "bin" / "MediaSDKTest.exe",
            self.sdk_path / "bin" / "RealTimeStitcherSDKTest.exe",
            
            # MediaSDK 3.0.5 structure
            self.sdk_path / "MediaSDK-3.0.5-20250619-win64" / "MediaSDK" / "bin" / "MediaSDKTest.exe",
            self.sdk_path / "MediaSDK-3.0.5-20250619-win64" / "MediaSDK" / "bin" / "RealTimeStitcherSDKTest.exe",
            self.sdk_path / "MediaSDK" / "bin" / "MediaSDKTest.exe",
            self.sdk_path / "MediaSDK" / "bin" / "RealTimeStitcherSDKTest.exe",
            
            # Demo executable (common in older SDK versions)
            self.sdk_path / "Demo" / "Windows" / "Media SDK" / "MediaSDK-Demo.exe",
            self.sdk_path / "Demo" / "Windows" / "Media SDK" / "Release" / "MediaSDK-Demo.exe",
            
            # Alternate locations
            self.sdk_path / "bin" / "MediaSDK-Demo.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found SDK executable: {path}")
                # Store SDK base directory for model file search
                # Path is like: .../MediaSDK/bin/MediaSDKTest.exe
                # Go up 2 levels: MediaSDK/bin -> MediaSDK
                self.sdk_base = path.parent.parent
                return path
        
        logger.warning(f"MediaSDK executable not found in {len(possible_paths)} checked locations")
        return None
    
    def _locate_model_files(self):
        """Find model files in SDK installation (multiple possible locations)."""
        if not self.demo_exe or not hasattr(self, 'sdk_base'):
            # SDK not found, use default paths
            self.ai_model_v1 = self.sdk_path / "data" / "ai_stitch_model_v1.ins"
            self.ai_model_v2 = self.sdk_path / "data" / "ai_stitch_model_v2.ins"
            self.colorplus_model = self.sdk_path / "data" / "colorplus_model.ins"
            self.denoise_model = self.sdk_path / "data" / "jpg_denoise_9d006262.ins"
            self.defringe_model = self.sdk_path / "modelfile" / "defringe_hr_dynamic_7b56e80f.ins"
            self.deflicker_model = self.sdk_path / "modelfile" / "deflicker_86ccba0d.ins"
            return
        
        # Search in SDK-relative locations
        possible_model_dirs = [
            self.sdk_base / "modelfile",  # MediaSDK 3.0.5 location
            self.sdk_base / "data",       # Older SDK versions
            self.sdk_path / "data",       # Root SDK path
            self.sdk_path / "modelfile",  # Root SDK path alternate
        ]
        
        # Find AI stitch model v1 (try different names)
        self.ai_model_v1 = self._find_model_file(
            possible_model_dirs,
            ["ai_stitcher_model_v1.ins", "ai_stitch_model_v1.ins"]
        )
        
        # Find AI stitch model v2
        self.ai_model_v2 = self._find_model_file(
            possible_model_dirs,
            ["ai_stitcher_model_v2.ins", "ai_stitch_model_v2.ins"]
        )
        
        # Find Color Plus model
        self.colorplus_model = self._find_model_file(
            possible_model_dirs,
            ["colorplus_model.ins"]
        )
        
        # Find denoise model
        self.denoise_model = self._find_model_file(
            possible_model_dirs,
            ["jpg_denoise_9d006262.ins", "denoise_model.ins"]
        )
        
        # Find defringe model
        self.defringe_model = self._find_model_file(
            possible_model_dirs,
            ["defringe_hr_dynamic_7b56e80f.ins", "defringe_model.ins"]
        )
        
        # Find deflicker model
        self.deflicker_model = self._find_model_file(
            possible_model_dirs,
            ["deflicker_86ccba0d.ins", "deflicker_model.ins"]
        )
    
    def _find_model_file(self, search_dirs: List[Path], filenames: List[str]) -> Path:
        """Search for a model file in multiple directories with multiple possible names."""
        for directory in search_dirs:
            if not directory.exists():
                continue
            for filename in filenames:
                model_path = directory / filename
                if model_path.exists():
                    logger.info(f"Found model: {model_path}")
                    return model_path
        
        # Not found - return first possible path as fallback
        return search_dirs[0] / filenames[0] if search_dirs and filenames else Path("not_found.ins")
    
    def _check_sdk_available(self) -> bool:
        """Check if SDK is properly installed and GPU is available."""
        if not self.demo_exe or not self.demo_exe.exists():
            logger.error("SDK executable not found")
            return False
        
        # Check AI model availability (optional - can use other stitch types)
        if not self.ai_model_v1.exists():
            logger.warning("AI stitch model v1 not found - will use Optical Flow stitching")
            # SDK is still available, just without AI stitching
        
        # Check GPU availability (CUDA required for best/good presets)
        self._gpu_available = self._check_gpu_available()
        
        return True
    
    def _check_gpu_available(self) -> bool:
        """Check if NVIDIA GPU is available for SDK stitching."""
        # Check for nvcuda.dll in System32
        sys32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
        nvcuda = sys32 / 'nvcuda.dll'
        if nvcuda.exists():
            logger.info(f"[OK] GPU available: Found nvcuda.dll at {nvcuda}")
            return True
        
        # Check via CUDA environment variable
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and Path(cuda_path).exists():
            logger.info(f"[OK] GPU available: Found CUDA at {cuda_path}")
            return True
        
        logger.warning("[WARNING] GPU not available - nvcuda.dll not found. SDK will use CPU fallback (template stitching).")
        return False
    
    def is_available(self) -> bool:
        """Check if SDK is available for use."""
        return self.available
    
    def cancel(self):
        """Cancel ongoing SDK extraction operation"""
        self.is_cancelled = True
        logger.info("SDKExtractor cancellation requested")
    
    def extract_frames(
        self,
        input_path: str,
        output_dir: str,
        fps: float = 1.0,
        quality: str = 'best',
        resolution: Optional[Tuple[int, int]] = None,
        output_format: str = 'jpg',
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[str]:
        """
        Extract frames using MediaSDK with stitching.
        
        Args:
            input_path: Path to .insv video file
            output_dir: Output directory for frames
            fps: Extraction rate (frames per second)
            quality: 'best', 'good', or 'draft'
            resolution: (width, height) for output (None = original)
            output_format: 'jpg' or 'png'
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (None = full video)
            progress_callback: Callback(progress_percent)
        
        Returns:
            List of extracted frame paths
        """
        if not self.available:
            raise RuntimeError("MediaSDK not available - use FFmpeg fallback")
        
        # Check GPU availability and adjust quality preset if needed
        if not self._gpu_available and quality in ['best', 'good', 'balanced']:
            logger.warning(f"[GPU FALLBACK] '{quality}' preset requires GPU. Falling back to 'draft' (template stitching).")
            logger.warning("[GPU FALLBACK] For best quality, run on a PC with NVIDIA GPU.")
            quality = 'draft'
        
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info for frame calculation
        video_info = self._get_video_info(input_path)
        total_frames = video_info.get('total_frames', 0)
        video_fps = video_info.get('fps', 24.0)
        duration = video_info.get('duration', 0)
        
        # Apply time range constraints
        if end_time is None:
            end_time = duration
        
        # Convert time range to frame range
        start_frame = int(start_time * video_fps)
        end_frame = min(int(end_time * video_fps), total_frames)
        
        # Calculate frame indices to extract based on desired FPS within time range
        frame_interval = max(1, int(video_fps / fps))
        frame_indices = list(range(start_frame, end_frame, frame_interval))
        
        logger.info(f"Time range: {start_time}s - {end_time}s (frames {start_frame} - {end_frame})")
        logger.info(f"Extracting {len(frame_indices)} frames from {total_frames} total")
        logger.info(f"Frame interval: {frame_interval} (video FPS: {video_fps}, target FPS: {fps})")
        
        # Detect dual-track vs single-track
        input_files = self._detect_input_files(input_path)
        
        # Build MediaSDK command
        cmd = self._build_extraction_command(
            input_files=input_files,
            output_dir=output_dir,
            frame_indices=frame_indices,
            quality=quality,
            resolution=resolution,
            output_format=output_format
        )
        
        logger.info(f"Running MediaSDK extraction...")
        logger.info(f"Command: {' '.join(str(c) for c in cmd)}")
        
        # Execute SDK with Popen for termination support
        try:
            # Set CWD to SDK bin directory to ensure DLLs are found
            sdk_cwd = self.demo_exe.parent if self.demo_exe else None
            
            # Prepare environment with _internal in PATH (for msvcp140.dll etc)
            env = os.environ.copy()
            
            # Add SDK bin directory to PATH (CRITICAL for finding DLLs in same folder)
            if sdk_cwd:
                env['PATH'] = str(sdk_cwd) + os.pathsep + env.get('PATH', '')
            
            if getattr(sys, 'frozen', False):
                exe_dir = Path(sys.executable).parent
                internal_dir = exe_dir / '_internal'
                if internal_dir.exists():
                    env['PATH'] = str(internal_dir) + os.pathsep + env.get('PATH', '')
                    logger.debug(f"Added {internal_dir} to SDK environment PATH")
                    
                    # CRITICAL FIX: Copy missing dependencies from _internal to SDK bin
                    # MediaSDKTest.exe might not respect PATH for some DLLs (DLL Hell)
                    if sdk_cwd:
                        try:
                            import shutil
                            # List of DLLs that often cause issues if not in the same folder
                            # zlib.dll: often needed by opencv/libpng
                            # libiomp5md.dll: Intel OpenMP
                            # msvcp140.dll: VC++ runtime (sometimes version mismatch)
                            # concrt140.dll: Concurrency runtime
                            deps_to_copy = ['zlib.dll', 'libiomp5md.dll', 'concrt140.dll', 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll']
                            
                            for dep in deps_to_copy:
                                src_dll = internal_dir / dep
                                dst_dll = sdk_cwd / dep
                                if src_dll.exists() and not dst_dll.exists():
                                    logger.info(f"Copying missing dependency to SDK bin: {dep}")
                                    shutil.copy2(src_dll, dst_dll)
                        except Exception as e:
                            logger.warning(f"Failed to copy dependencies: {e}")

            # DIAGNOSTIC: Log PATH and CWD
            logger.info(f"SDK CWD: {sdk_cwd}")
            # Log full PATH to debug missing system paths
            logger.info(f"SDK PATH: {env['PATH']}") 
            
            # Check for critical DLLs in SDK folder
            if sdk_cwd:
                found_dlls = [f.name for f in sdk_cwd.glob('*.dll')]
                logger.info(f"DLLs in SDK bin: {found_dlls}")
                
                # Check for nvcuda.dll in System32
                sys32 = Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32'
                nvcuda = sys32 / 'nvcuda.dll'
                if nvcuda.exists():
                    logger.info(f"Found nvcuda.dll at {nvcuda}")
                else:
                    logger.warning(f"nvcuda.dll NOT FOUND in {sys32} (GPU stitching may fail)")

            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=sdk_cwd,  # CRITICAL: Run from SDK bin folder to find DLLs
                env=env       # CRITICAL: Include _internal in PATH
            )
            
            # Calculate timeout based on frame count
            # Empirical: ~0.5-1.5s per frame for optflow stitching + 180s overhead
            # For 7680x3840 output with stitching and stitchfusion, can take 1-2s per frame
            # Add 50% safety margin for system load variations
            frame_count = len(frame_indices)
            base_time = frame_count * 1.5  # 1.5s per frame average (conservative)
            overhead = 180  # 3 minutes overhead for SDK startup/shutdown/finalization
            safety_margin = (base_time + overhead) * 0.5  # 50% safety margin
            estimated_time = base_time + overhead + safety_margin
            estimated_time = max(600, min(7200, estimated_time))  # Min 10min, Max 2 hours
            logger.info(f"[INFO] Timeout set to {estimated_time:.0f}s for {frame_count} frames (~{estimated_time/frame_count:.2f}s/frame with safety margin)")
            
            # Monitor progress in real-time with smart completion detection
            import time
            import threading
            
            extracted_frames = []
            last_count = 0
            completion_detected = False
            no_change_duration = 0
            
            def monitor_progress():
                """Monitor extraction progress by counting output files and detect completion"""
                nonlocal extracted_frames, last_count, completion_detected, no_change_duration
                consecutive_no_change = 0
                
                while self._current_process and self._current_process.poll() is None:
                    try:
                        current_files = list(output_dir.glob('*.*'))
                        current_count = len(current_files)
                        
                        if current_count > last_count:
                            # Progress detected
                            last_count = current_count
                            consecutive_no_change = 0
                            no_change_duration = 0
                            progress_percent = int((last_count / frame_count * 100)) if frame_count > 0 else 0
                            if progress_callback:
                                progress_callback(progress_percent)
                            logger.debug(f"Progress: {last_count}/{frame_count} frames ({progress_percent}%)")
                        else:
                            # No new files - check for completion
                            consecutive_no_change += 1
                            no_change_duration += 2  # 2 seconds per check
                            
                            # If we have all expected frames and no changes for 10+ seconds → completed
                            if current_count >= frame_count and consecutive_no_change >= 5:
                                logger.info(f"[DETECTION] All {current_count} frames extracted, no changes for {no_change_duration}s")
                                completion_detected = True
                                # Terminate the waiting process
                                if self._current_process and self._current_process.poll() is None:
                                    logger.info("[DETECTION] SDK appears complete - terminating wait")
                                    try:
                                        self._current_process.terminate()
                                    except:
                                        pass
                                break
                            
                            # If we have substantial frames (>90%) and no changes for 30s → likely completed
                            elif current_count > 0 and current_count >= frame_count * 0.9 and consecutive_no_change >= 15:
                                logger.info(f"[DETECTION] {current_count}/{frame_count} frames extracted, no changes for {no_change_duration}s")
                                completion_detected = True
                                if self._current_process and self._current_process.poll() is None:
                                    logger.info("[DETECTION] SDK appears complete - terminating wait")
                                    try:
                                        self._current_process.terminate()
                                    except:
                                        pass
                                break
                    except Exception as e:
                        logger.debug(f"Error monitoring progress: {e}")
                    
                    time.sleep(2)  # Check every 2 seconds
            
            # Start progress monitor thread
            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()
            
            # Wait for completion with timeout
            stdout = ""
            stderr = ""
            returncode = 0
            
            try:
                stdout, stderr = self._current_process.communicate(timeout=estimated_time)
                returncode = self._current_process.returncode
                self._current_process = None
                
                # Final progress update
                if progress_callback:
                    progress_callback(100)
                
                if returncode != 0 and not completion_detected:
                    # Handle specific error codes
                    if returncode == 3221225781 or returncode == -1073741515: # 0xC0000135
                        error_msg = (
                            "MediaSDK failed to start (Missing DLL Dependency).\n"
                            "This usually means:\n"
                            "1. NVIDIA Drivers are missing or outdated (nvcuda.dll required)\n"
                            "2. Visual C++ Redistributable 2015-2022 is missing\n"
                            "3. System DLLs are corrupted\n"
                            "Please install latest NVIDIA Drivers and VC++ Redistributable."
                        )
                        logger.error(f"[CRITICAL] {error_msg}")
                        raise RuntimeError(error_msg)
                    elif returncode == 3221225477 or returncode == -1073741819: # 0xC0000005
                        error_msg = "MediaSDK crashed (Access Violation). Likely GPU driver incompatibility."
                        logger.error(f"[CRITICAL] {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)
                
                logger.info("[OK] MediaSDK extraction completed successfully!")
                if stdout:
                    logger.debug(f"SDK output: {stdout}")
                    
            except subprocess.TimeoutExpired:
                # Timeout - check if extraction actually completed
                logger.warning(f"[WARNING] SDK timeout after {estimated_time:.0f}s")
                
                # Check current file count
                current_files = list(output_dir.glob('*.*'))
                current_count = len(current_files)
                
                if current_count >= frame_count:
                    logger.info(f"[OK] All {current_count} frames extracted - SDK completed despite timeout")
                    completion_detected = True
                    # Kill the waiting process
                    try:
                        self._current_process.terminate()
                        self._current_process.wait(timeout=5)
                    except:
                        self._current_process.kill()
                        self._current_process.wait()
                    finally:
                        self._current_process = None
                elif current_count >= frame_count * 0.9:
                    logger.warning(f"[WARNING] {current_count}/{frame_count} frames extracted - accepting partial result")
                    completion_detected = True
                    try:
                        self._current_process.terminate()
                        self._current_process.wait(timeout=5)
                    except:
                        self._current_process.kill()
                        self._current_process.wait()
                    finally:
                        self._current_process = None
                else:
                    logger.error(f"[ERROR] Only {current_count}/{frame_count} frames extracted before timeout")
                    try:
                        self._current_process.kill()
                        self._current_process.wait()
                    except:
                        pass
                    finally:
                        self._current_process = None
                    raise RuntimeError(f"SDK timeout with insufficient frames: {current_count}/{frame_count}")
                
                logger.debug(f"SDK output: {stdout}")
            
            # Collect extracted frame paths
            extracted_frames = self._collect_frame_paths(output_dir, frame_indices, output_format)
            
            # Validate extraction - should have at least 70% of expected frames
            expected_count = len(frame_indices)
            actual_count = len(extracted_frames)
            success_rate = (actual_count / expected_count * 100) if expected_count > 0 else 0
            
            if actual_count == 0:
                raise RuntimeError(f"SDK produced no output frames in {output_dir}")
            elif success_rate < 70:
                logger.warning(f"[WARNING] Low extraction rate: {actual_count}/{expected_count} ({success_rate:.1f}%)")
            
            logger.info(f"[OK] Extracted {actual_count}/{expected_count} frames ({success_rate:.1f}%)")
            return extracted_frames
            
        except subprocess.CalledProcessError as e:
            # Check for known SDK issues
            error_msg = e.stderr if e.stderr else str(e)
            
            # Exit code 3221225786 (0xC000013A) = STATUS_CONTROL_C_EXIT (process terminated)
            # Exit code -1073741819 (0xC0000005) = STATUS_ACCESS_VIOLATION (crash)
            if 'FrameTypeTimelapseQuat failed' in error_msg:
                logger.warning("[WARNING] SDK metadata read error (timelapse data missing)")
                logger.warning("INFO: This is a known SDK issue with some .insv files")
            elif 'no device found' in error_msg:
                logger.warning("[WARNING] SDK failed: No GPU device found (expected on CPU-only systems)")
                raise RuntimeError("MediaSDK requires GPU (no device found)")
            elif e.returncode == 3221225786:
                logger.warning("[WARNING] SDK process was terminated (user cancel or timeout)")
            else:
                logger.error(f"[ERROR] MediaSDK extraction failed (exit code {e.returncode})")

            if e.stderr:
                logger.error("[SDK STDERR]\n%s", e.stderr.rstrip())
            else:
                logger.error(f"Error details: {error_msg}")
            raise RuntimeError(f"MediaSDK extraction failed: {error_msg[:200]}")
    
    def stop(self):
        """Stop currently running SDK process."""
        if self._current_process and self._current_process.poll() is None:
            logger.warning("[WARNING] Terminating SDK process...")
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=5)
                logger.info("[OK] SDK process terminated")
            except:
                logger.warning("[WARNING] Force killing SDK process...")
                self._current_process.kill()
                self._current_process.wait()
            finally:
                self._current_process = None
    
    def _get_video_info(self, video_path: Path) -> Dict:
        """Get video metadata using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Could not open video with OpenCV: {video_path}")
            return {'total_frames': 0, 'fps': 24.0, 'duration': 0}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height
        }
    
    def _detect_input_files(self, input_path: Path) -> List[str]:
        """
        Detect if input is dual-track or single-track.
        
        Insta360 5.7K+ videos use dual-track:
        - VID_XXX_00_XXX.insv (main track)
        - VID_XXX_10_XXX.insv (second track)
        
        X4 cameras use single-track with embedded dual video streams.
        """
        input_files = [str(input_path)]
        
        # Check for dual-track pattern: _00_ → _10_
        if "_00_" in input_path.name:
            second_track = input_path.parent / input_path.name.replace("_00_", "_10_")
            if second_track.exists():
                input_files.append(str(second_track))
                logger.info(f"Detected dual-track video: {input_path.name} + {second_track.name}")
        
        return input_files
    
    def _build_extraction_command(
        self,
        input_files: List[str],
        output_dir: Path,
        frame_indices: List[int],
        quality: str,
        resolution: Optional[Tuple[int, int]],
        output_format: str
    ) -> List[str]:
        """Build MediaSDK command line arguments."""
        cmd = [str(self.demo_exe)]
        
        # Input files (SetInputPath)
        cmd.extend(["-inputs"] + input_files)
        
        # Output directory for image sequence (SetImageSequenceInfo)
        cmd.extend(["-image_sequence_dir", str(output_dir)])
        
        # Image format (IMAGE_TYPE enum)
        image_type = "png" if output_format.lower() == "png" else "jpg"
        cmd.extend(["-image_type", image_type])
        
        # Frame indices to extract (SetExportFrameSequence)
        # CRITICAL: Use dash-separated format (NOT comma-separated)
        # Correct: "0-24-48" | Wrong: "0,24,48"
        frame_seq = "-".join(str(i) for i in frame_indices)
        cmd.extend(["-export_frame_index", frame_seq])  # Singular, not plural!
        
        # Quality preset
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['best'])
        
        # Determine stitch type
        stitch_type_key = preset['stitch_type']
        
        # Select AI model based on preset preference
        # v2 model produces PERFECT results (tested), prefer it when available
        use_v2 = preset.get('use_ai_model_v2', False)
        if use_v2 and self.ai_model_v2.exists():
            ai_model = self.ai_model_v2
            logger.info(f"[OK] Using AI Model V2 (best quality): {ai_model.name}")
        elif self.ai_model_v1.exists():
            ai_model = self.ai_model_v1
            logger.info(f"[OK] Using AI Model V1: {ai_model.name}")
        else:
            ai_model = None
            logger.warning("[WARNING] No AI model found")
        
        # Add AI model for stitch types that benefit from it
        # NOTE: Even dynamicstitch can use AI model for better results
        if ai_model and ai_model.exists():
            cmd.extend(["-ai_stitching_model", str(ai_model)])
        
        # Set stitch type (SetStitchType)
        stitch_type_value = STITCH_TYPES.get(stitch_type_key, 'dynamicstitch')
        cmd.extend(["-stitch_type", stitch_type_value])
        logger.info(f"[SDK] Stitch Method: {stitch_type_value.upper()}")
        
        # CRITICAL: Enable chromatic calibration for seamless blending
        if preset.get('enable_stitchfusion', False):
            cmd.append("-enable_stitchfusion")
        
        # Stabilization (EnableFlowState)
        if preset.get('enable_flowstate', False):
            cmd.append("-enable_flowstate")
        
        # Color Plus (EnableColorPlus)
        if preset.get('enable_colorplus', False):
            cmd.append("-enable_colorplus")
            if self.colorplus_model.exists():
                cmd.extend(["-colorplus_model", str(self.colorplus_model)])
        
        # Denoise (EnableSequenceDenoise) - requires model path for image denoising
        if preset.get('enable_denoise', False):
            cmd.append("-enable_denoise")
            if self.denoise_model.exists():
                cmd.extend(["-image_denoise_model", str(self.denoise_model)])
        
        # Defringe (EnableDefringe)
        if preset.get('enable_defringe', False):
            cmd.append("-enable_defringe")
            if self.defringe_model.exists():
                cmd.extend(["-defringe_model", str(self.defringe_model)])
        
        # Output resolution (SetOutputSize - must be 2:1 ratio)
        if resolution:
            width, height = resolution
            cmd.extend(["-output_size", f"{width}x{height}"])
        else:
            # Default to 8K for best quality
            cmd.extend(["-output_size", "7680x3840"])
        
        return cmd
    
    def _collect_frame_paths(
        self,
        output_dir: Path,
        frame_indices: List[int],
        output_format: str
    ) -> List[str]:
        """
        Collect paths to extracted frames.
        
        MediaSDK names files by frame index: 0.jpg, 10.jpg, 20.jpg, etc.
        Tolerates missing frames (up to 30%) due to SDK behavior on some systems.
        """
        ext = f".{output_format.lower()}"
        extracted = []
        missing = []

        # First try exact index-based filenames (0.jpg, 12.jpg, ...)
        for idx in frame_indices:
            frame_path = output_dir / f"{idx}{ext}"
            if frame_path.exists():
                extracted.append(str(frame_path))
            else:
                missing.append(idx)

        # If no files were collected by index, try a more permissive discovery
        if len(extracted) == 0:
            logger.debug("No index-named frames found - trying permissive image discovery")
            image_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            permissive = sorted([str(p) for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
            if permissive:
                logger.info(f"Found {len(permissive)} image files via permissive discovery")
                return permissive

        # Log summary instead of per-frame warnings
        if missing:
            if len(missing) <= 10:
                logger.debug(f"Missing frames: {missing}")
            else:
                logger.debug(f"Missing {len(missing)} frames (first 10: {missing[:10]}...)")

        return extracted


# Convenience functions for batch_orchestrator.py integration

def extract_frames_sdk(
    input_path: str,
    output_dir: str,
    fps: float = 1.0,
    quality: str = 'best',
    resolution: Optional[Tuple[int, int]] = None,
    output_format: str = 'jpg',
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    progress_callback: Optional[Callable[[int], None]] = None
) -> List[str]:
    """
    Extract frames using MediaSDK (convenience function).
    
    Args:
        input_path: Path to .insv video
        output_dir: Output directory
        fps: Extraction rate
        quality: 'best', 'good', 'draft'
        resolution: Output resolution (None = 8K default)
        output_format: 'jpg' or 'png'
        start_time: Start time in seconds
        end_time: End time in seconds (None = video end)
        progress_callback: Progress callback
    
    Returns:
        List of extracted frame paths
    
    Raises:
        RuntimeError: If SDK not available or extraction fails
    """
    extractor = SDKExtractor()
    
    if not extractor.is_available():
        raise RuntimeError(
            "MediaSDK not available. Please install SDK or use FFmpeg fallback.\n"
            f"Expected SDK path: {DEFAULT_SDK_PATH}"
        )
    
    return extractor.extract_frames(
        input_path=input_path,
        output_dir=output_dir,
        fps=fps,
        quality=quality,
        resolution=resolution,
        output_format=output_format,
        start_time=start_time,
        end_time=end_time,
        progress_callback=progress_callback
    )


def is_sdk_available() -> bool:
    """Check if MediaSDK is available."""
    extractor = SDKExtractor()
    return extractor.is_available()
