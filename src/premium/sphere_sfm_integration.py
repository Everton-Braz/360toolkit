"""
SphereSfM Integration Module for 360ToolKit

Provides two reconstruction modes for 360° panoramic images:

Panorama SfM (SphereSfM Native): 
  - Uses SphereSfM binary (modified COLMAP) for direct spherical feature matching
  - Matches features in equirectangular space (handles wrap-around)
  - Outputs spherical reconstruction, then converts to cubic (perspective) views
  - Best for: Urban scenes, street-level captures

Perspective Reconstruction (Rig-based SfM):
  - Renders virtual perspective cameras from panoramas
  - Uses COLMAP rig constraints for alignment
  - Proven 100% registration rate
  - Best for: Indoor scenes, small-scale captures

References:
  - SphereSfM: https://github.com/json87/SphereSfM
  - Jiang et al. "3D reconstruction of spherical images" (IJRS 2024)
"""

import logging
import shlex
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import os

logger = logging.getLogger(__name__)

# Preferred bundled COLMAP release location (user-provided GPU build)
DEFAULT_COLMAP_RELEASE_PATH = Path(__file__).parent.parent.parent / "bin" / "colmap" / "colmap.exe"
# Preferred SphereSfM release path (user-downloaded V1.2)
DEFAULT_SPHERESFM_RELEASE_PATH = Path(__file__).parent.parent.parent / "bin" / "SphereSfM-2024-12-14" / "colmap.exe"
# Legacy SphereSfM binary location
LEGACY_SPHERESFM_PATH = Path(__file__).parent.parent.parent / "bin" / "SphereSfM" / "colmap.exe"

DEFAULT_SPHERESFM_FEATURE_FLAGS = (
    "--ImageReader.single_camera 1 "
    "--SiftExtraction.max_num_orientations 2 "
    "--SiftExtraction.peak_threshold 0.00667 "
    "--SiftExtraction.edge_threshold 10.0"
)

DEFAULT_SPHERESFM_MATCHER_FLAGS = (
    "--SequentialMatching.quadratic_overlap 1 "
    "--SequentialMatching.loop_detection 0 "
    "--SiftMatching.max_ratio 0.8 "
    "--SiftMatching.max_distance 0.7 "
    "--SiftMatching.cross_check 1 "
    "--SiftMatching.max_error 4.0 "
    "--SiftMatching.confidence 0.999 "
    "--SiftMatching.max_num_trials 10000 "
    "--SiftMatching.min_inlier_ratio 0.25"
)

DEFAULT_SPHERESFM_MAPPER_FLAGS = (
    "--Mapper.ba_refine_focal_length 0 "
    "--Mapper.ba_refine_principal_point 0 "
    "--Mapper.ba_refine_extra_params 0 "
    "--Mapper.init_min_num_inliers 100 "
    "--Mapper.init_num_trials 200 "
    "--Mapper.init_max_error 4 "
    "--Mapper.init_max_forward_motion 0.95 "
    "--Mapper.init_min_tri_angle 16 "
    "--Mapper.abs_pose_min_num_inliers 50 "
    "--Mapper.abs_pose_max_error 8 "
    "--Mapper.abs_pose_min_inlier_ratio 0.25 "
    "--Mapper.max_reg_trials 3 "
    "--Mapper.tri_min_angle 1.5 "
    "--Mapper.tri_max_transitivity 1 "
    "--Mapper.tri_ignore_two_view_tracks 1 "
    "--Mapper.filter_max_reproj_error 4 "
    "--Mapper.filter_min_tri_angle 1.5 "
    "--Mapper.multiple_models 1"
)

DEFAULT_SPHERESFM_RUNTIME_MAX_IMAGE_SIZE = 3200
DEFAULT_SPHERESFM_RUNTIME_MAX_NUM_FEATURES = 8192
DEFAULT_SPHERESFM_RUNTIME_SEQUENTIAL_OVERLAP = 10
DEFAULT_SPHERESFM_RUNTIME_MIN_NUM_MATCHES = 15
DEFAULT_SPHERESFM_RUNTIME_MAX_NUM_MATCHES = 32000


def _split_cli_flags(flags: str) -> List[str]:
    """Split CLI flags while preserving quoted values when possible."""
    if not flags or not flags.strip():
        return []
    try:
        return shlex.split(flags, posix=False)
    except ValueError:
        return flags.split()


def _override_cli_flag_values(flags: str, overrides: Dict[str, str]) -> str:
    """Override CLI flag values while preserving unmodified flags."""
    tokens = _split_cli_flags(flags)
    if not tokens:
        tokens = []

    result: List[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            value = None
            if "=" in token:
                token, inline_value = token.split("=", 1)
                value = inline_value
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                value = tokens[index + 1]
                index += 1

            if token in overrides:
                result.extend([token, str(overrides[token])])
                seen.add(token)
            else:
                result.append(token)
                if value is not None:
                    result.append(value)
        else:
            result.append(token)
        index += 1

    for token, value in overrides.items():
        if token not in seen:
            result.extend([token, str(value)])

    return " ".join(result)


def _normalize_binary_candidate(path_value) -> Optional[Path]:
    """Normalize binary candidate (supports direct exe/bat path or folder path)."""
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_dir():
        for filename in ("colmap.exe", "colmap.bat", "colmap.cmd"):
            candidate = p / filename
            if candidate.exists():
                return candidate
    return p


def _compose_binary_command(binary_path: Path, args: List[str]) -> List[str]:
    """Build subprocess command, including .bat/.cmd wrappers on Windows."""
    suffix = binary_path.suffix.lower()
    if suffix in {".bat", ".cmd"}:
        return ["cmd", "/c", str(binary_path)] + args
    return [str(binary_path)] + args


def _probe_binary(binary_path: Path) -> Tuple[bool, bool]:
    """Check whether a COLMAP-compatible binary is runnable and supports sphere mapper option."""
    if not binary_path.exists():
        return False, False

    run_env = os.environ.copy()
    binary_dir = str(binary_path.parent)
    run_env["PATH"] = binary_dir + os.pathsep + run_env.get("PATH", "")

    try:
        help_cmd = _compose_binary_command(binary_path, ["help"])
        help_proc = subprocess.run(
            help_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            cwd=binary_dir,
            env=run_env,
        )
        help_text = f"{help_proc.stdout}\n{help_proc.stderr}".lower()

        is_runnable = help_proc.returncode == 0 and (
            "available commands" in help_text or "usage:" in help_text or "colmap" in help_text
        )
        if not is_runnable:
            return False, False

        mapper_cmd = _compose_binary_command(binary_path, ["mapper", "-h"])
        mapper_proc = subprocess.run(
            mapper_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            cwd=binary_dir,
            env=run_env,
        )
        mapper_text = f"{mapper_proc.stdout}\n{mapper_proc.stderr}".lower()
        supports_sphere = "mapper.sphere_camera" in mapper_text
        return True, supports_sphere
    except Exception:
        return False, False


def resolve_spheresfm_binary_path(settings=None, override_path: Optional[Path] = None) -> Tuple[Path, List[str]]:
    """
    Resolve SphereSfM/COLMAP binary path.

    Search order (SphereSfM-first):
    1) Explicit override path
    2) settings.sphere_alignment_path (if present)
    3) Environment variables (SPHERESFM_PATH, SPHERESFM_BIN)
    4) Default bundled SphereSfM path
    5) Common local SphereSfM installs
    6) Generic COLMAP fallbacks (COLMAP_PATH, bundled colmap, PATH lookup)
    """
    candidates: List[Path] = []

    def add_candidate(value):
        normalized = _normalize_binary_candidate(value)
        if normalized is not None and normalized not in candidates:
            candidates.append(normalized)

    add_candidate(override_path)

    settings_path = getattr(settings, "sphere_alignment_path", None) if settings is not None else None
    add_candidate(settings_path)

    for env_name in ("SPHERESFM_PATH", "SPHERESFM_BIN"):
        env_value = os.environ.get(env_name)
        if env_value:
            add_candidate(env_value)

    add_candidate(DEFAULT_SPHERESFM_RELEASE_PATH)
    add_candidate(LEGACY_SPHERESFM_PATH)

    home = Path.home()
    spheresfm_common_paths = [
        home / "Documents" / "APLICATIVOS" / "360ToolKit" / "bin" / "SphereSfM-2024-12-14" / "colmap.exe",
        home / "Documents" / "APLICATIVOS" / "360toolkit" / "bin" / "SphereSfM-2024-12-14" / "colmap.exe",
        home / "Documents" / "APLICATIVOS" / "360ToolKit" / "bin" / "SphereSfM" / "colmap.exe",
        home / "Documents" / "APLICATIVOS" / "360toolkit" / "bin" / "SphereSfM" / "colmap.exe",
        home / "Documents" / "SphereSfM" / "colmap.exe",
    ]
    for path in spheresfm_common_paths:
        add_candidate(path)

    # Generic fallback binaries (only if SphereSfM-specific candidates do not work)
    colmap_env_value = os.environ.get("COLMAP_PATH")
    if colmap_env_value:
        add_candidate(colmap_env_value)

    add_candidate(DEFAULT_COLMAP_RELEASE_PATH)

    generic_colmap_paths = [
        home / "Documents" / "APLICATIVOS" / "360toolkit" / "bin" / "colmap" / "colmap.exe",
        home / "Documents" / "APLICATIVOS" / "360ToolKit" / "bin" / "colmap" / "colmap.exe",
        home / "Documents" / "colmap-x64-windows-cuda" / "colmap.bat",
        home / "Documents" / "colmap-x64-windows-cuda" / "colmap.exe",
        Path("C:/colmap/colmap.exe"),
    ]
    for path in generic_colmap_paths:
        add_candidate(path)

    for command_name in ("colmap", "colmap.exe", "colmap.bat", "colmap.cmd"):
        which_result = shutil.which(command_name)
        if which_result:
            add_candidate(which_result)

    existing_candidates = [candidate for candidate in candidates if candidate.exists()]

    for candidate in existing_candidates:
        runnable, _ = _probe_binary(candidate)
        if runnable:
            return candidate, [str(c) for c in candidates]

    if existing_candidates:
        return existing_candidates[0], [str(c) for c in candidates]

    fallback = candidates[0] if candidates else DEFAULT_SPHERESFM_RELEASE_PATH
    return fallback, [str(c) for c in candidates]


class SphereSfMIntegrator:
    """
    SphereSfM integration for direct spherical feature matching.
    
    Panorama SfM: Uses native SphereSfM binary for end-to-end spherical reconstruction.
    """
    
    def __init__(self, settings, spheresfm_path: Optional[Path] = None):
        self.settings = settings
        self.spheresfm_path, self.searched_paths = resolve_spheresfm_binary_path(settings, spheresfm_path)
        self._command_help_cache: Dict[str, str] = {}
        self.supports_sphere_camera_mapper = self._detect_mapper_sphere_support()
        
        # Validate SphereSfM binary exists
        if not self.spheresfm_path.exists():
            logger.warning(f"SphereSfM binary not found at {self.spheresfm_path}")
    
    def is_available(self) -> bool:
        """Check if SphereSfM binary is available."""
        return self.spheresfm_path.exists()

    def _get_command_help(self, command_name: str) -> str:
        """Return lowercase help text for a specific COLMAP command."""
        if command_name in self._command_help_cache:
            return self._command_help_cache[command_name]

        if not self.spheresfm_path.exists():
            return ""

        try:
            cmd = _compose_binary_command(self.spheresfm_path, [command_name, "-h"])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            help_text = f"{proc.stdout}\n{proc.stderr}".lower()
        except Exception:
            help_text = ""

        self._command_help_cache[command_name] = help_text
        return help_text

    def _resolve_supported_option(self, command_name: str, candidates: List[str]) -> str:
        """Return the first supported option for a command, falling back to the first candidate."""
        help_text = self._get_command_help(command_name)
        for candidate in candidates:
            if candidate.lstrip("-").lower() in help_text:
                return candidate
        return candidates[0]

    def _filter_supported_cli_args(self, command_name: str, args: List[str]) -> List[str]:
        """Drop unsupported option/value pairs for a specific COLMAP command."""
        help_text = self._get_command_help(command_name)
        if not help_text:
            return list(args)

        filtered: List[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if not token.startswith("--"):
                filtered.append(token)
                index += 1
                continue

            option_name = token.split("=", 1)[0].lstrip("-").lower()
            has_inline_value = "=" in token
            value_token = None
            has_separate_value = False
            if not has_inline_value and index + 1 < len(args) and not args[index + 1].startswith("--"):
                value_token = args[index + 1]
                has_separate_value = True

            if option_name in help_text:
                filtered.append(token)
                if has_separate_value:
                    filtered.append(value_token)
            else:
                logger.warning("[SphereSfM] Dropping unsupported %s option for %s: %s", command_name, self.spheresfm_path.name, token)

            index += 2 if has_separate_value else 1

        return filtered

    def _prepare_colmap_masks(
        self,
        images_dir: Path,
        masks_dir: Path,
        colmap_masks_dir: Path,
    ) -> Optional[Path]:
        """Convert project mask naming into COLMAP/SphereSfM per-image mask naming.

        COLMAP expects a mask folder where each image `foo.jpg` maps to
        `foo.jpg.png` below the same relative subpath. Our pipeline commonly
        stores masks as `foo_mask.png`. This helper copies the available masks
        into a temporary folder with COLMAP-compatible names.
        """
        if not images_dir.exists() or not masks_dir.exists():
            return None

        colmap_masks_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        rel_images: Dict[str, str] = {}
        for img in images_dir.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in valid_exts:
                continue
            rel_path = img.relative_to(images_dir)
            rel_images[img.stem] = rel_path.as_posix()

        converted = 0
        for mask_file in masks_dir.rglob("*.png"):
            if not mask_file.is_file():
                continue

            rel_mask = mask_file.relative_to(masks_dir)
            mask_stem = rel_mask.stem
            candidate_stems = [
                mask_stem,
                mask_stem.removesuffix("_mask"),
                mask_stem.replace("_mask", ""),
            ]

            image_rel = None
            for candidate in candidate_stems:
                if candidate in rel_images:
                    image_rel = rel_images[candidate]
                    break

            if image_rel is None:
                continue

            dest = colmap_masks_dir / f"{image_rel}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(mask_file, dest)
            converted += 1

        if converted == 0:
            logger.warning("[SphereSfM] No per-image masks could be matched in %s", masks_dir)
            return None

        logger.info("[SphereSfM] Prepared %d COLMAP-compatible per-image masks in %s", converted, colmap_masks_dir)
        return colmap_masks_dir

    def _detect_mapper_sphere_support(self) -> bool:
        """Detect if current binary supports --Mapper.sphere_camera option."""
        if not self.spheresfm_path.exists():
            return False
        try:
            cmd = _compose_binary_command(self.spheresfm_path, ["mapper", "-h"])
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            help_text = f"{proc.stdout}\n{proc.stderr}".lower()
            return "mapper.sphere_camera" in help_text
        except Exception:
            return False
    
    def get_version(self) -> Optional[str]:
        """Get SphereSfM version string."""
        if not self.is_available():
            return None
        try:
            result = subprocess.run(
                [str(self.spheresfm_path), "--version"],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL
            )
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            logger.debug(f"Could not get SphereSfM version: {e}")
            return "Unknown"
    
    def _run_command(self, args: List[str], progress_callback: Optional[Callable] = None) -> subprocess.CompletedProcess:
        """Run SphereSfM command with logging."""
        cmd = _compose_binary_command(self.spheresfm_path, args)
        logger.info(f"Running: {' '.join(cmd)}")
        
        if progress_callback:
            progress_callback(f"SphereSfM: {args[0]}...")
        
        # Add SphereSfM directory to PATH for DLL dependencies
        env = os.environ.copy()
        spheresfm_dir = str(self.spheresfm_path.parent)
        env["PATH"] = spheresfm_dir + os.pathsep + env.get("PATH", "")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='replace',  # Avoid UnicodeDecodeError on binary output
            env=env,
            cwd=spheresfm_dir,  # Run from SphereSfM directory
            stdin=subprocess.DEVNULL
        )
        
        if result.returncode != 0:
            # Log both stdout and stderr — COLMAP variants sometimes print errors to stdout
            out = (result.stdout or '').strip()[-2000:]
            err = (result.stderr or '').strip()[-2000:]
            logger.error(f"SphereSfM '{args[0]}' failed (exit {result.returncode})")
            if err:
                logger.error(f"  stderr: {err}")
            if out:
                logger.error(f"  stdout: {out}")
        
        return result

    @staticmethod
    def _format_cmd_error(result: subprocess.CompletedProcess) -> str:
        """Return a compact error string from a failed CompletedProcess."""
        parts = []
        err = (result.stderr or '').strip()[-1500:]
        out = (result.stdout or '').strip()[-1500:]
        if err:
            parts.append(err)
        if out and out != err:
            parts.append(out)
        return '\n'.join(parts) if parts else f'(exit {result.returncode} — no output)'

    def _cleanup_partial_outputs(self, output_dir: Path) -> None:
        """Best-effort cleanup for failed Panorama SfM runs."""
        cleanup_targets = [
            output_dir / "sparse-cubic",
            output_dir / "sparse",
            output_dir / "database.db",
        ]
        for target in cleanup_targets:
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()
            except Exception as e:
                logger.debug(f"Cleanup skipped for {target}: {e}")

    def _is_feature_extractor_retryable(self, result: subprocess.CompletedProcess) -> bool:
        """Return whether a failed extractor run should be retried with safer settings."""
        if result.returncode == 0:
            return False

        output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        retryable_markers = (
            "access violation",
            "segmentation fault",
            "stack buffer",
            "bad allocation",
            "out of memory",
        )
        windows_access_violation_codes = {3221225477, -1073741819}
        return result.returncode in windows_access_violation_codes or any(marker in output for marker in retryable_markers)

    def _is_mapper_initialization_failure(self, result: subprocess.CompletedProcess) -> bool:
        """Detect mapper failures that benefit from relaxed initialization thresholds."""
        output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        markers = (
            "no good initial image pair",
            "initial image pair",
            "could not find initial image pair",
            "failed to create an initial image pair",
        )
        return any(marker in output for marker in markers)

    def _build_feature_extractor_args(
        self,
        *,
        database_path: Path,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        camera_model: str,
        is_sphere_model: bool,
        width: int,
        height: int,
        gpu_flag: str,
        max_image_size: int,
        max_num_features: int,
        feature_flags: str,
        extra_args: str,
        extractor_use_gpu_option: str,
        extractor_max_image_size_option: str,
    ) -> List[str]:
        extractor_args = [
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(frames_dir),
            "--ImageReader.camera_model", camera_model,
        ]

        if is_sphere_model:
            extractor_args.extend(["--ImageReader.camera_params", f"1,{width},{height}"])

        if masks_dir and masks_dir.exists():
            prepared_masks_dir = self._prepare_colmap_masks(
                frames_dir,
                masks_dir,
                output_dir / "masks_colmap",
            )
            if prepared_masks_dir is not None:
                extractor_args.extend(["--ImageReader.mask_path", str(prepared_masks_dir)])
            else:
                mask_file = masks_dir / "camera_mask.png"
                if mask_file.exists():
                    extractor_args.extend(["--ImageReader.camera_mask_path", str(mask_file)])

        extractor_args.extend([
            extractor_use_gpu_option, gpu_flag,
            extractor_max_image_size_option, str(max_image_size),
            "--SiftExtraction.max_num_features", str(max_num_features),
        ])
        extractor_args.extend(self._filter_supported_cli_args(
            "feature_extractor",
            _split_cli_flags(feature_flags or DEFAULT_SPHERESFM_FEATURE_FLAGS),
        ))
        extractor_args.extend(self._filter_supported_cli_args(
            "feature_extractor",
            _split_cli_flags(extra_args),
        ))
        return extractor_args

    def _run_feature_extractor_with_fallback(
        self,
        *,
        database_path: Path,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        camera_model: str,
        is_sphere_model: bool,
        width: int,
        height: int,
        gpu_flag: str,
        max_image_size: int,
        max_num_features: int,
        feature_flags: str,
        extra_args: str,
        extractor_use_gpu_option: str,
        extractor_max_image_size_option: str,
        progress_callback: Optional[Callable],
    ) -> Tuple[Optional[subprocess.CompletedProcess], Optional[str]]:
        attempts: List[Tuple[str, int, int]] = []
        for label, profile in [
            ("requested", (max_image_size, max_num_features)),
            ("safe", (DEFAULT_SPHERESFM_RUNTIME_MAX_IMAGE_SIZE, DEFAULT_SPHERESFM_RUNTIME_MAX_NUM_FEATURES)),
            ("low-memory", (2560, 6144)),
        ]:
            if profile not in [(size, features) for _, size, features in attempts]:
                attempts.append((label, profile[0], profile[1]))

        failure_messages: List[str] = []
        last_result: Optional[subprocess.CompletedProcess] = None

        for attempt_index, (label, attempt_image_size, attempt_num_features) in enumerate(attempts, start=1):
            if database_path.exists():
                database_path.unlink()

            db_result = self._run_command([
                "database_creator",
                "--database_path", str(database_path)
            ], progress_callback)
            if db_result.returncode != 0:
                return db_result, f"Database creation failed: {self._format_cmd_error(db_result)}"

            logger.info(
                "[SphereSfM] Feature extraction attempt %d/%d (%s): max_image_size=%d, max_num_features=%d",
                attempt_index,
                len(attempts),
                label,
                attempt_image_size,
                attempt_num_features,
            )
            if progress_callback:
                progress_callback(
                    f"Extracting features ({camera_model}, {label} profile {attempt_image_size}px/{attempt_num_features})..."
                )

            extractor_args = self._build_feature_extractor_args(
                database_path=database_path,
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                camera_model=camera_model,
                is_sphere_model=is_sphere_model,
                width=width,
                height=height,
                gpu_flag=gpu_flag,
                max_image_size=attempt_image_size,
                max_num_features=attempt_num_features,
                feature_flags=feature_flags,
                extra_args=extra_args,
                extractor_use_gpu_option=extractor_use_gpu_option,
                extractor_max_image_size_option=extractor_max_image_size_option,
            )
            last_result = self._run_command(extractor_args, progress_callback)
            if last_result.returncode == 0:
                return last_result, None

            failure_summary = (
                f"{label} profile ({attempt_image_size}px/{attempt_num_features}) failed: "
                f"{self._format_cmd_error(last_result)}"
            )
            failure_messages.append(failure_summary)
            logger.warning("[SphereSfM] %s", failure_summary)

            if not self._is_feature_extractor_retryable(last_result):
                break

        if last_result is None:
            return None, "Feature extraction did not run"
        return last_result, "Feature extraction failed across all profiles:\n" + "\n".join(failure_messages)

    def _build_mapper_args(
        self,
        *,
        database_path: Path,
        frames_dir: Path,
        sparse_path: Path,
        min_num_matches: int,
        mapper_flags: str,
        extra_args: str,
        is_sphere_model: bool,
    ) -> List[str]:
        mapper_args = [
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(frames_dir),
            "--output_path", str(sparse_path),
            "--Mapper.min_num_matches", str(min_num_matches),
        ]

        mapper_args.extend(self._filter_supported_cli_args(
            "mapper",
            _split_cli_flags(mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS),
        ))
        mapper_args.extend(self._filter_supported_cli_args(
            "mapper",
            _split_cli_flags(extra_args),
        ))

        mapper_flag_tokens = mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS
        if (
            is_sphere_model
            and self.supports_sphere_camera_mapper
            and "--Mapper.sphere_camera" not in mapper_flag_tokens
        ):
            mapper_args.extend(["--Mapper.sphere_camera", "1"])
        elif is_sphere_model and not self.supports_sphere_camera_mapper:
            logger.warning("Binary does not expose --Mapper.sphere_camera; continuing without it.")

        return mapper_args

    def _run_mapper_with_fallback(
        self,
        *,
        database_path: Path,
        frames_dir: Path,
        sparse_path: Path,
        mapper_flags: str,
        extra_args: str,
        min_num_matches: int,
        is_sphere_model: bool,
        progress_callback: Optional[Callable],
    ) -> Tuple[subprocess.CompletedProcess, Optional[str]]:
        attempts: List[Tuple[str, str, int]] = [
            ("strict", mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS, min_num_matches),
            (
                "lenient",
                _override_cli_flag_values(
                    mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS,
                    {
                        "--Mapper.init_min_num_inliers": "50",
                        "--Mapper.init_num_trials": "500",
                        "--Mapper.init_min_tri_angle": "8",
                        "--Mapper.abs_pose_min_num_inliers": "15",
                        "--Mapper.abs_pose_max_error": "12",
                    },
                ),
                min(min_num_matches, 15),
            ),
            (
                "single-model-lenient",
                _override_cli_flag_values(
                    mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS,
                    {
                        "--Mapper.init_min_num_inliers": "50",
                        "--Mapper.init_num_trials": "500",
                        "--Mapper.init_min_tri_angle": "8",
                        "--Mapper.abs_pose_min_num_inliers": "15",
                        "--Mapper.abs_pose_max_error": "12",
                        "--Mapper.max_reg_trials": "6",
                        "--Mapper.tri_ignore_two_view_tracks": "0",
                        "--Mapper.filter_max_reproj_error": "6",
                        "--Mapper.multiple_models": "0",
                    },
                ),
                min(min_num_matches, 12),
            ),
            (
                "aggressive-registration",
                _override_cli_flag_values(
                    mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS,
                    {
                        "--Mapper.init_min_num_inliers": "40",
                        "--Mapper.init_num_trials": "800",
                        "--Mapper.init_max_forward_motion": "0.99",
                        "--Mapper.init_min_tri_angle": "6",
                        "--Mapper.abs_pose_min_num_inliers": "12",
                        "--Mapper.abs_pose_max_error": "16",
                        "--Mapper.abs_pose_min_inlier_ratio": "0.15",
                        "--Mapper.max_reg_trials": "8",
                        "--Mapper.tri_min_angle": "0.75",
                        "--Mapper.tri_max_transitivity": "2",
                        "--Mapper.tri_ignore_two_view_tracks": "0",
                        "--Mapper.filter_max_reproj_error": "8",
                        "--Mapper.filter_min_tri_angle": "0.75",
                        "--Mapper.multiple_models": "0",
                    },
                ),
                min(min_num_matches, 10),
            ),
            (
                "ultra-lenient",
                _override_cli_flag_values(
                    mapper_flags or DEFAULT_SPHERESFM_MAPPER_FLAGS,
                    {
                        "--Mapper.init_min_num_inliers": "30",
                        "--Mapper.init_num_trials": "1000",
                        "--Mapper.init_max_error": "8",
                        "--Mapper.init_max_forward_motion": "0.99",
                        "--Mapper.init_min_tri_angle": "4",
                        "--Mapper.abs_pose_min_num_inliers": "10",
                        "--Mapper.abs_pose_max_error": "16",
                        "--Mapper.abs_pose_min_inlier_ratio": "0.1",
                        "--Mapper.max_reg_trials": "5",
                        "--Mapper.tri_min_angle": "0.5",
                        "--Mapper.tri_max_transitivity": "2",
                        "--Mapper.tri_ignore_two_view_tracks": "0",
                        "--Mapper.filter_max_reproj_error": "8",
                        "--Mapper.filter_min_tri_angle": "0.5",
                    },
                ),
                min(min_num_matches, 10),
            ),
        ]

        last_result: Optional[subprocess.CompletedProcess] = None
        failure_messages: List[str] = []
        model_path = sparse_path / "0"

        for attempt_index, (label, attempt_flags, attempt_min_matches) in enumerate(attempts, start=1):
            shutil.rmtree(sparse_path, ignore_errors=True)
            sparse_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                "[SphereSfM] Mapper attempt %d/%d (%s): min_num_matches=%d",
                attempt_index,
                len(attempts),
                label,
                attempt_min_matches,
            )
            if progress_callback:
                progress_callback(f"Running reconstruction (mapper, {label})...")

            mapper_args = self._build_mapper_args(
                database_path=database_path,
                frames_dir=frames_dir,
                sparse_path=sparse_path,
                min_num_matches=attempt_min_matches,
                mapper_flags=attempt_flags,
                extra_args=extra_args,
                is_sphere_model=is_sphere_model,
            )
            last_result = self._run_command(mapper_args, progress_callback)

            if last_result.returncode == 0 and model_path.exists():
                return last_result, None

            failure_summary = f"{label} mapper failed: {self._format_cmd_error(last_result)}"
            if last_result.returncode == 0 and not model_path.exists():
                failure_summary = f"{label} mapper produced no reconstruction model"
            failure_messages.append(failure_summary)
            logger.warning("[SphereSfM] %s", failure_summary)

            if attempt_index == 1 and not self._is_mapper_initialization_failure(last_result):
                break
            if attempt_index > 1 and not self._is_mapper_initialization_failure(last_result):
                break

        if last_result is None:
            raise RuntimeError("Mapper did not execute")
        return last_result, "Mapper failed across all profiles:\n" + "\n".join(failure_messages)
    
    def run_alignment_mode_a(
        self,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Panorama SfM: Native SphereSfM reconstruction.
        
        Workflow:
        1. database_creator - Create COLMAP database
        2. feature_extractor - Extract features with SPHERE camera model
        3. sequential_matcher / spatial_matcher - Match features
        4. mapper - Incremental SfM with spherical geometry
        5. sphere_cubic_reprojector - Convert to perspective views
        
        Args:
            frames_dir: Directory containing equirectangular images
            masks_dir: Optional directory with camera masks
            output_dir: Output directory for reconstruction
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with success status and output paths
        """
        if not self.is_available():
            return {
                'success': False,
                'error': f'SphereSfM binary not found at {self.spheresfm_path}'
            }
        
        output_dir.mkdir(parents=True, exist_ok=True)

        def fail_with_cleanup(error_message: str) -> Dict:
            self._cleanup_partial_outputs(output_dir)
            return {'success': False, 'error': error_message}
        
        # Setup paths
        database_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"
        sparse_path.mkdir(exist_ok=True)
        
        if database_path.exists():
            database_path.unlink()
        
        # Count input images (search recursively to handle lens subfolders if not yet flattened)
        image_files = sorted(
            p for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
            for p in frames_dir.rglob(ext)
        )
        if not image_files:
            return fail_with_cleanup('No input images found')
        
        # Detect image dimensions (used for SPHERE model camera params)
        from PIL import Image
        test_img = Image.open(image_files[0])
        width, height = test_img.size
        
        # Read SphereSfM params from settings (populated by batch_orchestrator from UI)
        camera_model = getattr(self.settings, 'spheresfm_camera_model', 'SPHERE')
        use_gpu = getattr(self.settings, 'spheresfm_use_gpu', False)
        matching_method = getattr(self.settings, 'spheresfm_matching_method', 'sequential')
        max_image_size = getattr(self.settings, 'spheresfm_max_image_size', DEFAULT_SPHERESFM_RUNTIME_MAX_IMAGE_SIZE)
        max_num_features = getattr(self.settings, 'spheresfm_max_num_features', DEFAULT_SPHERESFM_RUNTIME_MAX_NUM_FEATURES)
        sequential_overlap = getattr(self.settings, 'spheresfm_sequential_overlap', DEFAULT_SPHERESFM_RUNTIME_SEQUENTIAL_OVERLAP)
        min_num_matches = getattr(self.settings, 'spheresfm_min_num_matches', DEFAULT_SPHERESFM_RUNTIME_MIN_NUM_MATCHES)
        feature_flags = getattr(self.settings, 'spheresfm_feature_flags', DEFAULT_SPHERESFM_FEATURE_FLAGS)
        matcher_flags = getattr(self.settings, 'spheresfm_matcher_flags', DEFAULT_SPHERESFM_MATCHER_FLAGS)
        mapper_flags = getattr(self.settings, 'spheresfm_mapper_flags', DEFAULT_SPHERESFM_MAPPER_FLAGS)
        extra_args = getattr(self.settings, 'spheresfm_extra_args', '')
        gpu_flag = "1" if use_gpu else "0"

        # SPHERE model uses fixed focal length (f=1), others use standard COLMAP calibration
        is_sphere_model = camera_model in ('SPHERE', 'SIMPLE_SPHERE')
        extractor_use_gpu_option = self._resolve_supported_option(
            "feature_extractor",
            ["--SiftExtraction.use_gpu", "--FeatureExtraction.use_gpu"],
        )
        extractor_max_image_size_option = self._resolve_supported_option(
            "feature_extractor",
            ["--SiftExtraction.max_image_size", "--FeatureExtraction.max_image_size"],
        )

        logger.info(
            f"SphereSfM Panorama: {len(image_files)} images at {width}x{height}, "
            f"camera={camera_model}, match={matching_method}, gpu={use_gpu}"
        )
        
        try:
            # Step 2: Feature extraction
            if extra_args.strip():
                logger.debug("Global SphereSfM extra args appended to feature_extractor")

            result, extractor_error = self._run_feature_extractor_with_fallback(
                database_path=database_path,
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_dir=output_dir,
                camera_model=camera_model,
                is_sphere_model=is_sphere_model,
                width=width,
                height=height,
                gpu_flag=gpu_flag,
                max_image_size=max_image_size,
                max_num_features=max_num_features,
                feature_flags=feature_flags,
                extra_args=extra_args,
                extractor_use_gpu_option=extractor_use_gpu_option,
                extractor_max_image_size_option=extractor_max_image_size_option,
                progress_callback=progress_callback,
            )
            if extractor_error:
                return fail_with_cleanup(extractor_error)
            
            # Step 3: Feature matching
            if progress_callback:
                progress_callback(f"Matching features ({matching_method})...")

            # Build matching command based on chosen method
            if matching_method == 'exhaustive':
                matching_cmd = "exhaustive_matcher"
            elif matching_method == 'vocab_tree':
                matching_cmd = "vocab_tree_matcher"
            else:
                matching_cmd = "sequential_matcher"

            matcher_user_flags = self._filter_supported_cli_args(
                matching_cmd,
                _split_cli_flags(matcher_flags or DEFAULT_SPHERESFM_MATCHER_FLAGS),
            )
            global_extra_flags = self._filter_supported_cli_args(
                matching_cmd,
                _split_cli_flags(extra_args),
            )

            matcher_use_gpu_option = self._resolve_supported_option(
                matching_cmd,
                ["--SiftMatching.use_gpu", "--FeatureMatching.use_gpu"],
            )
            matcher_max_num_matches_option = self._resolve_supported_option(
                matching_cmd,
                ["--SiftMatching.max_num_matches", "--FeatureMatching.max_num_matches"],
            )
            sift_match_flags = [
                matcher_use_gpu_option, gpu_flag,
                matcher_max_num_matches_option, str(max(DEFAULT_SPHERESFM_RUNTIME_MAX_NUM_MATCHES, max_num_features)),
                "--SiftMatching.min_num_inliers", str(min_num_matches),
            ]

            if matching_method == 'exhaustive':
                matching_args = [matching_cmd, "--database_path", str(database_path)] + sift_match_flags
            elif matching_method == 'vocab_tree':
                matching_args = [matching_cmd, "--database_path", str(database_path)] + sift_match_flags
            else:
                matching_args = [
                    matching_cmd,
                    "--database_path", str(database_path),
                    "--SequentialMatching.overlap", str(sequential_overlap),
                ] + sift_match_flags

            matching_args.extend(matcher_user_flags)
            matching_args.extend(global_extra_flags)
            
            result = self._run_command(matching_args, progress_callback)
            
            if result.returncode != 0:
                return fail_with_cleanup(f'Feature matching failed: {self._format_cmd_error(result)}')
            
            # Step 4: Incremental mapping
            result, mapper_error = self._run_mapper_with_fallback(
                database_path=database_path,
                frames_dir=frames_dir,
                sparse_path=sparse_path,
                mapper_flags=mapper_flags,
                extra_args=extra_args,
                min_num_matches=min_num_matches,
                is_sphere_model=is_sphere_model,
                progress_callback=progress_callback,
            )
            if mapper_error:
                return fail_with_cleanup(mapper_error)

            model_path = sparse_path / "0"
            
            # Convert binary model to text format for easier analysis
            cameras_bin = model_path / "cameras.bin"
            cameras_txt = model_path / "cameras.txt"
            if cameras_bin.exists() and not cameras_txt.exists():
                try:
                    self._run_command([
                        "model_converter",
                        "--input_path", str(model_path),
                        "--output_path", str(model_path),
                        "--output_type", "TXT"
                    ], progress_callback)
                except Exception as e:
                    logger.debug(f"Model conversion to TXT failed: {e}")
            
            # Count results
            images_file = model_path / "images.txt"
            num_registered = 0
            if images_file.exists():
                with open(images_file, 'r') as f:
                    lines = [l for l in f if not l.startswith('#') and l.strip()]
                    num_registered = len(lines) // 2  # 2 lines per image
            
            points_file = model_path / "points3D.txt"
            num_points = 0
            if points_file.exists():
                with open(points_file, 'r') as f:
                    num_points = sum(1 for l in f if not l.startswith('#') and l.strip())
            
            logger.info(f"SphereSfM Panorama complete: {num_registered} images, {num_points} points")
            
            return {
                'success': True,
                'mode': 'Panorama SfM (SphereSfM)',
                'colmap_output': model_path,
                'cubic_output': None,
                'frames_dir': frames_dir,
                'num_aligned': num_registered,
                'num_points': num_points
            }
            
        except Exception as e:
            logger.error(f"SphereSfM Panorama Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._cleanup_partial_outputs(output_dir)
            return {'success': False, 'error': str(e)}
    
class DualModeAlignmentIntegrator:
    """
    Dual-mode alignment integrator providing both Panorama SfM and Perspective Reconstruction.
    
    Panorama SfM: SphereSfM (native spherical matching)
    Perspective Reconstruction: Rig-based SfM (virtual perspectives + rig constraints)
    """
    
    MODE_A = "sphere_sfm"      # SphereSfM native
    MODE_B = "rig_sfm"         # Rig-based (hybrid)
    
    def __init__(self, settings):
        self.settings = settings
        self.sphere_sfm = SphereSfMIntegrator(settings)
        
        # Import rig SfM integrator
        from .rig_colmap_integration import RigColmapIntegrator
        self.rig_sfm = RigColmapIntegrator(settings)
    
    def is_mode_a_available(self) -> bool:
        """Check if SphereSfM (Panorama SfM) is available."""
        return self.sphere_sfm.is_available()
    
    def get_available_modes(self) -> List[str]:
        """Get list of available alignment modes."""
        modes = [self.MODE_B]  # Rig SfM always available (uses pycolmap)
        if self.is_mode_a_available():
            modes.insert(0, self.MODE_A)
        return modes
    
    def get_mode_description(self, mode: str) -> str:
        """Get human-readable description of a mode."""
        descriptions = {
            self.MODE_A: "SphereSfM (Native Spherical) - Direct equirectangular matching",
            self.MODE_B: "Rig-based SfM (Recommended) - Virtual perspectives with rig constraints"
        }
        return descriptions.get(mode, "Unknown mode")
    
    def run_alignment(
        self,
        mode: str,
        frames_dir: Path,
        masks_dir: Optional[Path],
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run alignment with specified mode.
        
        Args:
            mode: Either MODE_A ('sphere_sfm') or MODE_B ('rig_sfm')
            frames_dir: Directory containing input images
            masks_dir: Optional mask directory
            output_dir: Output directory
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with reconstruction results
        """
        if mode == self.MODE_A:
            if not self.is_mode_a_available():
                return {
                    'success': False,
                    'error': 'SphereSfM binary not available. Please use Rig-based SfM.'
                }
            logger.info("Running Panorama SfM: SphereSfM (native spherical)")
            return self.sphere_sfm.run_alignment_mode_a(
                frames_dir, masks_dir, output_dir, progress_callback
            )
        
        elif mode == self.MODE_B:
            logger.info("Running Perspective Reconstruction: Rig-based SfM (virtual perspectives)")
            return self.rig_sfm.run_alignment(
                frames_dir, masks_dir, output_dir, progress_callback
            )
        
        else:
            return {'success': False, 'error': f'Unknown alignment mode: {mode}'}


def create_pole_mask(width: int, height: int, pole_degrees: float = 15) -> 'np.ndarray':
    """
    Create a mask to exclude polar regions from equirectangular images.
    
    Polar regions in ERP have high distortion and unreliable features.
    
    Args:
        width: Image width
        height: Image height  
        pole_degrees: Degrees from pole to mask (0-90)
        
    Returns:
        Binary mask (255 = valid, 0 = masked)
    """
    import numpy as np
    
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Calculate pixel rows to mask
    pole_fraction = pole_degrees / 90.0
    pole_pixels = int(height * pole_fraction / 2)
    
    # Mask top and bottom poles
    mask[:pole_pixels, :] = 0
    mask[-pole_pixels:, :] = 0
    
    return mask


def verify_spheresfm_installation(settings=None) -> Dict:
    """
    Verify SphereSfM installation and return status.
    
    Returns:
        Dictionary with installation status and details
    """
    spheresfm_path, searched_paths = resolve_spheresfm_binary_path(settings)
    
    result = {
        'installed': False,
        'path': str(spheresfm_path),
        'searched_paths': searched_paths,
        'version': None,
        'error': None,
        'supports_sphere_camera': False,
    }
    
    if not spheresfm_path.exists():
        result['error'] = f"SphereSfM binary not found at {spheresfm_path}"
        return result
    
    # Try to execute binary/help
    try:
        runnable, supports_sphere = _probe_binary(spheresfm_path)
        result['installed'] = runnable
        result['version'] = "SphereSfM/COLMAP-compatible" if runnable else None
        result['supports_sphere_camera'] = supports_sphere
        if not runnable:
            result['error'] = f"Binary exists but failed to run correctly: {spheresfm_path}"
        
    except subprocess.TimeoutExpired:
        result['error'] = "SphereSfM command timed out"
    except Exception as e:
        result['error'] = str(e)
    
    return result
