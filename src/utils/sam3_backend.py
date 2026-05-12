from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SAM3_BACKEND_AUTO = 'auto'
SAM3_BACKEND_CPU = 'cpu'
SAM3_BACKEND_CUDA = 'cuda'
SAM3_BACKEND_VULKAN = 'vulkan'
SAM3_BACKEND_UNVERIFIED_VULKAN = 'unverified-vulkan'
SAM3_GPU_BACKENDS = {SAM3_BACKEND_CUDA, SAM3_BACKEND_VULKAN}
SAM3_VALID_BACKEND_MODES = {SAM3_BACKEND_AUTO, SAM3_BACKEND_CPU, SAM3_BACKEND_CUDA, SAM3_BACKEND_VULKAN}


@dataclass(frozen=True)
class SAM3BackendInfo:
    backend: str
    detail: str
    build_dir: Path | None = None


def normalize_sam3_backend_mode(mode: str | None, use_gpu: bool = True) -> str:
    value = str(mode or '').strip().lower()
    if value in SAM3_VALID_BACKEND_MODES:
        return value
    return SAM3_BACKEND_AUTO if use_gpu else SAM3_BACKEND_CPU


def sam3_backend_supports_gpu(backend: str) -> bool:
    return backend in SAM3_GPU_BACKENDS


def sam3_backend_matches_mode(backend: str, mode: str, *, allow_unverified_vulkan: bool = False) -> bool:
    normalized_mode = normalize_sam3_backend_mode(mode)
    if normalized_mode == SAM3_BACKEND_AUTO:
        if allow_unverified_vulkan:
            return backend in {SAM3_BACKEND_CPU, SAM3_BACKEND_CUDA, SAM3_BACKEND_VULKAN, SAM3_BACKEND_UNVERIFIED_VULKAN}
        return backend in {SAM3_BACKEND_CPU, SAM3_BACKEND_CUDA, SAM3_BACKEND_VULKAN}
    if normalized_mode == SAM3_BACKEND_VULKAN:
        if backend == SAM3_BACKEND_VULKAN:
            return True
        return allow_unverified_vulkan and backend == SAM3_BACKEND_UNVERIFIED_VULKAN
    return backend == normalized_mode


_BUILD_CONFIG_DIRS = {'release', 'debug', 'relwithdebinfo', 'minsizerel'}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return ''


def _build_dir_from_executable(executable_path: Path) -> Path | None:
    exe_path = Path(executable_path)
    parent = exe_path.parent

    if parent.name.lower() in _BUILD_CONFIG_DIRS and parent.parent.name.lower() == 'examples':
        return parent.parent.parent
    if parent.name.lower() == 'examples':
        return parent.parent
    return None


def _runtime_search_dirs(executable_path: Path, build_dir: Path | None) -> list[Path]:
    exe_path = Path(executable_path)
    exe_dir = exe_path.parent
    candidates: list[Path] = [exe_dir]

    if build_dir is not None:
        candidates.extend([
            build_dir / 'bin',
            build_dir / 'bin' / 'Release',
            build_dir / 'bin' / 'Debug',
        ])

    if exe_dir.name.lower() in _BUILD_CONFIG_DIRS and exe_dir.parent.name.lower() == 'examples':
        root = exe_dir.parent.parent
        candidates.extend([
            root / 'bin',
            root / 'bin' / exe_dir.name,
        ])
    elif exe_dir.name.lower() == 'examples':
        root = exe_dir.parent
        candidates.extend([
            root / 'bin',
            root / 'bin' / 'Release',
            root / 'bin' / 'Debug',
        ])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _runtime_has_file(search_dirs: list[Path], filename: str) -> bool:
    target = filename.lower()
    for directory in search_dirs:
        try:
            if not directory.exists():
                continue
            for child in directory.iterdir():
                if child.is_file() and child.name.lower() == target:
                    return True
        except OSError:
            continue
    return False


def _cache_value(cache_text: str, key: str) -> str:
    match = re.search(rf'^{re.escape(key)}:[^=]*=(.*)$', cache_text, re.MULTILINE)
    return match.group(1).strip() if match else ''


def _cache_bool(cache_text: str, key: str) -> bool:
    return _cache_value(cache_text, key).upper() == 'ON'


def _target_link_snippet(build_ninja_text: str, executable_name: str) -> str:
    patterns = [
        f'build examples\\{executable_name}:',
        f'build examples/{executable_name}:',
    ]
    for marker in patterns:
        index = build_ninja_text.find(marker)
        if index >= 0:
            return build_ninja_text[index:index + 1600]
    return ''


def inspect_sam3_executable_backend(executable_path: str | Path) -> SAM3BackendInfo:
    exe_path = Path(executable_path)
    if not exe_path.is_file():
        return SAM3BackendInfo('missing', f'Executable not found: {exe_path}')

    build_dir = _build_dir_from_executable(exe_path)
    if build_dir is None:
        runtime_dirs = _runtime_search_dirs(exe_path, None)
        if _runtime_has_file(runtime_dirs, 'ggml-vulkan.dll'):
            return SAM3BackendInfo('vulkan', f'Runtime Vulkan backend inferred for {exe_path.name}', None)
        if _runtime_has_file(runtime_dirs, 'ggml-cuda.dll'):
            return SAM3BackendInfo('cuda', f'Runtime CUDA backend inferred for {exe_path.name}', None)
        if _runtime_has_file(runtime_dirs, 'ggml-cpu.dll'):
            return SAM3BackendInfo('cpu', f'Runtime CPU backend inferred for {exe_path.name}', None)
        return SAM3BackendInfo('unknown', f'Could not infer build directory for {exe_path}', None)

    cache_text = _read_text(build_dir / 'CMakeCache.txt')
    ninja_text = _read_text(build_dir / 'build.ninja')
    link_snippet = _target_link_snippet(ninja_text, exe_path.name)
    runtime_dirs = _runtime_search_dirs(exe_path, build_dir)

    ggml_vulkan = _cache_bool(cache_text, 'GGML_VULKAN')
    sam3_vulkan = _cache_bool(cache_text, 'SAM3_VULKAN')
    ggml_cuda = _cache_bool(cache_text, 'GGML_CUDA')
    sam3_cuda = _cache_bool(cache_text, 'SAM3_CUDA')
    vulkan_include = _cache_value(cache_text, 'Vulkan_INCLUDE_DIR')
    vulkan_library = _cache_value(cache_text, 'Vulkan_LIBRARY')
    sdk_resolved = bool(vulkan_include and not vulkan_include.endswith('NOTFOUND')) and bool(
        vulkan_library and not vulkan_library.endswith('NOTFOUND')
    )

    link_has_vulkan = any(token in link_snippet.lower() for token in ('ggml-vulkan', 'vulkan-1.lib', 'vulkan.lib'))
    link_has_cuda = any(
        token in link_snippet.lower()
        for token in ('ggml-cuda', 'cudart.lib', 'cuda.lib', 'cublas.lib', 'cublaslt.lib')
    )
    link_has_cpu = 'ggml-cpu.lib' in link_snippet.lower()
    runtime_has_vulkan = _runtime_has_file(runtime_dirs, 'ggml-vulkan.dll')
    runtime_has_cuda = _runtime_has_file(runtime_dirs, 'ggml-cuda.dll')
    runtime_has_cpu = _runtime_has_file(runtime_dirs, 'ggml-cpu.dll')

    if (ggml_vulkan or sam3_vulkan) and sdk_resolved and link_has_vulkan:
        return SAM3BackendInfo('vulkan', f'Verified Vulkan backend for {exe_path.name}', build_dir)

    if (ggml_cuda or sam3_cuda) and link_has_cuda:
        return SAM3BackendInfo('cuda', f'CUDA backend for {exe_path.name}', build_dir)

    if runtime_has_vulkan and not runtime_has_cuda:
        return SAM3BackendInfo('vulkan', f'Runtime Vulkan backend inferred for {exe_path.name}', build_dir)

    if runtime_has_cuda and not runtime_has_vulkan:
        return SAM3BackendInfo('cuda', f'Runtime CUDA backend inferred for {exe_path.name}', build_dir)

    if not (ggml_vulkan or sam3_vulkan) and (link_has_cpu or runtime_has_cpu) and not (link_has_vulkan or runtime_has_vulkan):
        return SAM3BackendInfo('cpu', f'CPU-only backend for {exe_path.name}', build_dir)

    if ggml_vulkan or sam3_vulkan or 'build-vulkan' in str(build_dir).lower():
        detail = f'Unverified Vulkan build for {exe_path.name}'
        if not sdk_resolved:
            detail += ' (Vulkan SDK paths unresolved)'
        elif not link_has_vulkan:
            detail += ' (no Vulkan link evidence in build.ninja)'
        return SAM3BackendInfo('unverified-vulkan', detail, build_dir)

    return SAM3BackendInfo('unknown', f'Unknown backend for {exe_path.name}', build_dir)


def is_verified_vulkan_sam3_executable(executable_path: str | Path) -> bool:
    return inspect_sam3_executable_backend(executable_path).backend == 'vulkan'