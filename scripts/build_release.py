from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.defaults import APP_NAME, APP_VERSION  # noqa: E402


VARIANTS = {
    "full-bundled": {
        "build_name": "360ToolkitGS",
        "bundle_external_tools": "1",
        "display_name": "Full Bundled GPU",
    },
    "customer-managed": {
        "build_name": "360ToolkitGS-Managed",
        "bundle_external_tools": "0",
        "display_name": "Customer Managed GPU",
    },
}


def _zip_directory(source_dir: Path, target_zip: Path) -> None:
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    if target_zip.exists():
        target_zip.unlink()

    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def _variant_config(variant: str) -> dict[str, str]:
    try:
        return VARIANTS[variant]
    except KeyError as exc:
        raise SystemExit(f"Unsupported variant: {variant}") from exc


def _dist_dir(build_name: str) -> Path:
    return REPO_ROOT / "dist" / build_name


def _zip_path(variant: str, build_name: str) -> Path:
    artifact = f"{build_name}-v{APP_VERSION}-windows-x64-{variant}.zip"
    return REPO_ROOT / "releases" / artifact


def _ensure_windows_icon_asset() -> Path | None:
    jpg_icon = REPO_ROOT / "resources" / "images" / "logo-favicon.jpg"
    ico_icon = jpg_icon.with_suffix(".ico")
    if not jpg_icon.exists():
        return None

    rebuild_icon = not ico_icon.exists() or jpg_icon.stat().st_mtime > ico_icon.stat().st_mtime
    if rebuild_icon:
        image = Image.open(jpg_icon).convert("RGBA")
        image.save(
            ico_icon,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        print(f"[OK] Generated Windows icon: {ico_icon}")

    return ico_icon if ico_icon.exists() else None


def _conda_env_prefix(conda_exe: str, env_name: str) -> Path:
    command = [
        conda_exe,
        "run",
        "-n",
        env_name,
        "python",
        "-c",
        "import sys; print(sys.prefix)",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Failed to resolve conda environment prefix for {env_name}: {result.stderr.strip() or result.stdout.strip()}"
        )

    prefix = Path((result.stdout or "").strip().splitlines()[-1])
    if not prefix.exists():
        raise SystemExit(f"Resolved conda environment prefix does not exist: {prefix}")
    return prefix


def _sync_runtime_dlls(dist_dir: Path, conda_prefix: Path) -> None:
    internal_dir = dist_dir / "_internal"
    if not internal_dir.exists():
        raise SystemExit(f"Build output missing internal runtime directory: {internal_dir}")

    runtime_bin = conda_prefix / "Library" / "bin"
    required_dlls = [
        "libcrypto-3-x64.dll",
        "liblzma.dll",
        "libssl-3-x64.dll",
        "sqlite3.dll",
        "vcomp140.dll",
    ]
    runtime_patterns = [
        "api-ms-win-core-*.dll",
        "api-ms-win-crt-*.dll",
        "concrt140.dll",
        "msvcp140*.dll",
        "ucrtbase.dll",
        "vcruntime140*.dll",
        "zlib.dll",
    ]

    copied: set[str] = set()

    for dll_name in required_dlls:
        source = runtime_bin / dll_name
        if not source.exists():
            raise SystemExit(f"Required runtime DLL not found in conda environment: {source}")
        destination = internal_dir / dll_name
        shutil.copy2(source, destination)
        copied.add(dll_name.lower())
        print(f"[OK] Synced runtime DLL: {destination}")

    for source in sorted(conda_prefix.glob("*.dll")):
        if source.name.lower() in copied:
            continue
        if not any(fnmatch.fnmatch(source.name.lower(), pattern.lower()) for pattern in runtime_patterns):
            continue
        destination = internal_dir / source.name
        shutil.copy2(source, destination)
        copied.add(source.name.lower())
        print(f"[OK] Synced Python runtime DLL: {destination}")


def _sync_torch_runtime(dist_dir: Path, conda_prefix: Path) -> None:
    internal_dir = dist_dir / "_internal"
    torch_source_dir = conda_prefix / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_source_dir.exists():
        raise SystemExit(f"Torch runtime directory not found in build environment: {torch_source_dir}")

    target_dirs = [internal_dir, internal_dir / "torch" / "lib"]
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)

    synced_count = 0
    for source in sorted(torch_source_dir.glob("*.dll")):
        for target_dir in target_dirs:
            destination = target_dir / source.name
            shutil.copy2(source, destination)
        synced_count += 1

    print(
        f"[OK] Synced Torch runtime DLLs: {synced_count} files to "
        f"{internal_dir} and {internal_dir / 'torch' / 'lib'}"
    )


def _sync_directory_tree(source_dir: Path, destination_dir: Path, label: str) -> None:
    if not source_dir.exists():
        raise SystemExit(f"{label} source directory not found: {source_dir}")

    copied_files = 0
    for source in sorted(source_dir.rglob("*")):
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files += 1

    print(f"[OK] Synced {label}: {copied_files} files -> {destination_dir}")


def _sync_external_tool_runtimes(dist_dir: Path) -> None:
    internal_dir = dist_dir / "_internal"

    colmap_source_dir = REPO_ROOT / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    colmap_destination_dir = internal_dir / "bin" / "COLMAP-windows-latest-CUDA-cuDSS-GUI"
    if colmap_source_dir.exists():
        _sync_directory_tree(colmap_source_dir, colmap_destination_dir, "COLMAP runtime tree")


def run_pyinstaller(variant: str, env_name: str, clean: bool) -> None:
    config = _variant_config(variant)
    build_name = config["build_name"]

    if clean:
        shutil.rmtree(REPO_ROOT / "build" / build_name, ignore_errors=True)
        shutil.rmtree(_dist_dir(build_name), ignore_errors=True)

    conda = shutil.which("conda")
    if not conda:
        raise SystemExit("conda executable not found in PATH")
    conda_prefix = _conda_env_prefix(conda, env_name)
    icon_path = _ensure_windows_icon_asset()

    env = os.environ.copy()
    env.update(
        {
            "TOOLKIT_RELEASE_VARIANT": variant,
            "TOOLKIT_BUILD_NAME": build_name,
            "TOOLKIT_BUILD_VERSION": APP_VERSION,
            "TOOLKIT_BUNDLE_EXTERNAL_TOOLS": config["bundle_external_tools"],
            "TOOLKIT_WINDOWS_ICON": str(icon_path) if icon_path else "",
        }
    )

    command = [conda, "run", "-n", env_name, "python", "-m", "PyInstaller", "360ToolkitGS.spec", "--noconfirm"]
    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    exe_path = _dist_dir(build_name) / f"{build_name}.exe"
    if not exe_path.exists():
        raise SystemExit(f"Build finished without expected executable: {exe_path}")

    _sync_runtime_dlls(_dist_dir(build_name), conda_prefix)
    _sync_torch_runtime(_dist_dir(build_name), conda_prefix)
    _sync_external_tool_runtimes(_dist_dir(build_name))

    print(f"[OK] Built {config['display_name']}: {exe_path}")


def create_zip(variant: str) -> None:
    config = _variant_config(variant)
    build_name = config["build_name"]
    dist_dir = _dist_dir(build_name)
    if not dist_dir.exists():
        raise SystemExit(f"Build output not found: {dist_dir}")

    zip_path = _zip_path(variant, build_name)
    _zip_directory(dist_dir, zip_path)
    print(f"[OK] Created ZIP: {zip_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 360toolkit release variants")
    parser.add_argument("action", choices=["build", "zip"], nargs="?", default="build")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full-bundled")
    parser.add_argument("--env", default="360pipeline", help="Conda environment used for PyInstaller")
    parser.add_argument("--no-clean", action="store_true", help="Skip removing previous build artifacts")
    parser.add_argument("--zip", action="store_true", dest="zip_after_build", help="Create ZIP after a successful build")
    args = parser.parse_args()

    if args.action == "build":
        run_pyinstaller(args.variant, args.env, clean=not args.no_clean)
        if args.zip_after_build:
            create_zip(args.variant)
        return 0

    create_zip(args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())