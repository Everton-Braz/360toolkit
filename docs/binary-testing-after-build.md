# Binary Testing After Build (Post-Package Validation)

This document explains how to test the built 360toolkit executable automatically.

## Why this exists

After packaging, Python-source tests are not enough. We must validate that the generated `.exe` can:

1. Launch in CLI mode
2. Execute Stage 1 + Stage 2 + Stage 3
3. Produce real output artifacts
4. Return stable pass/fail reports

The binary already supports CLI mode through `--cli` in `src/main.py`.

## Test runner

Use:

- `scripts/test_built_binary.py`

It runs the packaged executable on real project inputs and writes reports to:

- `C:/Users/Everton-PC/Documents/ARQUIVOS_TESTE/automated_tests_binary`

## Inputs used

- X4 INSV:
  - `D:/ARQUIVOS_TESTE_2/Pecem_8K/VID_20260415_112943_00_245.insv`
- A1 INSV:
  - `D:/ARQUIVOS_TESTE_2/VID_20260327_162728_005_Antigravity_A1_Sample.insv`
- MP4:
  - `D:/ARQUIVOS_TESTE_2/PECEM_MP4/VID_20260415_112948.mp4`

## Build first

Example build flow (adjust to your process):

```powershell
py -m PyInstaller 360ToolkitGS.spec --noconfirm
```

## Run post-build tests

Smoke test all 3 inputs:

```powershell
py scripts/test_built_binary.py --mode smoke
```

Full test on X4:

```powershell
py scripts/test_built_binary.py --mode full --full-target x4
```

Smoke + full:

```powershell
py scripts/test_built_binary.py --mode both --full-target x4
```

If auto-discovery cannot find the executable, pass it explicitly:

```powershell
py scripts/test_built_binary.py --binary "C:/path/to/360ToolkitGS-Simple.exe" --mode smoke
```

## Exit codes

- `0`: all profiles passed
- `1`: at least one hard fail
- `2`: partial pass (for example timeout)

## Artifacts per run

Each run creates:

- `run_summary.json`
- `run_summary.txt`
- `<profile>/stdout.log`
- `<profile>/stderr.log`
- `<profile>/` output folders (`extracted_frames`, `perspective_views`, masks)

## Pass/fail rules (current)

A profile is considered `pass` when:

1. Binary exit code is zero
2. Stage 1 generated images (`extracted_frames` > 0)
3. Stage 2 generated images (`perspective_views` > 0)
4. Stage 3 execution marker exists in stdout (`Stage 3 complete: success=True` or `CLI_PIPELINE_PASSED`)

Note: Stage 3 may produce zero mask files if no matching objects are detected.

## Recommended release gate

Run these commands before creating final zip/release:

```powershell
py scripts/test_built_binary.py --mode smoke
py scripts/test_built_binary.py --mode full --full-target x4
```

Only publish if both return `0`.
