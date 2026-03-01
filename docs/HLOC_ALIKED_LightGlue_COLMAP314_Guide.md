# HLOC + ALIKED + LightGlue with COLMAP 3.14 (CUDA/cuDSS)

Version: 2026-02-17  
Target: 360Toolkit (Windows, RTX 5070 Ti)

## 1) What is new in your COLMAP build

Verified from your installed binary and release notes:

- Build: **COLMAP 3.14.0.dev0** (commit `a6f539d`, CUDA enabled).
- `global_mapper` is built-in (GLOMAP workflow is now inside COLMAP CLI).
- ONNX learned pipeline is available:
  - `--FeatureExtraction.type ALIKED_N16ROT` / `ALIKED_N32`
  - `--FeatureMatching.type ALIKED_LIGHTGLUE` / `SIFT_LIGHTGLUE`
- pycolmap wheel family includes CPU / CUDA / CUDA+cuDSS variants.
- Driver requirement for CUDA 12.8 packages: NVIDIA 570+ (your machine is already above this).

Release reference:
- https://github.com/lyehe/build_gpu_colmap/releases/tag/v3.14.0-dev1

## 2) Practical insight: best stack by stage

For robust large-scene reconstruction (your use-case):

1. **Pair generation/retrieval**: HLOC (NetVLAD or sequential-by-time).
2. **Features + matching**: ALIKED + LightGlue (COLMAP ONNX path).
3. **SfM**: `colmap global_mapper` first; fallback to incremental `mapper` for hard/fragmented scenes.
4. **Dense/MVS**:
   - Start with COLMAP PatchMatch for speed and integration.
   - Use OpenMVS when you need denser/cleaner mesh output.

## 3) Recommended settings (starting presets)

## 3.1 Feature extraction (learned)

Use learned features first:

```bash
colmap feature_extractor \
  --database_path database.db \
  --image_path images \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.gpu_index 0 \
  --FeatureExtraction.type ALIKED_N16ROT \
  --AlikedExtraction.n16rot_model_path aliked-n16rot.onnx \
  --ImageReader.single_camera_per_folder 1
```

Why `single_camera_per_folder` first:
- Good compromise for mixed rigs/splits while avoiding one giant shared intrinsics model.
- Use `single_camera_per_image=1` only for severe zoom/intrinsics drift.

## 3.2 Matching (learned)

```bash
colmap sequential_matcher \
  --database_path database.db \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index 0 \
  --FeatureMatching.type ALIKED_LIGHTGLUE \
  --AlikedMatching.lightglue_model_path aliked-lightglue.onnx \
  --SequentialMatching.overlap 12 \
  --SequentialMatching.loop_detection 1 \
  --FeatureMatching.max_num_matches 32768
```

For unstable runs, retry with:
- `--SequentialMatching.loop_detection 0`
- `--SequentialMatching.overlap 8`

## 3.3 Global SfM (new COLMAP path)

```bash
colmap global_mapper \
  --database_path database.db \
  --image_path images \
  --output_path sparse \
  --GlobalMapper.gp_use_gpu 1 \
  --GlobalMapper.ba_ceres_use_gpu 1 \
  --GlobalMapper.min_num_matches 20 \
  --GlobalMapper.ba_num_iterations 3 \
  --GlobalMapper.ba_ceres_max_num_iterations 200
```

Notes:
- Increase `min_num_matches` (15 -> 20/25) to reduce weak edges in noisy datasets.
- Keep BA intrinsics refinement enabled unless camera calibration is truly fixed and trusted.

## 3.4 Incremental fallback (if global graph is weak)

```bash
colmap mapper \
  --database_path database.db \
  --image_path images \
  --output_path sparse_inc \
  --Mapper.min_num_matches 20 \
  --Mapper.abs_pose_min_num_inliers 40 \
  --Mapper.filter_max_reproj_error 3 \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1
```

## 4) 360 / ERP alignment guidance

You currently have two valid 360 routes:

1. **SphereSfM-style ERP route**: best when input is pure ERP panoramas.
2. **Perspective split route** (your Rig pipeline): best for compatibility with learned feature pipelines and mixed datasets.

Rule of thumb:
- Pure ERP set from one camera family: prefer shared or per-folder camera model.
- Mixed camera/lens/zoom: per-folder first, per-image only if reprojection/intrinsics instability persists.

## 5) HLOC + COLMAP integration pattern

Use HLOC mainly for scalable pairing and/or feature exports, then map with new COLMAP:

1. HLOC retrieval/pairs (`pairs-netvlad.txt` or sequential pairs).
2. COLMAP learned extraction + `match_from_pairs` (pycolmap) or matcher with filtered subsets.
3. `global_mapper` first, then incremental fallback only if needed.

## 6) Known pitfalls and fixes

- ONNX model files are not always bundled in the COLMAP zip -> ensure local paths exist.
- pycolmap ABI mismatch can crash on Windows -> keep wheel version aligned with this COLMAP build.
- If learned matcher crashes, fallback sequence should be:
  1) `ALIKED_N32 + ALIKED_LIGHTGLUE`
  2) `ALIKED_N16ROT + ALIKED_LIGHTGLUE`
  3) `SIFT_LIGHTGLUE`
  4) plain SIFT matcher

## 7) References for AI-agent fetch/analyze

- COLMAP docs: https://colmap.github.io/
- COLMAP repo: https://github.com/colmap/colmap
- GPU build release (your package family): https://github.com/lyehe/build_gpu_colmap/releases/tag/v3.14.0-dev1
- HLOC: https://github.com/cvg/Hierarchical-Localization
- ALIKED paper: https://arxiv.org/abs/2304.03608
- LightGlue paper: https://arxiv.org/abs/2306.13643

## 8) 360Toolkit-specific note

In this codebase, `mapping_backend='glomap'` already routes to COLMAP `global_mapper` behavior in the integration layer. Treat that option as "global COLMAP" on 3.14+ builds.
