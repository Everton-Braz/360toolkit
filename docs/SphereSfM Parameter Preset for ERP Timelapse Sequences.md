# SphereSfM Parameter Preset for ERP Timelapse Sequences

## Overview

This report proposes a practical baseline of SphereSfM and COLMAP parameters for equirectangular (ERP) 360 images, tuned for dense timelapse sequences (200–1000 frames at up to 8K) where the primary goal is reliable alignment (few broken tracks/outliers) rather than maximum speed.

The preset is grounded in: (1) the official SphereSfM README and example commands, (2) default COLMAP SIFT extraction and matching settings, and (3) typical best practices for ERP SfM from the underlying paper and COLMAP discussions.

## Assumptions and Goals

- Input images are 360×180 ERP (2:1 aspect ratio) from consumer spherical cameras (e.g., Insta360, Ricoh Theta).[^1]
- Sequences are dense timelapses extracted from video (frame every N frames) with strong overlap and relatively small baselines between neighbors.
- Image count per sequence is roughly 200–1000, with resolutions up to 7680×3840 (8K).
- The priority is robust, clean alignment (good tracks, minimal outliers or topology breaks) over absolute runtime.
- Workflow uses SphereSfM inside COLMAP (SPHERE camera model, spherical BA, optional camera mask, and optional POS data) as documented in the README.

## Key References and Defaults

The SphereSfM README illustrates a minimal working command-line pipeline:

- Feature extraction with SPHERE camera model, fixed intrinsics and optional camera mask and POS.
- Matching with spatial_matcher using `--SiftMatching.max_error 4` and `--SiftMatching.min_num_inliers 50` when approximate positions are available.
- Mapping with `--Mapper.ba_refine_focal_length 0`, `--Mapper.ba_refine_principal_point 0`, `--Mapper.ba_refine_extra_params 0`, and `--Mapper.sphere_camera 1` to keep intrinsics fixed and enable spherical BA.

COLMAP’s documented defaults for SIFT extraction and matching are, for example:

- `SiftExtraction.peak_threshold = 0.006666...`, `SiftExtraction.edge_threshold = 10`, `SiftExtraction.max_num_orientations = 2`.
- `SiftExtraction.max_image_size = 3200`, `SiftExtraction.max_num_features = 8192`.
- `SiftMatching.max_ratio = 0.8`, `SiftMatching.max_distance = 0.7`, `SiftMatching.max_error = 4`.

The proposed preset keeps these core defaults (which are also what SphereSfM assumes) and tightens or raises a few thresholds that matter for dense ERP timelapse.

## Recommended Feature Extraction Settings

### Core SphereSfM / COLMAP Flags

These flags should be considered the baseline for ERP feature extraction:

- `--ImageReader.camera_model SPHERE`
- `--ImageReader.single_camera 1`
- `--ImageReader.camera_params "1,<width>,<height>"` (unit-sphere intrinsics as in the README)
- Optional: `--ImageReader.camera_mask_path ./camera_mask.png` to mask poles/rig/tripod.

The README explicitly demonstrates SPHERE with fixed intrinsics and the use of a camera mask for ERP images; this is essential for reliable matching in the presence of heavy distortion and low-texture sky at the poles.

### SIFT Extraction Preset

| Parameter | Suggested value | Rationale |
|----------|-----------------|-----------|
| `--SiftExtraction.max_image_size` | `3200`–`4096` | Keep COLMAP default of 3200 for speed, or increase modestly for 8K ERPs to preserve detail without exploding cost. |
| `--SiftExtraction.max_num_features` | `12000` | Higher than the default 8192 to better cover the wide FOV of 8K ERP while staying manageable for 200–1000 images. |
| `--SiftExtraction.peak_threshold` | `0.00667` | Default COLMAP value; already tuned for good keypoint density on natural scenes. |
| `--SiftExtraction.edge_threshold` | `10` | Default; works well across scenes and is what COLMAP and SphereSfM examples use. |
| `--SiftExtraction.max_num_orientations` | `2` | Default; retains rotation invariance that matters on ERP, especially away from the equator. |

Other SIFT-related parameters (octaves, octave resolution, domain-size pooling) can remain at COLMAP defaults unless there is a very specific need to tweak them; SphereSfM does not recommend custom values in its documentation.

## Recommended Matching Settings (Sequential Timelapse)

For 200–1000-frame ERP timelapse sequences, sequential matching is typically sufficient and more stable than exhaustive matching, given the strong temporal overlap between neighbors.

### Matcher Choice

- Prefer `colmap sequential_matcher` (or SphereSfM’s equivalent CLI) for video-derived timelapse sequences with consistent capture order.
- Use `spatial_matcher` only when reliable approximate positions (GNSS/INS) are available; the SphereSfM README demonstrates this mode with explicit POS input and spatial thresholds.

### Sequential Matching Preset

| Parameter | Suggested value | Rationale |
|----------|-----------------|-----------|
| `--SequentialMatching.quadratic_overlap` | `1` | Keep enabled (as in your current config) so farther neighbors are still considered in larger sequences. |
| `--SequentialMatching.overlap` | `4`–`5` | For dense timelapse, matching ±4–5 frames provides redundancy while avoiding quadratic blow-up. |

### SIFT Matching Preset

| Parameter | Suggested value | Rationale |
|----------|-----------------|-----------|
| `--SiftMatching.max_ratio` | `0.8` | COLMAP default; standard Lowe ratio that balances recall and robustness. |
| `--SiftMatching.max_distance` | `0.7` | COLMAP default; avoids extremely distant descriptor matches. |
| `--SiftMatching.cross_check` | `1` | Ensures mutual consistency and significantly reduces outliers, important for ERP distortions. |
| `--SiftMatching.max_error` | `4.0` | Matches SphereSfM README and COLMAP defaults; RANSAC inlier threshold in pixels. |
| `--SiftMatching.confidence` | `0.999` | High RANSAC confidence; fine for small-baseline timelapse where inliers are abundant. |
| `--SiftMatching.max_num_trials` | `10000` | Default upper bound; keep to allow RANSAC to find good models in cluttered scenes. |
| `--SiftMatching.min_inlier_ratio` | `0.25` | Reasonable minimum inlier ratio for ERP; can be nudged to `0.3` if many degenerate matches remain. |
| `--SiftMatching.min_num_inliers` (runtime) | `50` | Aligns with SphereSfM’s spatial_matcher example; rejects weak or spurious links between ERP frames. |
| `--SiftMatching.max_num_matches` (runtime) | `32000` | Raises cap above default to avoid truncating good matches on 8K ERP, but not excessively large. |

The most impactful changes for robustness on ERP timelapse are:

- Ensuring `min_num_inliers` is not too low (≈50) so only geometrically solid edges are kept in the view-graph.
- Keeping default Lowe ratio and distance thresholds, which are already conservative in COLMAP.
- Using cross-check and sequential overlap to exploit temporal redundancy.

## Recommended Mapping Settings (SphereSfM)

SphereSfM’s README emphasizes keeping intrinsics fixed and enabling the spherical camera model in the mapper.

### Core Mapper Flags

- `--Mapper.sphere_camera 1`
- `--Mapper.ba_refine_focal_length 0`
- `--Mapper.ba_refine_principal_point 0`
- `--Mapper.ba_refine_extra_params 0`

This matches the example in the README and ensures bundle adjustment operates on poses and 3D points in the correct spherical geometry while avoiding overfitting intrinsics from ERP images alone.

### Robustness-Oriented Mapper Preset

| Parameter | Suggested value | Rationale |
|----------|-----------------|-----------|
| `--Mapper.init_min_num_inliers` | `100` | High-quality initial pair; your current choice is consistent with robust initialization. |
| `--Mapper.init_num_trials` | `200` | Sufficient trials for finding a good seed pair in dense timelapse. |
| `--Mapper.init_max_error` | `4` | Symmetric with matching threshold; appropriate for ERP with spherical BA. |
| `--Mapper.init_max_forward_motion` | `0.95` | Helps avoid pure forward-motion degeneracy; leave as in your config. |
| `--Mapper.init_min_tri_angle` | `16` | Strong baseline for seed pair; good for stability even with many frames. |
| `--Mapper.abs_pose_min_num_inliers` | `50` | More conservative than the default 30; improves robustness when registering new images in dense timelapse. |
| `--Mapper.abs_pose_max_error` | `8` | Slightly stricter than 12px, but looser than the 4px seed threshold; trades off robustness with tolerance for ERP noise. |
| `--Mapper.abs_pose_min_inlier_ratio` | `0.25` | Reasonable lower bound on inlier fraction for accepting new views. |
| `--Mapper.max_reg_trials` | `3` | Default-like; balances trying again with different hypotheses vs. runtime. |
| `--Mapper.tri_min_angle` | `1.5` | Low enough for small baselines; can be raised to `2.0` if many poorly conditioned triangles persist. |
| `--Mapper.tri_max_transitivity` | `1` | Keeps triangulation conservative in terms of track graph complexity. |
| `--Mapper.tri_ignore_two_view_tracks` | `1` | Avoids fragile two-view-only points; improves overall track quality. |
| `--Mapper.filter_max_reproj_error` | `4` | Consistent with matching threshold; filters poorly supported 3D points. |
| `--Mapper.filter_min_tri_angle` | `1.5` | As above; may be nudged to `2.0` for even stricter filtering. |
| `--Mapper.multiple_models` | `1` | Allows multiple disjoint components; useful if the timelapse contains hard cuts or resets.
| `--Mapper.min_num_matches` (runtime) | `50` | Ensures only strong image pairs contribute to registering new views in incremental SfM. |

These mapper choices focus on tightening the requirements for accepting new views and 3D points (via higher inlier counts and consistent reprojection thresholds) while leaving the core spherical modeling behaviour identical to SphereSfM’s examples.

## Putting It Together: Example CLI

A concrete end-to-end CLI skeleton for a typical ERP timelapse (without POS data) can look like this (values in angle brackets should be adapted):

```bash
# Step 1: database
colmap database_creator \
  --database_path ./colmap/database.db

# Step 2: feature extraction
colmap feature_extractor \
  --database_path ./colmap/database.db \
  --image_path ./images \
  --ImageReader.camera_model SPHERE \
  --ImageReader.camera_params "1,<width>,<height>" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_mask_path ./camera_mask.png \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.max_image_size 4096 \
  --SiftExtraction.max_num_features 12000 \
  --SiftExtraction.peak_threshold 0.00667 \
  --SiftExtraction.edge_threshold 10 \
  --SiftExtraction.max_num_orientations 2

# Step 3: sequential matching
colmap sequential_matcher \
  --database_path ./colmap/database.db \
  --SequentialMatching.overlap 4 \
  --SequentialMatching.quadratic_overlap 1 \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 \
  --SiftMatching.max_error 4 \
  --SiftMatching.confidence 0.999 \
  --SiftMatching.max_num_trials 10000 \
  --SiftMatching.min_inlier_ratio 0.25 \
  --SiftMatching.min_num_inliers 50 \
  --SiftMatching.max_num_matches 32000

# Step 4: mapping (SphereSfM build of COLMAP)
colmap mapper \
  --database_path ./colmap/database.db \
  --image_path ./images \
  --output_path ./colmap/sparse \
  --Mapper.sphere_camera 1 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  --Mapper.min_num_matches 50 \
  --Mapper.init_min_num_inliers 100 \
  --Mapper.init_num_trials 200 \
  --Mapper.init_max_error 4 \
  --Mapper.init_max_forward_motion 0.95 \
  --Mapper.init_min_tri_angle 16 \
  --Mapper.abs_pose_min_num_inliers 50 \
  --Mapper.abs_pose_max_error 8 \
  --Mapper.abs_pose_min_inlier_ratio 0.25 \
  --Mapper.max_reg_trials 3 \
  --Mapper.tri_min_angle 1.5 \
  --Mapper.tri_max_transitivity 1 \
  --Mapper.tri_ignore_two_view_tracks 1 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.filter_min_tri_angle 1.5 \
  --Mapper.multiple_models 1
```

This skeleton keeps the spirit of the SphereSfM README while explicitly encoding the robustness-oriented thresholds discussed above.

## When to Deviate from This Preset

- **Very small datasets (<100 ERPs):** Lower `min_num_inliers` and `min_num_matches` slightly (e.g., to 30–40) to avoid over-pruning the view-graph.
- **Extremely large datasets (>2000 ERPs):** Reduce `SequentialMatching.overlap` and possibly `max_num_features` to keep runtimes reasonable.
- **Indoor or low-texture scenes:** Consider slightly lowering `SiftExtraction.peak_threshold` (e.g., to 0.004) to increase keypoint density if too few matches are found.
- **Datasets with reliable POS / GNSS:** Switch to `spatial_matcher` with the POS-aware settings from the SphereSfM README, keeping `SiftMatching.max_error = 4` and `min_num_inliers = 50` as demonstrated.

Overall, the SphereSfM authors lean heavily on COLMAP’s defaults and a small set of carefully chosen thresholds (notably, fixed intrinsics, spherical BA, and a 4-pixel inlier threshold), so small, principled adjustments around these defaults are usually sufficient for robust ERP timelapse reconstruction.

---

## References

1. [Search for others methods to do the alignment of the erp images, papers, novels aproachs.. and list all known options.](https://www.perplexity.ai/search/e3c69b95-e257-45be-bbf0-e5fe80b0bb7f) - Here’s a structured list of known options to align ERP (equirectangular) 360 images. For details and...

