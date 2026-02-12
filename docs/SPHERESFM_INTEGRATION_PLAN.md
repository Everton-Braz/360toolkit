# SphereSfM Integration - Implementation Complete ✅

## Summary

360ToolKit now supports **two alignment modes** for Stage 4:

| Mode | Name | Description | Status | Test Results |
|------|------|-------------|--------|--------------|
| **Mode A** | SphereSfM (Native) | Direct spherical feature matching using SphereSfM binary | ✅ Implemented | **30/30 (100%), 5,553 pts, 75s** |
| **Mode B** | Rig-based SfM | Virtual perspectives with COLMAP rig constraints | ✅ Implemented (Default) | 270/270 (100%), 18,194 pts |

## Quick Start

1. **Open 360ToolKit** → Go to **Stage 4: Alignment** tab
2. **Select Mode**:
   - **Mode B (Recommended)**: Rig-based SfM - proven 100% registration rate
   - **Mode A**: SphereSfM - native spherical matching (for urban/street scenes)
3. SphereSfM status indicator shows if Mode A is available

## Test Results (2026-02-01)

### Mode A (SphereSfM Native) - 30 Equirectangular Frames
```
✅ Images registered: 30/30 (100%)
✅ 3D points: 5,553
✅ Processing time: 75 seconds
✅ Cubemap output: 180 perspective views (30 panos × 6 faces)
```

### Key Settings (from successful GUI run)
```ini
[ImageReader]
camera_model=SPHERE
camera_params=              # Empty - auto-detect (CRITICAL!)
single_camera=true

[SiftExtraction]
use_gpu=false
max_image_size=3200
max_num_features=8192
peak_threshold=0.00667

[SequentialMatching]
overlap=10
quadratic_overlap=true
loop_detection=false

[Mapper]
sphere_camera=true
ba_refine_focal_length=false
init_min_num_inliers=100
tri_ignore_two_view_tracks=true
```

## Files Modified/Created

### New Files
- `src/premium/sphere_sfm_integration.py` - SphereSfM binary integration
- `bin/SphereSfM/` - SphereSfM binary (colmap.exe + cudart64_12.dll + DLLs)

### Modified Files
- `src/ui/main_window.py` - Added Mode A/B radio buttons
- `src/pipeline/colmap_stage.py` - Dual-mode support
- `src/pipeline/batch_orchestrator.py` - Alignment mode routing

---

## Technical Details

### SphereSfM (json87/SphereSfM)

**What it is**: A modified COLMAP fork that natively supports spherical images (ERP format) with:
- **SPHERE camera model**: Custom camera type for equirectangular projection
- **Spherical bundle adjustment**: Cost function using lon/lat projection (`atan2`-based)
- **Cubic reprojection tool**: `sphere_cubic_reprojecer` to convert aligned spherical model to perspective views

**Key workflow**:
```
1. database_creator
2. feature_extractor --ImageReader.camera_model SPHERE --ImageReader.single_camera 1
3. sequential_matcher --SequentialMatching.overlap 10 --SequentialMatching.quadratic_overlap 1
4. mapper --Mapper.sphere_camera 1
5. sphere_cubic_reprojecer (transfers poses + reprojects features to cubemap)
```

**Camera parameters**: `SPHERE` model auto-detects from image dimensions (f=1, cx=width/2, cy=height/2)

### GPU Limitation

**Issue**: SphereSfM binary is compiled for CUDA 12.x; newer GPUs (CUDA 13.x) have kernel mismatch.
**Workaround**: Force CPU mode (`--SiftExtraction.use_gpu 0`, `--SiftMatching.use_gpu 0`)
**Performance**: ~75 seconds for 30 × 8K equirectangular images on CPU

### Kevin's panorama_sfm.py (Mode B)

**What it does**: Opposite approach - renders perspectives FIRST, then runs standard COLMAP with rig constraints.

**Workflow**:
```
1. Render 9 virtual perspectives per panorama
2. Extract features on perspectives (with visibility masks)
3. Apply rig config (hard geometric constraints)
4. Incremental mapping with rig BA refinement
```

---

## Two SphereSfM Integration Strategies

### Strategy A: Pure Spherical (Requires SphereSfM binary) ✅ IMPLEMENTED

**Concept**: Use SphereSfM's modified COLMAP directly on equirectangular images.

**Pros**:
- True spherical feature matching (handles wrap-around)
- Native equirectangular support
- Already proven in urban reconstruction
- 100% registration rate achieved

**Cons**:
- GPU kernels not compatible with newest CUDA (runs on CPU)
- Requires bundled SphereSfM binary
- Binary dependency management

**Implementation steps**:
1. Build SphereSfM binary (or distribute pre-built)
2. Create `sphere_sfm_integration.py` that calls SphereSfM CLI
3. After sparse reconstruction, call `sphere_cubic_reprojector` to generate perspective model
4. Output: Cubemap perspectives with inherited poses + reprojected features

---

### Strategy B: Hybrid Spherical-Perspective (Recommended)

**Concept**: Match on spheres conceptually, but use perspective rendering for standard COLMAP.

**Key insight from SphereSfM source code**:
- SphereSfM's `sphere_cubic_reprojector` shows how to transfer poses from sphere to perspective
- The math: `cubic_qvec = concatenate(sphere_qvec, inverse(cubic_rotation))`
- Features are reprojected: 3D point → project to each cubic face → verify visibility

**Workflow**:
```
STAGE A: Spherical Pose Estimation (coarse)
  1. Extract SIFT on equirectangular (with pole masking)
  2. Match between panoramas
  3. Estimate relative rotations using spherical geometry
  4. Build connectivity graph

STAGE B: Perspective Refinement (fine)
  1. Render perspectives at estimated poses
  2. Extract features on perspectives
  3. Apply rig constraints with INITIAL poses from Stage A
  4. Bundle adjustment refines the poses

STAGE C: Feature Transfer
  1. Spherical features → project to perspectives
  2. OR: Re-extract on perspectives (current approach, more robust)
```

---

## Recommended Implementation: Hybrid with Pose Initialization

### Phase 1: Spherical Pose Estimation

```python
def estimate_spherical_poses(equirect_images: List[Path]) -> Dict[str, np.ndarray]:
    """
    Estimate relative poses between panoramas using spherical matching.
    
    Returns dict mapping image_name -> rotation matrix (3x3)
    """
    # 1. Extract SIFT on equirectangular images
    # Use pole mask to ignore distorted regions near poles
    
    # 2. For each pair of images:
    #    - Match SIFT features
    #    - Convert 2D matches to 3D bearing vectors using spherical projection:
    #      u = (x - cx) / size
    #      v = (y - cy) / size  
    #      lon = u * 2 * pi
    #      lat = v * pi
    #      bearing = [sin(lon)*cos(lat), -sin(lat), cos(lon)*cos(lat)]
    #    - Estimate relative rotation using 5-point algorithm on unit sphere
    
    # 3. Build rotation averaging problem
    # 4. Solve for global rotations
    
    return poses
```

### Phase 2: Pose-Initialized Rig SfM

```python
def run_rig_sfm_with_init(equirect_dir, init_poses, output_dir):
    """
    Run Rig-based SfM with initial pose estimates from spherical matching.
    """
    # 1. Render perspectives (same as current approach)
    perspectives = render_perspectives(equirect_dir, output_dir)
    
    # 2. Extract features on perspectives
    # 3. Apply rig config
    
    # 4. CRITICAL: Initialize reconstruction with spherical poses
    #    - Write custom cameras.txt with estimated poses
    #    - Use mapper --Mapper.init_pose_from_file
    
    # 5. Bundle adjustment refines everything
```

### Phase 3: Feature Reprojection (Optional Enhancement)

```python
def reproject_spherical_features(
    sphere_features: Dict[str, np.ndarray],  # {img: Nx2 keypoints}
    sphere_to_perspective_rotations: Dict[int, np.ndarray],  # Virtual camera rotations
    camera: pycolmap.Camera
) -> Dict[str, np.ndarray]:
    """
    Reproject features from equirectangular to perspective views.
    
    This allows feature INHERITANCE from sphere to perspectives.
    """
    for kp_uv in sphere_features[img]:
        # 1. Convert 2D to bearing vector (sphere surface)
        lon = (kp_uv[0] / width - 0.5) * 2 * np.pi
        lat = (0.5 - kp_uv[1] / height) * np.pi
        bearing = [np.sin(lon)*np.cos(lat), -np.sin(lat), np.cos(lon)*np.cos(lat)]
        
        # 2. For each perspective camera:
        for cam_idx, R in sphere_to_perspective_rotations.items():
            # Transform bearing to camera frame
            bearing_cam = R @ bearing
            
            # Check visibility (z > 0 and within FOV)
            if bearing_cam[2] > 0:
                # Project to image plane
                x = focal * bearing_cam[0] / bearing_cam[2] + cx
                y = focal * bearing_cam[1] / bearing_cam[2] + cy
                
                if 0 <= x < width and 0 <= y < height:
                    # Feature visible in this perspective
                    perspective_features[cam_idx].append([x, y])
```

---

## SphereSfM Camera Model Reference

From SphereSfM source (`camera_models.h`):

```cpp
// SphereCameraModel::WorldToImage
void SphereCameraModel::WorldToImage(const T* params, const T u, const T v, T* x, T* y) {
    const T c1 = params[1];  // cx = width/2
    const T c2 = params[2];  // cy = height/2
    const T size = std::max(2*c1, 2*c2);
    
    *x = u * size + c1;
    *y = v * size + c2;
}

// SphereCameraModel::ImageToWorld
void SphereCameraModel::ImageToWorld(const T* params, const T x, const T y, T* u, T* v) {
    const T c1 = params[1];
    const T c2 = params[2];
    const T size = std::max(2*c1, 2*c2);
    
    *u = (x - c1) / size;
    *v = (y - c2) / size;
}

// Spherical Bundle Adjustment Cost Function
bool operator()(...) {
    // Rotate and translate point
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection += tvec;
    
    // Project to spherical coordinates
    const T lon = ceres::atan2(projection[0], projection[2]);
    const T lat = ceres::atan2(-projection[1], ceres::hypot(projection[0], projection[2]));
    
    // Normalize to image coordinates
    projection[0] = lon / (2 * M_PI);   // -0.5 to 0.5
    projection[1] = -lat / (2 * M_PI);  // -0.25 to 0.25 (for full sphere)
    
    // Convert to pixels
    CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);
    residuals[0] -= observed_x;
    residuals[1] -= observed_y;
}
```

---

## Cubic Reprojection Reference

From SphereSfM (`reconstruction.cc::ExportPerspectiveCubic`):

```cpp
// For each registered sphere image:
for (const image_t image_id : reg_image_ids_) {
    const Image& sphere_image = Image(image_id);
    const Camera& sphere_camera = Camera(sphere_image.CameraId());
    
    if (!sphere_camera.IsSpherical()) continue;
    
    // Create pinhole camera for cubemap face
    Camera pinhole_camera = PinholeCamera(image_size, image_size, field_of_view);
    
    // Generate 6 cubemap images (or custom faces)
    for (int cubic_id : cubic_image_ids) {
        // Get rotation for this cubemap face
        Eigen::Matrix3d rotation = rotations.at(cubic_id);
        
        // Compute cubic image pose
        // Key: Concatenate sphere rotation with cubemap face rotation
        cubic_image.Qvec() = ConcatenateQuaternions(
            RotationMatrixToQuaternion(sphere_image.RotationMatrix()),
            RotationMatrixToQuaternion(rotation.inverse())
        );
        cubic_image.Tvec() = -1.0 * QuaternionRotatePoint(
            cubic_image.Qvec(), 
            sphere_image.ProjectionCenter()
        );
    }
}

// Reproject 3D points to new perspectives
for (const Point3D& point3D : points3D_) {
    for (int cubic_id : cubic_image_ids) {
        // Project point to cubic image
        Eigen::Vector2d projection = ProjectPointToImage(
            point3D.XYZ(), 
            cubic_image.ProjectionMatrix(), 
            cubic_camera
        );
        
        // Check if visible (in bounds + facing camera)
        if (projection in bounds && angle < 90°) {
            new_track.AddElement(cubic_image_id, observation_index);
        }
    }
}
```

---

## Implementation Roadmap

### Phase 1: Basic Spherical Matching (Week 1)
- [ ] Create `src/premium/sphere_sfm_integration.py`
- [ ] Implement spherical feature extraction (SIFT on equirect with pole mask)
- [ ] Implement bearing vector conversion (2D → 3D unit sphere)
- [ ] Implement spherical relative rotation estimation

### Phase 2: Pose Transfer (Week 2)
- [ ] Implement rotation averaging (global pose from pairwise)
- [ ] Integrate with existing rig SfM as pose initialization
- [ ] Test on 30-frame dataset

### Phase 3: Feature Reprojection (Week 3)
- [ ] Implement feature projection from sphere to perspectives
- [ ] Compare registration rates: inherited vs re-extracted features
- [ ] Optimize for speed (vectorized operations)

### Phase 4: UI Integration (Week 4)
- [ ] Add "SphereSfM" option in Stage 4 method selector
- [ ] Add spherical matching parameters (pole mask, min matches)
- [ ] Update batch orchestrator to route appropriately

---

## Performance Comparison (Theoretical)

| Metric | Rig-based SfM | SphereSfM (Hybrid) |
|--------|---------------|-------------------|
| Feature Extraction | 9× per panorama | 1× per panorama (sphere) + optional 9× (perspective) |
| Feature Matching | O(N² × 9²) = 81N² | O(N²) on sphere + inherited |
| Registration Init | Cold start | Warm start (sphere poses) |
| Expected Speed | Baseline | 2-4× faster (less matching) |
| Expected Quality | 100% (proven) | TBD (depends on spherical matching) |

---

## Key Files to Create

1. **`src/premium/sphere_sfm_integration.py`** - Main SphereSfM implementation
2. **`src/premium/spherical_matching.py`** - Spherical feature matching utilities
3. **`src/premium/pose_transfer.py`** - Sphere-to-perspective pose conversion
4. **`src/premium/feature_reprojection.py`** - Feature transfer utilities

---

## References

1. SphereSfM GitHub: https://github.com/json87/SphereSfM
2. Jiang et al. "3D reconstruction of spherical images based on incremental SfM" (IJRS 2024)
3. Jiang et al. "Reliable Feature Matching for Spherical Images via Local Geometric Rectification" (Remote Sensing 2023)
4. Kevin's panorama_sfm.py (local reference implementation)

---

## Next Steps

1. **Immediate**: Implement basic spherical pose estimation
2. **Short-term**: Integrate as initialization for rig SfM
3. **Long-term**: Evaluate whether full SphereSfM binary is worth distributing

The hybrid approach is recommended because:
- Uses standard COLMAP (no custom binary needed)
- Leverages proven rig SfM (100% registration)
- Spherical matching provides better initialization
- Feature reprojection can reduce extraction time
C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\teste+01022026_1931