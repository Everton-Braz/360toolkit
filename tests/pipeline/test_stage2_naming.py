from pathlib import Path

from src.pipeline.export_formats import RealityScanExporter
from src.pipeline.stage2_naming import (
    build_stage2_frame_records,
    resolve_cubemap_output_path,
    resolve_perspective_output_path,
)


def test_build_stage2_frame_records_preserves_sparse_source_ids():
    records = build_stage2_frame_records(
        [
            Path("126.png"),
            Path("14.png"),
            Path("1008.png"),
            Path("994.png"),
            Path("0.png"),
        ],
        "preserve_source",
    )

    assert [path.name for path, _frame_id in records] == [
        "0.png",
        "14.png",
        "126.png",
        "994.png",
        "1008.png",
    ]
    assert [frame_id for _path, frame_id in records] == [0, 14, 126, 994, 1008]


def test_build_stage2_frame_records_supports_sequential_mode():
    records = build_stage2_frame_records(
        [Path("126.png"), Path("14.png"), Path("1008.png")],
        "sequential",
    )

    assert [path.name for path, _frame_id in records] == ["14.png", "126.png", "1008.png"]
    assert [frame_id for _path, frame_id in records] == [0, 1, 2]


def test_resolve_perspective_output_path_supports_camera_subfolders():
    output_path = resolve_perspective_output_path(
        Path("perspective_views"),
        frame_id=1008,
        cam_idx=3,
        extension="png",
        layout_mode="by_camera",
    )

    assert output_path.as_posix() == "perspective_views/cam_03/frame_01008_cam_03.png"


def test_resolve_cubemap_output_path_supports_tile_subfolders():
    output_path = resolve_cubemap_output_path(
        Path("perspective_views"),
        frame_id=1008,
        tile_name="front",
        extension="png",
        layout_mode="by_camera",
    )

    assert output_path.as_posix() == "perspective_views/front/frame_01008_front.png"


def test_build_perspective_mapping_supports_recursive_camera_layout(tmp_path):
    perspective_dir = tmp_path / "perspective_views"
    (perspective_dir / "cam_00").mkdir(parents=True)
    (perspective_dir / "cam_01").mkdir(parents=True)

    for relative_name in [
        "cam_00/frame_00000_cam_00.png",
        "cam_01/frame_00000_cam_01.png",
        "cam_00/frame_00014_cam_00.png",
        "cam_01/frame_00014_cam_01.png",
        "cam_00/frame_00994_cam_00.png",
        "cam_01/frame_00994_cam_01.png",
    ]:
        file_path = perspective_dir / relative_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"test")

    exporter = RealityScanExporter(str(tmp_path / "colmap"), str(tmp_path / "export"), str(perspective_dir))
    exporter.images = {
        1: {"name": "0.png", "camera_id": 1},
        2: {"name": "14.png", "camera_id": 1},
        3: {"name": "994.png", "camera_id": 1},
    }

    assert exporter.build_perspective_mapping() is True
    assert exporter.perspective_mapping["0.png"] == [
        "cam_00/frame_00000_cam_00.png",
        "cam_01/frame_00000_cam_01.png",
    ]
    assert exporter.perspective_mapping["14.png"] == [
        "cam_00/frame_00014_cam_00.png",
        "cam_01/frame_00014_cam_01.png",
    ]
    assert exporter.perspective_mapping["994.png"] == [
        "cam_00/frame_00994_cam_00.png",
        "cam_01/frame_00994_cam_01.png",
    ]