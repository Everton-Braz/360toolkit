from src.extraction.sdk_extractor import SDKExtractor


def test_sdk_recovery_scope_ignores_native_insta360_without_a1_tokens(tmp_path) -> None:
    sample = tmp_path / "sample_x4.insv"
    sample.write_bytes(b"Insta360 X4" + b"\x00" * 1024)

    extractor = SDKExtractor.__new__(SDKExtractor)

    assert extractor._source_requires_sdk_recovery([str(sample)]) is False
    assert extractor._should_use_sparse_retry_strategy(False, [0, 12, 24], 12, {}) is False
    assert extractor._should_use_dense_overlap_strategy(False, list(range(20)), 1, {}) is False


def test_sdk_recovery_scope_enables_for_a1_camtype_tokens(tmp_path) -> None:
    sample = tmp_path / "sample_a1.insv"
    sample.write_bytes(b"header" + b"_112_" + b"payload" + b"_155_" + b"trailer")

    extractor = SDKExtractor.__new__(SDKExtractor)

    assert extractor._source_requires_sdk_recovery([str(sample)]) is True
    assert extractor._should_use_sparse_retry_strategy(True, [0, 12, 24], 12, {}) is True
    assert extractor._should_use_dense_overlap_strategy(True, list(range(20)), 1, {}) is True


def test_sparse_retry_only_retries_missing_frames(tmp_path) -> None:
    extractor = SDKExtractor.__new__(SDKExtractor)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    sample = tmp_path / "sample_a1.insv"
    sample.write_bytes(b"header" + b"_112_" + b"payload" + b"_155_" + b"trailer")

    preflight_calls = []
    retried_frames = []

    def fake_detect_input_files(path):
        return [str(path)]

    def fake_patch_input_files(input_files):
        return []

    def fake_extract_frames(**kwargs):
        preflight_calls.append(kwargs)
        (output_dir / "0.png").write_bytes(b"ok")
        (output_dir / "24.png").write_bytes(b"ok")
        return [str(output_dir / "0.png"), str(output_dir / "24.png")]

    def fake_collect_frame_paths(current_output_dir, frame_indices, output_format):
        return [
            str(current_output_dir / f"{frame_index}.png")
            for frame_index in frame_indices
            if (current_output_dir / f"{frame_index}.png").exists()
        ]

    def fake_recover_single_frame_with_retry(**kwargs):
        frame_index = kwargs["frame_index"]
        retried_frames.append(frame_index)
        recovered = output_dir / f"{frame_index}.png"
        recovered.write_bytes(b"ok")
        return str(recovered)

    extractor._detect_input_files = fake_detect_input_files
    extractor._patch_insv_camtype_if_needed = fake_patch_input_files
    extractor.extract_frames = fake_extract_frames
    extractor._collect_frame_paths = fake_collect_frame_paths
    extractor._recover_single_frame_with_retry = fake_recover_single_frame_with_retry

    frames = extractor._extract_sparse_indices_with_retry(
        input_path=str(sample),
        output_dir=str(output_dir),
        fps=2.0,
        quality="best",
        resolution=None,
        output_format="png",
        progress_callback=None,
        sdk_options={},
        video_fps=24.0,
        frame_indices=[0, 12, 24],
        total_frames=120,
    )

    assert len(preflight_calls) == 1
    assert retried_frames == [12]
    assert [tmp_path / "output" / "0.png", tmp_path / "output" / "12.png", tmp_path / "output" / "24.png"]
    assert frames == [
        str(output_dir / "0.png"),
        str(output_dir / "12.png"),
        str(output_dir / "24.png"),
    ]