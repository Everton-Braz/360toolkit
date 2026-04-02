from pathlib import Path

from src.extraction.frame_extractor import FrameExtractor


def test_detects_x4_from_insv_trailer_string(tmp_path: Path) -> None:
    sample = tmp_path / "sample_x4.insv"
    sample.write_bytes(b'\x00' * 256 + b'Insta360 X4' + b'\x00' * 256)

    extractor = FrameExtractor()

    model, source = extractor._detect_camera_model(sample, '.insv', 3840, 3840)

    assert model == 'Insta360 X4'
    assert source == 'insv_trailer_string'


def test_detects_a1_from_camtype_tokens_when_name_missing(tmp_path: Path) -> None:
    sample = tmp_path / "sample_a1.insv"
    sample.write_bytes(b'header' + b'_112_' + b'payload' + b'_155_' + b'trailer')

    extractor = FrameExtractor()

    model, source = extractor._detect_camera_model(sample, '.insv', 3840, 3840)

    assert model == 'Antigravity A1'
    assert source == 'insv_trailer_camtype'