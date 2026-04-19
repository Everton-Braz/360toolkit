from src.extraction.frame_extractor import FrameExtractor


def test_ffmpeg_stitched_rejects_insv_input(tmp_path) -> None:
    extractor = FrameExtractor.__new__(FrameExtractor)
    extractor.has_ffmpeg = True
    extractor.ffmpeg_path = 'ffmpeg'
    extractor.ffprobe_path = 'ffprobe'
    extractor.is_cancelled = False

    input_file = tmp_path / 'sample.insv'
    input_file.write_bytes(b'insv')
    output_dir = tmp_path / 'output'

    result = extractor.extract_frames(
        input_file=str(input_file),
        output_dir=str(output_dir),
        method='ffmpeg_stitched',
    )

    assert result['success'] is False
    assert 'FFmpeg stitched extraction is not supported for .insv input' in result['error']


def test_get_video_stream_helpers_prefer_default_stream(monkeypatch) -> None:
    extractor = FrameExtractor.__new__(FrameExtractor)
    extractor.ffprobe_path = 'ffprobe'

    monkeypatch.setattr(
        extractor,
        '_read_media_metadata',
        lambda _path: {
            'streams': [
                {'index': 0, 'codec_type': 'video', 'width': 1920, 'height': 1080, 'disposition': {'default': 0}},
                {'index': 2, 'codec_type': 'audio'},
                {'index': 3, 'codec_type': 'video', 'width': 3840, 'height': 2160, 'disposition': {'default': 1}},
            ]
        },
    )

    assert extractor.get_video_stream_count('sample.mp4') == 2
    assert extractor.get_primary_video_stream_index('sample.mp4') == 3


def test_get_primary_video_stream_index_falls_back_to_largest_video_stream(monkeypatch) -> None:
    extractor = FrameExtractor.__new__(FrameExtractor)
    extractor.ffprobe_path = 'ffprobe'

    monkeypatch.setattr(
        extractor,
        '_read_media_metadata',
        lambda _path: {
            'streams': [
                {'index': 0, 'codec_type': 'video', 'width': 1280, 'height': 720},
                {'index': 1, 'codec_type': 'video', 'width': 3840, 'height': 2160},
            ]
        },
    )

    assert extractor.get_video_stream_count('sample.mp4') == 2
    assert extractor.get_primary_video_stream_index('sample.mp4') == 1