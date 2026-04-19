"""
Test: Extract equirectangular frames from INSV, then SAM3-mask them.
Stage 1: SDK Stitching  →  equirectangular PNGs
Stage 3: SAM3.cpp mask  →  *_mask.png beside each frame
(Stage 2 / perspective split is skipped intentionally)

Usage:
    python test_extract_mask_epr.py
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('test_epr')

INPUT_INSV  = r'C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\INPUT_TEST_INSV\VID_20251215_170106_00_211.insv'
OUTPUT_ROOT = Path(r'C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE') / f'test_epr_{datetime.now().strftime("%d%m%Y_%H%M")}'

SAM3_SEGMENTER = Path(r'C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit\downloads\sam3cpp\build\examples\Release\segment_persons.exe')
SAM3_MODEL     = Path(r'C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit\downloads\sam3cpp\models\sam3-q4_0.ggml')


def _hr(title=''):
    print('\n' + '=' * 60)
    if title:
        print(f'  {title}')
        print('=' * 60)


def stage1_extract(output_dir: Path) -> list[Path]:
    """Extract equirectangular frames via SDK stitching."""
    from src.extraction.sdk_extractor import SDKExtractor

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = SDKExtractor()

    def progress(pct):
        print(f'\r  [{pct:3d}%] Extracting...', end='', flush=True)

    t0 = time.perf_counter()
    frames = extractor.extract_frames(
        input_path=INPUT_INSV,
        output_dir=str(output_dir),
        fps=0.5,            # 1 frame every 2 seconds → ~5 frames in 10 s
        start_time=0.0,
        end_time=10.0,
        quality='best',
        resolution=(3840, 1920),  # 4K equirectangular
        output_format='png',
        progress_callback=progress,
    )
    elapsed = time.perf_counter() - t0
    print()  # newline after carriage-return progress

    count = len(frames) if frames else 0
    print(f'  Extracted {count} frames in {elapsed:.1f}s')
    if count == 0:
        raise RuntimeError('Stage 1 returned no frames')
    return [Path(f) for f in frames]


def stage3_mask(frames: list[Path], mask_dir: Path):
    """Run SAM3.cpp batch masking on the equirectangular frames."""
    from src.masking.sam3_external_masker import SAM3ExternalMasker

    mask_dir.mkdir(parents=True, exist_ok=True)

    masker = SAM3ExternalMasker(
        segment_persons_exe=str(SAM3_SEGMENTER),
        model_path=str(SAM3_MODEL),
        use_gpu=True,
        morph_radius=0,
        alpha_export=False,
        max_input_width=3840,   # downscale 8K → 3840 for faster encode
        score_threshold=0.5,
        nms_threshold=0.1,
    )
    masker.set_enabled_categories({
        'persons': True,
        'bags':    True,
        'phones':  True,
        'hats':    True,
        'helmets': True,
        'sky':     False,  # sky usually fills everything in equirects — keep off
    })

    total = len(frames)
    def progress(cur, tot, msg):
        pct = (cur / tot * 100) if tot else 0
        print(f'\r  [{pct:5.1f}%] {msg}', end='', flush=True)

    t0 = time.perf_counter()
    result = masker.process_batch(
        input_dir=str(frames[0].parent),
        output_dir=str(mask_dir),
        save_visualization=False,
        progress_callback=progress,
    )
    elapsed = time.perf_counter() - t0
    print()

    ok      = result.get('successful', result.get('masks_created', 0))
    failed  = result.get('failed', 0)
    skipped = result.get('skipped', 0)
    print(f'  Masks:  {ok} created  |  {failed} failed  |  {skipped} skipped  in {elapsed:.1f}s')
    return result


def main():
    _hr('360ToolKit  –  Extract EPR + SAM3 Mask test')
    print(f'  Input : {INPUT_INSV}')
    print(f'  Output: {OUTPUT_ROOT}')
    print(f'  Time  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    # Prereq check
    missing = []
    if not Path(INPUT_INSV).exists():     missing.append(f'Input INSV:   {INPUT_INSV}')
    if not SAM3_SEGMENTER.exists():       missing.append(f'segment_persons.exe: {SAM3_SEGMENTER}')
    if not SAM3_MODEL.exists():           missing.append(f'SAM3 model:   {SAM3_MODEL}')
    if missing:
        print('ERROR – missing prerequisites:')
        for m in missing:
            print(f'  ✗  {m}')
        sys.exit(1)

    frames_dir = OUTPUT_ROOT / 'equirect_frames'
    masks_dir  = OUTPUT_ROOT / 'masks'

    # ── Stage 1 ──────────────────────────────────────────────────────
    _hr('STAGE 1: SDK Stitching  (0.5 FPS, first 10 s, 4K)')
    frames = stage1_extract(frames_dir)

    if not frames:
        print('  ERROR: no frames extracted.')
        sys.exit(1)

    print(f'\n  Frames saved to: {frames_dir}')
    for f in frames:
        print(f'    {f.name}')

    # ── Stage 3 ──────────────────────────────────────────────────────
    _hr('STAGE 3: SAM3 Masking  (persons + bags + phones + hats + helmets)')
    stage3_result = stage3_mask(frames, masks_dir)

    print(f'\n  Masks saved to: {masks_dir}')
    for p in sorted(masks_dir.glob('*_mask.png')):
        print(f'    {p.name}')

    # ── Summary ──────────────────────────────────────────────────────
    _hr('DONE')
    print(f'  Frames extracted : {len(frames)}')
    print(f'  Masks created    : {stage3_result.get("successful", stage3_result.get("masks_created", 0))}')
    print(f'  Output folder    : {OUTPUT_ROOT}')
    print()


if __name__ == '__main__':
    main()
