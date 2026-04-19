"""Benchmark: CPU multiprocessing vs OpenCL UMat split on RGBA images."""
import sys, time, shutil, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

from pathlib import Path
from src.pipeline.batch_orchestrator import PipelineWorker

INPUT_DIR  = r'C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\INPUT_TEST_INSV\teste110420261144\alpha_cutouts'
OUTPUT_DIR = r'C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\INPUT_TEST_INSV\teste110420261144\split_alpha_cutouts_out'

# Clean output
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

config = {
    'output_dir': OUTPUT_DIR,
    'enable_stage1': False,
    'enable_stage2': True,
    'enable_stage3': False,
    'skip_transform': False,
    'transform_type': 'perspective',
    'stage2_input_dir': INPUT_DIR,
    'stage2_format': 'png',
    'output_width': 1920,
    'output_height': 1920,
    'perspective_params': {
        'camera_groups': [
            {'camera_count': 6, 'pitch': 0,   'fov': 110},
            {'camera_count': 6, 'pitch': -30, 'fov': 110},
            {'camera_count': 6, 'pitch':  30, 'fov': 110},
        ]
    },
}

from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)
worker = PipelineWorker(config)

t_start = time.time()

def on_progress(cur, tot, msg):
    print(f'  [{cur}/{tot}] {msg}')

def on_stage_complete(stage, result):
    elapsed = time.time() - t_start
    count = result.get('perspective_count', result.get('cubemap_count', '?'))
    print(f'Stage {stage} complete: success={result.get("success")}, count={count}, elapsed={elapsed:.1f}s')

def on_finished(result):
    elapsed = time.time() - t_start
    print(f'\nPipeline finished in {elapsed:.1f}s, success={result.get("success")}')
    import cv2
    out_dir = Path(OUTPUT_DIR) / 'perspective_views'
    files = sorted(out_dir.glob('*.png')) if out_dir.exists() else []
    print(f'Output files: {len(files)}')
    ok = fail = 0
    for f in files[:5]:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        ch = img.shape[2] if img is not None and img.ndim == 3 else 0
        status = 'OK (RGBA)' if ch == 4 else f'FAIL (ch={ch})'
        print(f'  {f.name}: {img.shape} -> {status}')
        ok += (1 if ch == 4 else 0)
        fail += (0 if ch == 4 else 1)
    # check all remaining
    for f in files[5:]:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        ch = img.shape[2] if img is not None and img.ndim == 3 else 0
        ok += (1 if ch == 4 else 0)
        fail += (0 if ch == 4 else 1)
    print(f'\nAlpha check: PASS={ok}  FAIL={fail}  (total={len(files)})')
    app.quit()

def on_error(msg):
    print(f'ERROR: {msg}')
    app.quit()

worker.progress.connect(on_progress)
worker.stage_complete.connect(on_stage_complete)
worker.finished.connect(on_finished)
worker.error.connect(on_error)
worker.start()
app.exec()
