from pathlib import Path
from datetime import datetime
import json
import hashlib
import sys
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_orchestrator import PipelineWorker

input_file = r"G:\.shortcut-targets-by-id\12X9Cn_caDGuRMIO-hF6196FMdQyGNUDA\PROJETOS - CHICO SOMBRA\VIDEOS 360\VID_20251215_170106_00_211.insv"
root = Path(r"C:\Users\User\Documents\ARQUIVOS_TESTE\reextract_sdk")
run_dir = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

config = {
    "input_file": input_file,
    "output_dir": str(run_dir),
    "enable_stage1": True,
    "enable_stage2": False,
    "enable_stage3": False,
    "fps": 1.0,
    "start_time": 0.0,
    "end_time": 12.0,
    "extraction_method": "sdk_stitching",
    "sdk_quality": "best",
    "sdk_resolution": "8k",
    "output_format": "jpg",
    "allow_fallback": False,
}

worker = PipelineWorker(config)
result = worker._execute_stage1()
frames_dir = run_dir / "extracted_frames"
frames = sorted([p for p in frames_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])

def md5(p):
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()

stats = []
for p in frames[:5]:
    im = Image.open(p).convert("RGB")
    st = ImageStat.Stat(im)
    stats.append({
        "name": p.name,
        "size": im.size,
        "mean": [round(x, 2) for x in st.mean],
        "stddev": [round(x, 2) for x in st.stddev],
        "md5": md5(p),
    })

summary = {
    "success": result.get("success"),
    "error": result.get("error"),
    "count": len(frames),
    "run_dir": str(run_dir),
    "unique_md5": len({md5(p) for p in frames}),
    "sample_stats": stats,
}
print(json.dumps(summary, indent=2))
