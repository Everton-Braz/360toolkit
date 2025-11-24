import sys
import os
from pathlib import Path
from PyQt6.QtCore import QCoreApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.batch_orchestrator import PipelineWorker

def run_test(test_name, output_dir, input_file):
    print(f"Starting {test_name}...")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    # Create QCoreApplication
    app = QCoreApplication(sys.argv)
    
    config = {
        'input_file': str(input_file),
        'output_dir': str(output_dir),
        'fps': 5.0, # Extract 5 frames per second
        'start_time': 0.0,
        'end_time': 1.0, # 1 second duration
        'extraction_method': 'sdk', # Use SDK to test the fix
        'sdk_quality': 'draft', # Faster
        'output_format': 'jpg',
        'enable_stage1': True,
        'enable_stage2': True,
        'enable_stage3': True,
        'split_count': 4,
        'h_fov': 110,
        'masking_enabled': True,
        'masking_categories': {
            'persons': True,
            'personal_objects': False,
            'animals': False
        },
        'masking_model': 'yolov8n-seg.pt',
        'masking_confidence': 0.25,
        'use_gpu': False, # Force CPU for safety in test, or True if you want to test GPU
        'allow_fallback': False # Disable fallback to FFmpeg to test SDK
    }
    
    worker = PipelineWorker(config)
    
    def on_progress(current, total, msg):
        print(f"[{test_name}] Progress: {current}/{total} - {msg}")
        
    def on_stage_complete(stage, results):
        print(f"[{test_name}] Stage {stage} Complete. Success: {results.get('success')}")
        if not results.get('success'):
            print(f"[{test_name}] Stage {stage} Error: {results.get('error')}")
        
    def on_finished(results):
        print(f"[{test_name}] Pipeline Finished.")
        print(f"Success: {results.get('success')}")
        if not results.get('success'):
            print(f"Error: {results.get('error')}")
        app.quit()
        
    def on_error(err):
        print(f"[{test_name}] Critical Error: {err}")
        app.quit()

    worker.progress.connect(on_progress)
    worker.stage_complete.connect(on_stage_complete)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    
    print("Starting worker...")
    worker.start()
    
    # Run event loop
    exit_code = app.exec()
    sys.exit(exit_code)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline_test.py <output_dir> <input_file>")
        sys.exit(1)
        
    out_dir = Path(sys.argv[1])
    in_file = Path(sys.argv[2])
    
    run_test("TestRun", out_dir, in_file)
