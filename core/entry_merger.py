
import sys
import os
import json
import argparse
import traceback
from pathlib import Path
import logging

# Configure logging to stderr to keep stdout clean for IPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Lazy imports will be done inside functions

def run_analyze(params):
    from core.tools.svd_merger import SVDMerger
    from core.accelerator import accelerator
    
    merger = SVDMerger(use_gpu=accelerator.has_gpu)
    result = merger.analyze(params['model_a'], params['model_b'], quick_mode=True)
    visualization = merger.get_visualization_data(result)
    print(f"RESULT_JSON:{json.dumps(visualization)}", flush=True)

def run_merge(params):
    from core.tools.svd_merger import SVDMerger, MergeMode
    from core.accelerator import accelerator
    
    merger = SVDMerger(use_gpu=accelerator.has_gpu, precision=params.get('output_precision', 'fp16'))
    
    # Mode enum
    try:
        mode = MergeMode(params.get('mode', 'weighted_sum'))
    except ValueError:
        mode = MergeMode.SVD_SMART
        
    merger.merge(
        model_a_path=params['model_a'],
        model_b_path=params['model_b'],
        output_path=params['output_path'],
        alpha_a=params.get('alpha', 0.5), # Assuming UI sends single alpha for ratio
        alpha_b=1.0 - params.get('alpha', 0.5),
        mode=mode,
        structure_bias=params.get('structure_bias', 0.5),
        detail_bias=params.get('detail_bias', 0.5),
        output_precision=params.get('output_precision', 'fp16'),
        layer_weights=params.get('block_weights')
    )
    logger.info("Merge completed successfully")
    print("Merge completed successfully", flush=True)

def run_extract(params):
    from core.tools.lora_surgery import LoRASurgeon
    surgeon = LoRASurgeon(device=params.get('device', 'cuda'))
    surgeon.extract_lora(
        base_model_path=params['base_model'],
        tuned_model_path=params['tuned_model'],
        output_path=params['output_path'],
        rank=params.get('rank', 128),
        device=params.get('device', 'cuda')
    )
    logger.info("Extraction completed successfully")

def run_merge_lora(params):
    from core.tools.lora_surgery import LoRASurgeon
    surgeon = LoRASurgeon(device=params.get('device', 'cuda'))
    surgeon.merge_loras_svd(
        lora_a_path=params['lora_a'],
        lora_b_path=params['lora_b'],
        output_path=params['output_path'],
        alpha_a=params.get('alpha_a', params.get('alpha', 0.5)),
        alpha_b=params.get('alpha_b', 1.0 - params.get('alpha', 0.5)),
        rank=params.get('rank', 128),
        layer_weights=params.get('block_weights')
    )
    logger.info("LoRA Merge completed successfully")

def run_bake(params):
    from core.tools.lora_surgery import LoRASurgeon
    surgeon = LoRASurgeon(device=params.get('device', 'cuda'))
    surgeon.bake_lora(
        base_model_path=params['base_model'],
        lora_path=params['lora_model'],
        output_path=params['output_path'],
        alpha=params.get('alpha', 1.0)
    )
    logger.info("Baking completed successfully")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        action = config.get("action")
        params = config.get("params", {})
        
        logger.info(f"[MergerWorker] Started. PID: {os.getpid()}")
        logger.info(f"[MergerWorker] Interpreter: {sys.executable}")
        logger.info(f"[MergerWorker] Starting action: {action}")
        
        if action == "analyze":
            run_analyze(params)
        elif action == "merge":
            run_merge(params)
        elif action == "extract":
            run_extract(params)
        elif action == "merge_lora":
            run_merge_lora(params)
        elif action == "bake":
            run_bake(params)
        else:
            raise ValueError(f"Unknown action: {action}")
            
    except Exception as e:
        logger.error(f"[Error] Worker failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
