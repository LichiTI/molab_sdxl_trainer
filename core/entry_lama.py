
import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import logging

# Configure logging to stderr to keep stdout clean for IPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def _coerce_params(config_or_params):
    if not isinstance(config_or_params, dict):
        raise TypeError("LAMA params must be a dict")
    params = config_or_params.get("params")
    if isinstance(params, dict):
        return params
    return config_or_params

def run_inpaint(params):
    from simple_lama_inpainting import SimpleLama
    
    input_path = params['input_path']
    mask_path = params['mask_path']
    output_path = params['output_path']
    
    logger.info(f"[LAMAWorker] Started. PID: {os.getpid()}")
    logger.info(f"[LAMAWorker] Interpreter: {sys.executable}")
    logger.info(f"Loading LAMA model...")
    lama = SimpleLama()
    
    logger.info(f"Loading images...")
    image = Image.open(input_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Ensure resize inside worker if needed, but router usually handles or we assume safety
    if image.size != mask.size:
        logger.info("Resizing mask to match image...")
        mask = mask.resize(image.size, Image.NEAREST)
    
    logger.info("Inpainting...")
    result = lama(image, mask)
    
    logger.info(f"Saving to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")
    logger.info("Inpainting completed successfully")

def run_lama(config_or_params):
    return run_inpaint(_coerce_params(config_or_params))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        run_lama(config)
            
    except Exception as e:
        logger.error(f"[Error] LAMA Worker failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
