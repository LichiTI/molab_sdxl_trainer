
import sys
import os
import json
import argparse
from pathlib import Path
import logging

# Configure logging to stderr to keep stdout clean for IPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "upscaler"

def _coerce_params(config_or_params):
    if not isinstance(config_or_params, dict):
        raise TypeError("Upscaler params must be a dict")
    params = config_or_params.get("params")
    if isinstance(params, dict):
        return params
    return config_or_params

def run_upscale(params):
    from core.upscaler.upscaler_engine import UpscalerEngine
    
    input_path = params['input_path']
    output_path = params['output_path']
    model_name = params['model_name']
    scale = params.get('scale', 4)
    fmt = params.get('format', 'png')
    
    # Locate model
    model_path = params.get('model_path')
    if model_path:
        model_path = Path(model_path)
    if not model_path:
        # Fallback search
        model_path = MODELS_DIR / f"{model_name}.pth"
        if not model_path.exists():
            possible = list(MODELS_DIR.glob(f"{model_name}*.pth"))
            if possible:
                model_path = possible[0]
            else:
                raise FileNotFoundError(f"Model not found: {model_name}")
    elif not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    logger.info(f"[UpscalerWorker] Started. PID: {os.getpid()}")
    logger.info(f"[UpscalerWorker] Interpreter: {sys.executable}")
    logger.info(f"Loading upscaler: {model_path} (x{scale})")
    
    engine = UpscalerEngine()
    engine.load_model(str(model_path), scale=scale)
    
    logger.info(f"Processing: {input_path} -> {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    engine.upscale_image(str(input_path), str(output_path), format=fmt)
    logger.info("Upscale completed successfully")

def probe_acceleration(_params=None):
    from core.upscaler.tensorrt_acceleration import probe_acceleration as _probe

    return _probe()

def inspect_artifacts(params):
    from core.upscaler.tensorrt_acceleration import inspect_artifacts as _inspect

    return _inspect(
        model_name=str(params.get('model_name') or ''),
        model_path=str(params.get('model_path') or ''),
        scale=int(params.get('scale', 4)),
        tile_presets=params.get('tile_presets') or [128, 256],
        precision=str(params.get('precision', 'fp16')),
        dynamic_axes=bool(params.get('dynamic_axes', False)),
        opset=int(params.get('opset', 18)),
        workspace_mb=int(params.get('workspace_mb', 2048)),
    )

def export_onnx(params):
    from core.upscaler.tensorrt_acceleration import default_onnx_path, export_rrdb_to_onnx

    model_path = params.get('model_path')
    model_name = params.get('model_name')
    if not model_path and model_name:
        candidate = MODELS_DIR / f"{model_name}.pth"
        if candidate.exists():
            model_path = str(candidate)
        else:
            matches = list(MODELS_DIR.glob(f"{model_name}*.pth"))
            if matches:
                model_path = str(matches[0])
    if not model_path:
        raise FileNotFoundError("No .pth upscaler model was provided for ONNX export")

    output_path = params.get('output_path')
    if not output_path:
        if params.get('output_dir'):
            output_dir = Path(params.get('output_dir'))
            output_path = str(output_dir / f"{Path(model_path).stem}_tile{int(params.get('tile_size', 256))}.onnx")
        else:
            output_path = str(default_onnx_path(model_path, tile_size=int(params.get('tile_size', 256))))

    return export_rrdb_to_onnx(
        model_path=str(model_path),
        output_path=str(output_path),
        scale=int(params.get('scale', 4)),
        tile_size=int(params.get('tile_size', 256)),
        opset=int(params.get('opset', 18)),
        dynamic_axes=bool(params.get('dynamic_axes', False)),
        device=str(params.get('device', 'cpu')),
    )

def build_tensorrt(params):
    from core.upscaler.tensorrt_acceleration import build_tensorrt_engine, default_engine_path

    onnx_path = params.get('onnx_path') or params.get('model_path')
    if not onnx_path:
        raise FileNotFoundError("No .onnx model was provided for TensorRT engine build")

    output_path = params.get('output_path')
    precision = str(params.get('precision', 'fp16'))
    opt_tile_size = int(params.get('opt_tile_size', params.get('tile_size', 256)))
    if not output_path:
        if params.get('output_dir'):
            output_dir = Path(params.get('output_dir'))
            output_path = str(output_dir / f"{Path(onnx_path).stem}_{precision}.engine")
        else:
            output_path = str(default_engine_path(onnx_path, precision=precision, opt_tile_size=opt_tile_size))

    return build_tensorrt_engine(
        onnx_path=str(onnx_path),
        output_path=str(output_path),
        precision=precision,
        min_tile_size=int(params.get('min_tile_size', 128)),
        opt_tile_size=opt_tile_size,
        max_tile_size=int(params.get('max_tile_size', 512)),
        workspace_mb=int(params.get('workspace_mb', 2048)),
    )

def run_tensorrt_upscale(params):
    from core.upscaler.tensorrt_runtime import upscale_image_with_tensorrt

    engine_path = params.get('engine_path') or params.get('model_path')
    if not engine_path:
        raise FileNotFoundError("No TensorRT engine was provided for upscaling")
    input_path = params.get('input_path')
    if not input_path:
        raise FileNotFoundError("No input image was provided for TensorRT upscaling")

    output_path = params.get('output_path')
    if not output_path:
        output_dir = Path(params.get('output_dir') or Path(input_path).parent)
        fmt = str(params.get('format', 'png'))
        output_path = str(output_dir / f"{Path(input_path).stem}_tensorrt_x{int(params.get('scale', 4))}.{fmt}")

    return upscale_image_with_tensorrt(
        engine_path=str(engine_path),
        input_path=str(input_path),
        output_path=str(output_path),
        scale=int(params.get('scale', 4)),
        fmt=str(params.get('format', 'png')),
        tile_size=int(params.get('tile_size', 256)),
        tile_pad=int(params.get('tile_pad', 0)),
    )

def run_upscaler(config_or_params):
    params = _coerce_params(config_or_params)
    action = str(params.get('action') or 'upscale')
    if action == 'probe_acceleration':
        return probe_acceleration(params)
    if action == 'inspect_artifacts':
        return inspect_artifacts(params)
    if action == 'export_onnx':
        return export_onnx(params)
    if action == 'build_tensorrt':
        return build_tensorrt(params)
    if action == 'upscale_tensorrt':
        return run_tensorrt_upscale(params)
    return run_upscale(params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        result = run_upscaler(config)
        if result is not None:
            print("__LULYNX_UPSCALER_JSON__" + json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str))
            
    except Exception as e:
        logger.error(f"[Error] Upscaler Worker failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
