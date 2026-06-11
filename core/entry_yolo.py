import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.system_sleep_guard import SleepGuard


def _coerce_params(config_or_params):
    if not isinstance(config_or_params, dict):
        raise TypeError("YOLO params must be a dict")
    params = config_or_params.get("params")
    if isinstance(params, dict):
        return params
    return config_or_params


def run_yolo(config_or_params):
    from ultralytics import YOLO

    params = _coerce_params(config_or_params)
    model_name = str(params["model"]).strip()
    dataset_yaml = str(params["data"]).strip()
    output_dir = Path(str(params.get("output_dir") or "./output/yolo")).resolve()
    output_name = str(params.get("output_name") or "exp").strip()
    resume_path = str(params.get("resume") or "").strip()

    if not dataset_yaml:
        raise ValueError("Missing dataset yaml for YOLO training")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {
        "data": dataset_yaml,
        "epochs": int(params.get("epochs", 100)),
        "batch": int(params.get("batch", 16)),
        "imgsz": int(params.get("imgsz", 640)),
        "project": str(output_dir),
        "name": output_name,
        "exist_ok": True,
        "verbose": True,
    }

    workers = params.get("workers")
    if workers not in (None, ""):
        train_kwargs["workers"] = int(workers)

    device = str(params.get("device") or "").strip()
    if device:
        train_kwargs["device"] = device

    seed = params.get("seed")
    if seed not in (None, ""):
        train_kwargs["seed"] = int(seed)

    save_period = params.get("save_period")
    if save_period not in (None, ""):
        save_period = int(save_period)
        if save_period > 0:
            train_kwargs["save_period"] = save_period

    patience = params.get("patience")
    if patience not in (None, ""):
        train_kwargs["patience"] = int(patience)

    pretrained = params.get("pretrained")
    if pretrained not in (None, "") and not resume_path:
        train_kwargs["pretrained"] = bool(pretrained)

    logger.info("[YOLOWorker] Interpreter: %s", sys.executable)
    logger.info("[YOLOWorker] Model: %s", resume_path or model_name)
    logger.info("[YOLOWorker] Dataset: %s", dataset_yaml)
    logger.info("[YOLOWorker] Output: %s / %s", output_dir, output_name)

    if resume_path:
        model = YOLO(resume_path)
        train_kwargs["resume"] = True
    else:
        model = YOLO(model_name)

    with SleepGuard():
        model.train(**train_kwargs)
    logger.info("[YOLOWorker] Training completed successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        run_yolo(config)
    except Exception as exc:
        logger.error("[Error] YOLO Worker failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
