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


def _coerce_params(config_or_params):
    if not isinstance(config_or_params, dict):
        raise TypeError("YOLO annotation params must be a dict")
    params = config_or_params.get("params")
    if isinstance(params, dict):
        return params
    return config_or_params


def run_predict(config_or_params):
    from ultralytics import YOLO

    params = _coerce_params(config_or_params)
    image_path = str(params["image_path"]).strip()
    model_name = str(params.get("model") or "yolo11n.pt").strip()
    conf = float(params.get("conf", 0.25))
    iou = float(params.get("iou", 0.45))
    max_det = int(params.get("max_det", 300))
    device = str(params.get("device") or "").strip()

    logger.info("[YOLOAnnotate] Interpreter: %s", sys.executable)
    logger.info("[YOLOAnnotate] Model: %s", model_name)
    logger.info("[YOLOAnnotate] Image: %s", image_path)

    model = YOLO(model_name)
    predict_kwargs = {
        "source": image_path,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    records = []
    names = getattr(model, "names", {}) or {}
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().tolist()
        cls_list = boxes.cls.detach().cpu().tolist()
        conf_list = boxes.conf.detach().cpu().tolist()
        for idx, coords in enumerate(xyxy):
            class_id = int(cls_list[idx]) if idx < len(cls_list) else 0
            confidence = float(conf_list[idx]) if idx < len(conf_list) else None
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            records.append(
                {
                    "class_id": class_id,
                    "class_name": str(label),
                    "confidence": confidence,
                    "pixel_left": float(coords[0]),
                    "pixel_top": float(coords[1]),
                    "pixel_right": float(coords[2]),
                    "pixel_bottom": float(coords[3]),
                }
            )

    print(json.dumps({"boxes": records}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        run_predict(config)
    except Exception as exc:
        logger.error("[Error] YOLO annotation worker failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
