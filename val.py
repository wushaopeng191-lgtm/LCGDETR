"""Evaluate one locally trained checkpoint on a selected dataset split."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Local best.pt checkpoint.")
    parser.add_argument("--data", type=Path, default=ROOT / "URPC2020/data.yaml", help="Dataset YAML.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="Enable FP16; default is FP32.")
    parser.add_argument("--plots", action="store_true", help="Save PR curves and validation plots.")
    parser.add_argument("--profile", action="store_true", help="Also estimate GFLOPs with THOP; FP32 is recommended.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/val")
    parser.add_argument("--name", default="lcg-detr")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--json", type=Path, help="Summary path; default: metrics.json in this validation run.")
    return parser


def finite_number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("imgsz", "batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.device.lower() in {"cpu", "mps"}:
        parser.error("This source release requires an NVIDIA CUDA GPU")
    weights = args.weights.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    if weights.suffix.lower() != ".pt" or not weights.is_file():
        parser.error(f"--weights must be an existing local .pt checkpoint: {weights}")
    if data_yaml.suffix.lower() not in {".yaml", ".yml"} or not data_yaml.is_file():
        parser.error(f"--data must be an existing dataset YAML: {data_yaml}")
    if args.json and args.json.suffix.lower() != ".json":
        parser.error("--json must name a .json summary file")

    from prettytable import PrettyTable
    from ultralytics import RTDETR

    model = RTDETR(str(weights))
    result = model.val(
        data=str(data_yaml), split=args.split,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        device=args.device, half=args.half, plots=args.plots,
        save_json=False,
        project=str(args.project.expanduser().resolve()), name=args.name,
        exist_ok=args.exist_ok,
    )
    metrics = {name: finite_number(result.results_dict[key]) for name, key in METRIC_KEYS.items()}
    if any(value is None for value in metrics.values()):
        raise ValueError("The validator returned a non-finite detection metric")

    table = PrettyTable(["Class", "P (%)", "R (%)", "F1 (%)", "mAP50 (%)", "mAP75 (%)", "mAP50:95 (%)"])
    class_metrics = []
    box = result.box
    for index, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        values = {
            "precision": finite_number(box.p[index]),
            "recall": finite_number(box.r[index]),
            "F1": finite_number(box.f1[index]),
            "mAP50": finite_number(box.ap50[index]),
            "mAP75": finite_number(box.all_ap[index, 5]),
            "mAP50_95": finite_number(box.ap[index]),
        }
        name = result.names[class_id]
        class_metrics.append({"class_id": class_id, "class_name": name, **values})
        table.add_row([name] + [f"{value * 100:.2f}" if value is not None else "--" for value in values.values()])
    table.add_row([
        "all", f"{metrics['precision'] * 100:.2f}", f"{metrics['recall'] * 100:.2f}", "--",
        f"{metrics['mAP50'] * 100:.2f}", "--", f"{metrics['mAP50_95'] * 100:.2f}",
    ])

    speed = {key: finite_number(result.speed.get(key)) for key in ("preprocess", "inference", "postprocess")}
    total_ms = sum(speed.values()) if all(v is not None for v in speed.values()) else None
    fps = 1000 / total_ms if total_ms is not None and total_ms > 0 else None
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    flops = None
    profile_error = None
    if args.profile:
        try:
            from ultralytics.utils.torch_utils import get_flops
            estimate = finite_number(get_flops(model.model, imgsz=args.imgsz))
            if estimate is not None and estimate > 0:
                flops = estimate
            else:
                profile_error = "THOP did not return a positive GFLOPs estimate"
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"

    summary = {
        "checkpoint": str(weights), "data": str(data_yaml), "split": args.split,
        "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        "precision_mode": "FP16" if args.half else "FP32",
        "metric_scale": "0-1", "metrics": metrics, "class_metrics": class_metrics,
        "parameters_after_validation": parameter_count,
        "gflops_estimate": flops, "profile_error": profile_error,
        "checkpoint_size_MiB": weights.stat().st_size / (1024 ** 2),
        "speed_ms_per_image": speed, "fps_preprocess_inference_postprocess": fps,
        "timing_note": "Validator timings exclude image-file loading and output writing; not full predict() latency.",
    }
    save_dir = Path(result.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.json.expanduser().resolve() if args.json else save_dir / "metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = (
        f"Checkpoint: {weights}\nDataset: {data_yaml}\nSplit: {args.split}\n"
        f"Parameters after validation: {parameter_count:,}\n"
        f"GFLOPs estimate: {flops if flops is not None else 'not available / not requested'}\n"
        f"Validator FPS (preprocess + inference + postprocess): {fps if fps is not None else 'not available'}\n"
        f"Input: {args.imgsz}, batch={args.batch}, {'FP16' if args.half else 'FP32'}\n\n{table}\n"
    )
    (save_dir / "paper_data.txt").write_text(text, encoding="utf-8")
    print(text)
    if profile_error:
        print(f"Complexity estimate unavailable: {profile_error}")
    print(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    main()

