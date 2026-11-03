"""Train a paper detector from YAML on URPC2020 or DUO."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/ablation/WCSPELAN_LCGAConvGLU_SpatialSepC3.yaml"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Detector YAML; no pretrained checkpoint is loaded.")
    parser.add_argument("--data", type=Path, default=ROOT / "URPC2020/data.yaml", help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience; 0 disables early stopping.")
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA device; one visible GPU is the verified setup.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=2000, help="Warmup iterations in the bundled trainer.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="lcg-detr")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision; paper default is FP32.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action="store_true", help="Save training/validation plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing output name.")
    return parser


def checked_file(parser, path, suffixes, label):
    path = path.expanduser().resolve()
    if path.suffix.lower() not in suffixes:
        parser.error(f"{label} must have one of these extensions: {', '.join(suffixes)}")
    if not path.is_file():
        parser.error(f"{label} does not exist: {path}")
    return path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("epochs", "batch", "imgsz"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("workers", "patience", "warmup_iters"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.lr0 <= 0 or args.lrf <= 0 or args.weight_decay < 0:
        parser.error("Learning rates must be positive and weight decay must be nonnegative")
    if args.device.lower() in {"cpu", "mps"}:
        parser.error("This source release requires an NVIDIA CUDA GPU")
    model_yaml = checked_file(parser, args.model, (".yaml", ".yml"), "Model YAML")
    data_yaml = checked_file(parser, args.data, (".yaml", ".yml"), "Dataset YAML")

    # Keep help and input checks usable without importing the CUDA-based modules.
    from ultralytics import RTDETR

    model = RTDETR(str(model_yaml))
    return model.train(
        data=str(data_yaml),
        imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
        batch=args.batch, workers=args.workers, device=args.device,
        optimizer="AdamW", lr0=args.lr0, lrf=args.lrf, momentum=0.9,
        weight_decay=args.weight_decay,
        # The bundled trainer interprets warmup_epochs as iterations.
        warmup_epochs=args.warmup_iters,
        seed=args.seed, deterministic=args.deterministic, amp=args.amp,
        pretrained=False, cache=False, plots=args.plots,
        project=str(args.project.expanduser().resolve()), name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()

