"""
Fine-tune YOLOv8 on the PPE dataset.

Auto-detects CUDA, falls back to CPU if no GPU is available.
Accepts CLI args for all common options so you don't have to edit
the file between runs.

Examples
--------
    # Default: 30 epochs, 640 imgsz, batch 16, auto device
    python train_model.py

    # Smaller model, more epochs, explicit GPU
    python train_model.py --model yolov8n.pt --epochs 50 --device 0

    # CPU training (much slower — reduce epochs and batch first)
    python train_model.py --device cpu --epochs 5 --batch 4

    # Resume from a previous run
    python train_model.py --resume models/ppe_detector/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def auto_device() -> str | int:
    """Return CUDA GPU 0 if available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on the Construction Safety PPE dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="yolov8s.pt",
                    help="Base YOLO model (e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    ap.add_argument("--data", default="datasets/ppe/data.yaml",
                    help="Dataset YAML file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience")
    ap.add_argument("--device", default=None,
                    help="Device: 'cpu', '0', 'cuda:0', etc. Default: auto")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (0 on Windows to avoid issues)")
    ap.add_argument("--name", default="ppe_detector",
                    help="Run name (folder under --project)")
    ap.add_argument("--project", default="models",
                    help="Results parent folder")
    ap.add_argument("--resume", default=None,
                    help="Resume from a previous last.pt checkpoint")
    args = ap.parse_args()

    # Resolve device
    device = args.device if args.device is not None else auto_device()
    print(f"-> Device: {device}")

    # Check dataset yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset yaml not found: {data_path}", file=sys.stderr)
        print(
            "Fix: make sure your datasets/ folder is next to this script, "
            "and that data.yaml's train/val/test paths are correct.",
            file=sys.stderr,
        )
        return 2

    # Lazy import so --help works without torch/ultralytics installed
    from ultralytics import YOLO

    if args.resume:
        print(f"-> Resuming from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        print(f"-> Loading base model: {args.model}")
        model = YOLO(args.model)
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=device,
            workers=args.workers,
            name=args.name,
            project=args.project,
        )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print()
    print("-" * 60)
    print("Training complete!")
    print(f"  Best weights  : {best}")
    print(f"  Last weights  : {best.with_name('last.pt')}")
    print()
    print("Use with:")
    print(f"  python run_inference.py --model {best} -i your_image.jpg -o out.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
