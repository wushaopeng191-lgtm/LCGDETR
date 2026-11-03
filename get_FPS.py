"""
完整检测流程 FPS 测试：支持 YOLO 和 RT-DETR。

计时包含：
    预处理、CPU→GPU、模型推理、原生后处理、检测框返回 CPU、
    predict() 接口开销。

计时不包含：
    加载权重、图片读取/解码、预热、绘图、结果文件保存。

使用真实图片，不需要标签，不计算 mAP。
"""

import csv
import hashlib
import json
import platform
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO, RTDETR
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.utils.torch_utils import select_device


# ================= 修改这里，然后直接运行 =================
CONFIG = {
    # YOLO 填 "yolo"，RT-DETR 填 "rtdetr"
    "model_type": "rtdetr",
    "name": "Baseline",

    # 修改为实际权重路径
    "weights": r"runs_urpc_ablation/urpc_seed_0/abl_R18_AIFI_RepC3/weights/best.pt",

    # 修改为真实测试图片目录，也可以填写单张图片路径
    # 两个模型务必使用同一批图片
    "images": r"URPC2020/test/images",

    "device": "0",
    "imgsz": 640,
    "half": False,          # False=FP32；True=FP16

    "conf": 0.25,
    "iou": 0.70,           # 适用于需要 NMS 的模型
    "max_det": 300,

    "max_images": 100,     # 按路径排序，取前100张并缓存到内存
    "warmup": 200,         # 预热请求次数
    "testtime": 1000,      # 计时请求次数，循环使用缓存图片

    "seed": 0,
    "save_dir": "urpc_fps_results",
}
# 本脚本固定 batch=1，每次处理一张图片。
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_images(source):
    """读取与解码在计时外完成，不提前缩放图片。"""
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        raise FileNotFoundError(f"图片路径不存在：{source}")

    paths = paths[:CONFIG["max_images"]]
    if not paths:
        raise ValueError(f"未找到图片：{source}")

    images = []
    manifest = []

    for path in paths:
        # 支持含中文字符的本地路径
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"图片无法读取：{path}")

        images.append(image)
        manifest.append({
            "index": len(images) - 1,
            "path": str(path),
            "sha256": sha256_file(path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        })

    return images, manifest


def build_detector(weights):
    kind = CONFIG["model_type"].lower()
    size = CONFIG["imgsz"]

    if kind == "yolo":
        detector = YOLO(str(weights), task="detect")
        base_predictor = DetectionPredictor
    elif kind == "rtdetr":
        detector = RTDETR(str(weights))
        base_predictor = RTDETRPredictor
    else:
        raise ValueError("model_type 只能是 'yolo' 或 'rtdetr'。")

    class FixedSizePredictor(base_predictor):
        def pre_transform(self, images):
            if kind == "yolo":
                # 禁用自动矩形缩放，保证实际输入为固定640×640。
                # 兼容旧版中 rect=False 未控制此处行为的情况。
                transform = LetterBox(
                    self.imgsz,
                    auto=False,
                    stride=self.model.stride,
                )
                return [transform(image=image) for image in images]

            # RT-DETR 保留其原生缩放方式。
            return super().pre_transform(images)

        def preprocess(self, images):
            tensor = super().preprocess(images)
            expected = (1, 3, size, size)
            if tuple(tensor.shape) != expected:
                raise RuntimeError(
                    f"实际输入为 {tuple(tensor.shape)}，"
                    f"预期为 {expected}，停止测速。"
                )
            return tensor

    return detector, FixedSizePredictor


def main():
    weights = resolve_path(CONFIG["weights"])
    image_source = resolve_path(CONFIG["images"])

    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"请填写有效的本地 .pt 权重：{weights}")

    if CONFIG["warmup"] < 1 or CONFIG["testtime"] < 1:
        raise ValueError("warmup 和 testtime 必须大于0。")
    if CONFIG["max_images"] < 1:
        raise ValueError("max_images 必须大于0。")
    if CONFIG["imgsz"] < 1 or CONFIG["imgsz"] % 32:
        raise ValueError("imgsz 应为32的正整数倍，例如640。")
    if not 0 <= CONFIG["conf"] <= 1 or not 0 <= CONFIG["iou"] <= 1:
        raise ValueError("conf 和 iou 应在0到1之间。")
    if CONFIG["max_det"] < 1:
        raise ValueError("max_det 必须大于0。")
    if not re.fullmatch(r"cpu|[0-9]", str(CONFIG["device"])):
        raise ValueError("device 请填写单张 GPU 编号，如 '0'，或 'cpu'。")

    # 不传 batch 参数，兼容不同版本的 select_device()。
    device = select_device(str(CONFIG["device"]))

    if CONFIG["half"] and device.type != "cuda":
        raise ValueError("FP16 测试需要 CUDA；CPU 请使用 half=False。")

    torch.manual_seed(CONFIG["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG["seed"])

    # 所有对比模型使用相同设置。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)

    images, manifest = load_images(image_source)
    weight_hash = sha256_file(weights)

    detector, predictor_class = build_detector(weights)

    predict_kwargs = {
        "predictor": predictor_class,
        "imgsz": CONFIG["imgsz"],
        "device": device,
        "half": CONFIG["half"],
        "batch": 1,
        "rect": False,
        "conf": CONFIG["conf"],
        "iou": CONFIG["iou"],
        "max_det": CONFIG["max_det"],
        "agnostic_nms": False,
        "classes": None,
        "augment": False,
        "visualize": False,
        "save": False,
        "save_txt": False,
        "save_crop": False,
        "show": False,
        "verbose": False,
        "stream": False,
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run_one(image):
        # ndarray 输入：图片已经读入内存，但尚未预处理。
        results = detector.predict(source=image, **predict_kwargs)

        # 原生后处理之后，把检测框送回 CPU。
        # YOLOv8/YOLO11 的原生后处理包含 NMS；
        # RT-DETR 和无 NMS 检测头遵循各自原生流程。
        return [
            result.boxes.data.detach().cpu().numpy()
            for result in results
        ]

    print(f"模型：{CONFIG['name']}")
    print(f"权重：{weights}")
    print(f"图片数量：{len(images)}")
    print(f"Ultralytics：{ultralytics.__version__}")
    print(f"PyTorch：{torch.__version__}")
    print("计时范围：内存图片 → CPU 检测框，包含原生后处理。")

    raw_rows = []
    durations = []

    with torch.inference_mode():
        print(f"预热 {CONFIG['warmup']} 次……")
        for i in range(CONFIG["warmup"]):
            run_one(images[i % len(images)])

        synchronize()

        # 检查预测器实际使用的精度。
        actual_half = bool(detector.predictor.model.fp16)
        if actual_half != bool(CONFIG["half"]):
            raise RuntimeError("预测器实际精度与配置不一致，停止测速。")

        print(f"开始计时，共 {CONFIG['testtime']} 次……")
        for i in range(CONFIG["testtime"]):
            image_index = i % len(images)
            image = images[image_index]

            synchronize()
            start = time.perf_counter()

            boxes = run_one(image)

            synchronize()
            elapsed = time.perf_counter() - start

            durations.append(elapsed)
            raw_rows.append({
                "iteration": i + 1,
                "image_index": image_index,
                "pipeline_latency_ms": elapsed * 1000,
                "detections": sum(len(b) for b in boxes),
            })
            del boxes

    times_ms = np.asarray(durations, dtype=np.float64) * 1000
    if not np.isfinite(times_ms).all() or (times_ms <= 0).any():
        raise RuntimeError("计时结果异常，无法报告 FPS。")

    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std(ddof=0))
    fps = 1000.0 / mean_ms

    backend_model = detector.predictor.model.model
    head = (
        backend_model.model[-1]
        if hasattr(backend_model, "model") else None
    )
    end2end = getattr(head, "end2end", None)

    result = {
        "model_name": CONFIG["name"],
        "model_type": CONFIG["model_type"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "weights": str(weights),
        "weights_sha256": weight_hash,
        "measurement": "In-memory image to CPU detection boxes, batch=1",
        "included": (
            "Native preprocessing, host-to-device transfer, inference, "
            "native postprocessing, CPU box transfer, predict API overhead"
        ),
        "excluded": (
            "Weight loading, image file reading/decoding, warmup, "
            "visualization and output-file saving"
        ),
        "native_postprocessing_included": True,
        "head_class": type(head).__name__,
        "head_end2end": bool(end2end) if end2end is not None else None,
        "batch_size": 1,
        "input_height": CONFIG["imgsz"],
        "input_width": CONFIG["imgsz"],
        "precision": "FP16" if actual_half else "FP32",
        "conf": CONFIG["conf"],
        "iou": CONFIG["iou"],
        "max_det": CONFIG["max_det"],
        "unique_images": len(images),
        "warmup_requests": CONFIG["warmup"],
        "measured_requests": len(durations),
        "pipeline_latency_mean_ms": mean_ms,
        "pipeline_latency_std_ms": std_ms,
        "pipeline_latency_median_ms": float(np.median(times_ms)),
        "pipeline_latency_p95_ms": float(np.percentile(times_ms, 95)),
        "pipeline_fps": fps,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "pytorch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics_version": str(ultralytics.__version__),
        "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
        "opencv": cv2.__version__,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cpu_torch_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "fusion_policy": "Native predictor backend setup",
        "benchmark_script_sha256": sha256_file(__file__),
    }

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", CONFIG["name"]) or "model"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    precision = result["precision"].lower()
    save_dir = resolve_path(CONFIG["save_dir"]) / (
        f"{name}_{precision}_{stamp}_{uuid4().hex[:8]}"
    )
    save_dir.mkdir(parents=True, exist_ok=False)

    def write_csv(filename, rows):
        with (save_dir / filename).open(
            "x", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("summary.csv", [result])
    write_csv("latency_raw.csv", raw_rows)
    write_csv("images_manifest.csv", manifest)

    with (save_dir / "result.json").open("x", encoding="utf-8") as f:
        json.dump(
            {"result": result, "config": CONFIG},
            f, ensure_ascii=False, indent=2, allow_nan=False
        )

    shutil.copy2(
        Path(__file__).resolve(),
        save_dir / "benchmark_script.py",
    )

    summary = (
        f"模型：{CONFIG['name']}\n"
        f"精度：{result['precision']}，batch=1\n"
        f"输入：{CONFIG['imgsz']} × {CONFIG['imgsz']}\n"
        f"完整检测流程延迟：{mean_ms:.3f} ± {std_ms:.3f} ms\n"
        f"完整检测流程 FPS：{fps:.2f}\n"
        f"标准差为本次运行内的耗时波动。\n"
        f"范围：内存图片到 CPU 检测框，不含读盘、绘图和保存。\n"
        f"结果目录：{save_dir}\n"
    )
    with (save_dir / "summary.txt").open("x", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()