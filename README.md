# LCG-DETR

Research implementation for **LCG-DETR: Curvature-Guided Attention and Convolutional Gating for Lightweight Underwater Object Detection**.

**Authors:** Shaopeng Wu, Zefeng Wang, and Shaobin Cai.

LCG-DETR uses WCSPELAN, LCGAConvGLU, and SpatialSepC3 within a bundled Ultralytics RT-DETR implementation. This release includes the full detector, RT-DETR-R18 baseline, and paper component-comparison and ablation configurations.

No dataset images, annotations, trained weights, generated figures, or historical run directories are included. Train a model locally before evaluating a checkpoint.

## Installation

Use Linux or Linux under WSL2 with an NVIDIA CUDA GPU. The bundled MAC comparison module uses CUDA tensors during import; LSNet uses Triton. CPU-only execution is not supported by this release.

Run from the repository root:

```bash
conda create -n lcg-detr python=3.10 -y
conda activate lcg-detr
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-release.txt
python -m pip install -e . --no-deps --no-build-isolation
```

Use the bundled `ultralytics` package; installing or upgrading the separate PyPI package can replace the custom parser and modules. See [environment details](docs/ENVIRONMENT.md).

```bash
python - <<'PY'
from pathlib import Path
import torch
import ultralytics

assert Path(ultralytics.__file__).resolve().is_relative_to(Path.cwd().resolve())
assert torch.cuda.is_available()
from ultralytics import RTDETR
print(ultralytics.__file__)
print(torch.__version__, torch.version.cuda)
PY
```

## Data preparation

Download and prepare the data separately. See [dataset sources, labels, and preparation](docs/DATASETS.md).

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| URPC2020 | 4434 | 554 | 555 |
| DUO | 5336 | 1335 | 1111 |

These counts were verified from the retained experiment dataset snapshot. DUO's official 1111-image test membership is retained; only the official 6671-image training pool is divided into training and validation sets.

Fixed membership lists are included in [splits/](splits/README.md). Each entry is a complete path relative to the prepared dataset root. Keep the directory information because original DUO training and test images can share filenames.

Set the paths in `URPC2020/data.yaml` and `DUO/data.yaml` to your local data. The existing files retain the experiment server paths. Class IDs must follow this order:

```text
0: holothurian
1: echinus
2: scallop
3: starfish
```

## Train the full detector

A single `train.py` is used for both datasets and every paper configuration. The full LCG-DETR YAML is selected by default. Models are initialized from YAML without loading pretrained weights.

URPC2020:

```bash
python train.py --data URPC2020/data.yaml --project runs/urpc2020 --name lcg-detr
```

DUO:

```bash
python train.py --data DUO/data.yaml --project runs/duo --name lcg-detr
```

To run a baseline, component comparison, or ablation, pass its YAML from the table below. For example, the historical URPC2020 baseline used seed 3407:

```bash
python train.py \
  --model ultralytics/cfg/models/ablation/baseline.yaml \
  --data URPC2020/data.yaml \
  --seed 3407 --project runs/urpc2020 --name baseline
```

Training defaults are `imgsz=640`, `epochs=250`, `patience=20`, `batch=20`, `workers=4`, `seed=0`, deterministic computation, and FP32. AdamW is used with `lr0=1e-4`, `lrf=1.0`, beta values `(0.9, 0.999)`, and weight decay `1e-4`.

`--warmup-iters 2000` is passed as `warmup_epochs=2000` to the bundled trainer, where the value means **iterations**. Defaults enable HSV perturbation, translation, scaling, and horizontal flipping; Mosaic, MixUp, and Copy-Paste are disabled.

Use `--plots` to save training plots, or `--amp` to enable mixed precision. These options are off by default. To select another physical GPU, make it visible before starting Python, for example `CUDA_VISIBLE_DEVICES=1 python train.py ... --device 0`. The checked setup uses one visible CUDA GPU.

The full-model command uses seed 0. Historical runs do not all use the same seed, so a new seed-0 baseline should not be described as the identical historical run.

## Evaluate a locally trained checkpoint

A single `val.py` is used for both datasets. The weight path is required; the default split is `test`, with `imgsz=640`, `batch=1`, and FP32.

URPC2020:

```bash
python val.py \
  --weights runs/urpc2020/lcg-detr/weights/best.pt \
  --data URPC2020/data.yaml \
  --project runs/evaluation --name urpc2020-lcg-detr
```

DUO:

```bash
python val.py \
  --weights runs/duo/lcg-detr/weights/best.pt \
  --data DUO/data.yaml \
  --project runs/evaluation --name duo-lcg-detr
```

The script prints aggregate and class-wise detection metrics and validator timings. It writes `paper_data.txt` and `metrics.json` in the validation output directory. Tables display percentages; JSON accuracy values remain on the 0-1 scale. Use `--json /path/to/summary.json` to select another summary path.

Use `--profile` to also estimate GFLOPs. Profiling failures are recorded separately and do not discard the accuracy results. Use `--plots` to save PR curves and other validator plots. `--half` enables FP16; FP32 remains the default for paper comparisons.

Validation fitness is `0.1 * mAP50 + 0.9 * mAP50:95` for RT-DETR checkpoint selection. Precision, recall, mAP50, and mAP50:95 are read together from one invocation of `model.val()` on the supplied checkpoint. Do not combine independent validation-metric maxima from different epochs.

Validator FPS is calculated as `1000 / (preprocess_ms + inference_ms + postprocess_ms)`. These timings exclude image-file loading and output writing; they are different from full `predict()` interface latency. State batch size, numerical precision, and device when reporting speed.

Check all options with:

```bash
python train.py --help
python val.py --help
```

## Inference

Replace the checkpoint and image paths with your own files.

```bash
python - <<'PY'
from ultralytics import RTDETR

model = RTDETR("runs/urpc2020/lcg-detr/weights/best.pt")
results = model.predict(
    source="/absolute/path/to/image.jpg",
    imgsz=640, device=0, half=False, save=False,
)
print(results[0].boxes)
PY
```

## Paper configurations

All paths are relative to `ultralytics/cfg/models/`. Component comparisons and ablations are performed on URPC2020; the baseline and full detector are also compared on DUO.

| Group | Model | YAML |
|---|---|---|
| Overall ablation / baseline | R18 + AIFI + RepC3 | `ablation/baseline.yaml` |
| Overall ablation / backbone comparison | WCSPELAN + AIFI + RepC3 | `ablation/DKCSPELAN_AIFI_RepC3.yaml` |
| Overall ablation / encoder comparison | R18 + LCGAConvGLU + RepC3 | `ablation/ResNet18_LCGAConvGLU_RepC3.yaml` |
| Overall ablation | R18 + AIFI + SpatialSepC3 | `ablation/ResNet18_AIFI_SpatialSepC3.yaml` |
| Overall ablation | WCSPELAN + LCGAConvGLU + RepC3 | `ablation/DKCSPELAN_LCGAConvGLU_RepC3.yaml` |
| Overall / internal ablation reference | WCSPELAN + AIFI + SpatialSepC3 | `ablation/DKCSPELAN_AIFI_SpatialSepC3.yaml` |
| Overall ablation | R18 + LCGAConvGLU + SpatialSepC3 | `ablation/ResNet18_LCGAConvGLU_SpatialSepC3.yaml` |
| Full LCG-DETR / overall / internal ablation | WCSPELAN + LCGAConvGLU + SpatialSepC3 | `ablation/DKCSPELAN_LCGAConvGLU_SpatialSepC3.yaml` |
| Internal ablation | WCSPELAN + LCGA-FFN + SpatialSepC3 | `lcga_convglu_ablation/DKCSPELAN_LCGA_SpatialSepC3.yaml` |
| Internal ablation | WCSPELAN + AIFI-ConvGLU + SpatialSepC3 | `lcga_convglu_ablation/DKCSPELAN_AIFI_ConvGLU_SpatialSepC3.yaml` |
| Backbone comparison | FAT | `rt-detr/rtdetr-C2f-FAT.yaml` |
| Backbone comparison | LGLB | `rt-detr/rtdetr-C2f-LGLB.yaml` |
| Backbone comparison | MAC | `rt-detr/rtdetr-C2f-MAC.yaml` |
| Backbone comparison | LSNet-T | `rt-detr/rtdetr-lsnet.yaml` |
| Encoder comparison | DPB | `attention_comparison/R18_DPB_RepC3.yaml` |
| Encoder comparison | TSSA | `attention_comparison/R18_TSSA_RepC3.yaml` |
| Encoder comparison | MSMHSA | `attention_comparison/R18_MSMHSA_RepC3.yaml` |
| Encoder comparison | DHSA | `attention_comparison/R18_DHSA_RepC3.yaml` |

Historical filenames and the `DKCSPELAN` alias are retained for configuration/checkpoint compatibility. The corresponding backbone is called **WCSPELAN** in the paper.

YOLOv8/YOLO11/YOLO26 were trained in separate implementations, which are not included here. Literature-reported detectors are also not bundled.

## Complexity and timing

- `get_all_yaml_param_and_flops.py` estimates complexity for its configured folder. Its default is `ultralytics/cfg/models/rt-detr`; it does not recursively scan all 18 models.
- `get_FPS.py` benchmarks a local checkpoint and image source. Fill its `CONFIG` paths before use. Its full `predict()` timing includes interface overhead and differs from the validator's recorded preprocessing/inference/postprocessing time.
- State input resolution, batch size, precision, device, and fusion status when comparing results. Raw and fused parameter counts differ.

## Validation scope

Before weights were removed from this source copy, 24 paper-related `best.pt` files covering all 18 configurations and the DUO baseline/full detector passed loading, a 640 x 640 CUDA FP32 forward pass, and the standard `predict()` interface. Identical contents were deduplicated by SHA256.

Those checks used the existing experiment environment. They were not full retraining, a clean-environment installation, or a new dataset-accuracy evaluation.

## License and sources

See [LICENSE](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md), and [source acknowledgements](docs/SOURCES.md). Upstream license terms and notices apply to the corresponding code. Dataset rights remain with their providers.

Source-code permissions for FTIC, LSNet, CMTFNet, and Histoformer remain unconfirmed. See the notices before publicly redistributing those comparison implementations; the root license does not supply missing upstream permission.

## Citation and contact

Please cite the manuscript title and authors given above. No publication venue or DOI is assigned in this release. See [CITATION.cff](CITATION.cff).

Implementation contact: Shaopeng Wu, `yusi88105@163.com`.

