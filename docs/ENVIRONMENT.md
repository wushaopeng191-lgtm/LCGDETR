# Environment and dependencies

## Supported setup

Use Python 3.10 on Linux, or Linux under WSL2, with an NVIDIA GPU and a CUDA-compatible driver. The release was checked in the existing experiment environment with PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, and Triton 2.2.0. Use CUDA FP32 for the supplied paper comparisons.

The complete package requires a CUDA GPU even when the selected model does not use MAC. `ultralytics.nn.tasks` imports the retained custom modules, and `extra_modules/MAC.py` creates CUDA tensors during import. CPU-only and native Windows execution are not supported by this release. macOS, other GPU backends, and multi-GPU training were not validated.

## Installation

From the repository root, create a separate environment and install the CUDA PyTorch wheels before the remaining dependencies:

```bash
conda create -n lcg-detr python=3.10 -y
conda activate lcg-detr
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-release.txt
python -m pip install -e . --no-deps --no-build-isolation
```

`requirements-release.txt` pins the main numerical/model packages and bounds the utility dependencies. It is a focused release dependency list, not a freeze of the experiment server. Its version ranges do not mean that every allowed combination has been tested. A clean-environment installation from this list has not yet been tested.

The bundled `setup.py` keeps the upstream package name `ultralytics` and reads the original `requirements.txt`. The original file does not list all custom-module dependencies and requests the GUI OpenCV distribution. The installation sequence above installs the release dependencies explicitly, then registers the local source without resolving that older dependency list again. `--no-build-isolation` uses the installed setuptools and wheel packages.

Do not install or upgrade the separate PyPI `ultralytics` package over this source. Its parser and custom modules differ from this repository. Run commands from the repository root and verify the imported file path below.

## OpenCV

Use only `opencv-python-headless==4.11.0.86`, which was found in the experiment environment and provides `cv2` version 4.11.0. Do not install `opencv-python`, `opencv-contrib-python`, or another OpenCV wheel alongside it; these distributions share the same `cv2` namespace.

The headless build supports image reading, writing, training, evaluation, and saved visualizations. It does not provide GUI display through `cv2.imshow`; keep `show=False`.

Because the unchanged setup metadata still declares `opencv-python`, `pip check` can report that distribution as missing even when the required `cv2` functionality is provided by the headless build. Installing both distributions is not the recommended fix. The CUDA/import checks and the actual model command are the relevant runtime checks for this source snapshot.

## Verify the installation

Run this in the activated environment from the repository root:

```bash
python - <<'PY'
from pathlib import Path
import torch

assert torch.cuda.is_available(), "An NVIDIA CUDA GPU is required."
assert torch.__version__.split("+")[0] == "2.2.2", torch.__version__
torch.cuda.set_device(0)

import cv2
import triton
import ultralytics
from ultralytics import RTDETR

source = Path(ultralytics.__file__).resolve()
assert source.is_relative_to(Path.cwd().resolve()), source
print("source:", source)
print("torch/CUDA:", torch.__version__, torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("OpenCV:", cv2.__version__)
print("Triton:", triton.__version__)
PY
```

If another physical GPU must be used, select it before Python starts, for example `CUDA_VISIBLE_DEVICES=1 python ...`; it will be exposed to the process as CUDA device 0. The verified checks used CUDA device 0.

## Model-specific constraints

- **LSNet-T:** the spatial kernel aggregation implementation uses Triton kernels. Its broad import guards do not implement a CPU fallback. A successful Python import alone does not establish that its kernels can run. Triton 2.2.0 and CUDA FP32 were used for verification; keep the paired PyTorch version above.
- **LSNet-T evaluation:** its attention module caches a bias tensor when evaluation mode is entered. When using the underlying PyTorch model directly, move it to the final device first and then call `eval()`, for example `model.float().to("cuda:0").eval()`. Calling `eval()` on CPU before moving to CUDA can leave the cache on CPU and cause a device mismatch. The standard `RTDETR.predict(..., device=0, half=False)` interface was also checked.
- **MAC:** the module allocates fixed CUDA kernels during import. Make the desired GPU visible before importing `ultralytics`. CUDA FP32 was checked; CPU fallback is not provided.
- **FAT, LGLB, and LSNet-T:** the retained code imports `timm.layers`, so timm is a runtime dependency, including during import of the bundled model registry.
- **Attention and FAT modules:** einops is imported for tensor rearrangement.
- **RT-DETR training:** SciPy supplies the Hungarian assignment operation. torchvision supplies framework transforms and supporting operations.
- **Reporting:** PrettyTable is used by the supplied validation scripts; THOP is used for complexity reporting. `py-cpuinfo` supports framework hardware reporting and is included even though its distribution metadata was absent from the inspected GPU environment.

The paper configurations do not require installing MMCV, MMDetection, a separate TransNeXt CUDA extension, or an entire third-party model repository. Unused backbone files are not instructions to install every upstream dependency.

## Inspected versions and validation limits

These versions were read from installed distribution metadata in the existing experiment environment. They describe that environment; they are not evidence of a new installation test.

| Package | Inspected version |
|---|---|
| torch | 2.2.2+cu121 |
| torchvision | 0.17.2+cu121 |
| triton | 2.2.0 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.11.0.86 |
| timm | 1.0.7 |
| einops | 0.8.2 |
| Pillow | 12.2.0 |
| matplotlib | 3.10.9 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| tqdm | 4.67.3 |
| psutil | 7.2.2 |
| dill | 0.3.8 |
| prettytable | 3.17.0 |
| thop | 0.1.1-2209072238 |
| setuptools | 69.5.1 |

The THOP requirement uses the normalized package version `0.1.1.post2209072238`. `py-cpuinfo` and wheel are included as installation/runtime recommendations; their installed versions were not established by this metadata inspection.

Before trained weights were removed from the source copy, 24 paper-related `best.pt` paths covering 18 configurations and the DUO baseline/full detector passed checkpoint loading, a 640 x 640 CUDA FP32 forward pass, and the standard `predict()` interface. Identical checkpoint contents were deduplicated by SHA256. These checks did not reinstall dependencies, retrain the models, run a new dataset-accuracy evaluation, or establish support for every device, precision, or export backend.

TensorBoard, experiment-tracking integrations, Albumentations, COCO JSON evaluation packages, and export toolchains are not needed for the documented detector commands. Install an optional package only when enabling the corresponding feature. Their installation and export compatibility were not checked as part of this release.
