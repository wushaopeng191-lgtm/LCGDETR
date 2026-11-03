# Third-party notices

LCG-DETR is developed from Ultralytics 8.0.201. Existing Ultralytics copyright,
author, and license statements remain applicable. The complete GNU Affero
General Public License, version 3, is reproduced in the root `LICENSE` from
[the Ultralytics v8.0.201 release](https://github.com/ultralytics/ultralytics/blob/v8.0.201/LICENSE).
The upstream `setup.py` identifies AGPL-3.0 and its package classifier identifies
AGPLv3 or later. This repository does not replace the upstream license with MIT.

Third-party components retain their own applicable terms and notices. The root
license does not supply missing permissions for a separate upstream component.
The following inventory covers the implementations used by the retained paper
configurations. A paper citation identifies a method; it does not itself grant
permission to redistribute that method's source code.

## Verified upstream license statements

| Component | Retained implementation | Upstream project | Verified terms and included text |
| --- | --- | --- | --- |
| Ultralytics / RT-DETR implementation | `ultralytics/`, training and evaluation framework | [Ultralytics v8.0.201](https://github.com/ultralytics/ultralytics/tree/v8.0.201) | AGPL-3.0; complete text in [`LICENSE`](LICENSE) |
| CGA / local curvature-guided attention | `ultralytics/nn/extra_modules/LCGA.py` | [mwaleedaslam/CGA](https://github.com/mwaleedaslam/CGA) | Apache-2.0; [`LICENSES/CGA-LICENSE.txt`](LICENSES/CGA-LICENSE.txt) |
| TransNeXt / convolutional gated linear unit | `ConvolutionalGLU` in `ultralytics/nn/extra_modules/block.py`; encoder integration in `transformer.py` | [DaiShiResearch/TransNeXt](https://github.com/DaiShiResearch/TransNeXt) | Apache-2.0; [`LICENSES/TransNeXt-LICENSE.txt`](LICENSES/TransNeXt-LICENSE.txt) |
| Mobile U-ViT / LGL block | `ultralytics/nn/extra_modules/MobileUViT.py`; `C2f_LGLB` in `block.py` | [FengheTan9/Mobile-U-ViT](https://github.com/FengheTan9/Mobile-U-ViT) | Apache-2.0; [`LICENSES/Mobile-U-ViT-LICENSE.txt`](LICENSES/Mobile-U-ViT-LICENSE.txt) |
| HDNet / MAC | `ultralytics/nn/extra_modules/MAC.py`; `C2f_MAC` in `block.py` | [iLearn-Lab/TGRS25-HDNet](https://github.com/iLearn-Lab/TGRS25-HDNet) | Official README declares Apache-2.0. A standalone upstream LICENSE was not found. The unmodified standard text from the Apache Software Foundation is in [`LICENSES/HDNet-Apache-2.0.txt`](LICENSES/HDNet-Apache-2.0.txt). |
| CrossFormer / dynamic position bias | `DynamicPosBias` and DPB encoder in `ultralytics/nn/extra_modules/transformer.py`; position-bias helper in `LCGA.py` | [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer) | MIT; Copyright (c) 2021 cheerss. Complete notice in [`LICENSES/CrossFormer-LICENSE.txt`](LICENSES/CrossFormer-LICENSE.txt). |
| ToST / token statistics self-attention | `AttentionTSSA` in `ultralytics/nn/extra_modules/attention.py`; TSSA encoder in `transformer.py` | [RobinWu218/ToST](https://github.com/RobinWu218/ToST) | MIT; Copyright (c) 2024 Ziyang Wu. Complete notice in [`LICENSES/ToST-LICENSE.txt`](LICENSES/ToST-LICENSE.txt). |
| Swin Transformer helpers acknowledged by FTIC | Window-attention implementation in `ultralytics/nn/extra_modules/filc.py` | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | MIT; Copyright (c) Microsoft Corporation. Complete notice in [`LICENSES/Swin-Transformer-LICENSE.txt`](LICENSES/Swin-Transformer-LICENSE.txt). This does not establish a license for the FTIC-specific additions. |

The full license files above are preserved without replacing upstream copyright
holders with the LCG-DETR authors. Adapted components have been integrated into
the detector and wrapped or configured for the paper experiments. They are not
claimed as entirely original LCG-DETR implementations. See
[`docs/SOURCES.md`](docs/SOURCES.md) for method references and scope.

## Upstream permissions requiring confirmation before public release

| Component | Retained implementation | Verified source identity | Current finding |
| --- | --- | --- | --- |
| FTIC / FAT | `ultralytics/nn/extra_modules/filc.py`; `C2f_FAT` in `block.py` | [qingshi9974/ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC) | The official repository and `models/flic.py` were inspected. No repository license or file-level license for the FTIC-specific implementation was located. Permission remains unverified. |
| LSNet-T | `ultralytics/nn/backbone/lsnet.py` | [THU-MIG/lsnet](https://github.com/THU-MIG/lsnet) (the paper's `jameslahm/lsnet` link redirects here) | No repository license or file-level license in `model/lsnet.py` was located. Permission remains unverified, including the retained SKA/Triton implementation. |
| CMTFNet / MSMHSA | `LayerNorm`, `MutilScal`, `Mutilscal_MHSA`, and MSMHSA encoder in `ultralytics/nn/extra_modules/transformer.py` | [DrWuHonglin/CMTFNet](https://github.com/DrWuHonglin/CMTFNet) | An official implementation was identified, but no public root license was located. A source-specific permission check remains necessary. |
| Histoformer / DHSA | `Attention_histogram` and DHSA encoder in `ultralytics/nn/extra_modules/transformer.py` | [sunshangquan/Histoformer](https://github.com/sunshangquan/Histoformer) | The official repository and `basicsr/models/archs/histoformer_arch.py` were inspected. No public root or file-level license for this implementation was located. A model-hosting metadata label is not treated as a source-code license grant. |

These entries are retained to support all paper comparisons and ablations; this
document records unresolved provenance rather than silently assigning a license.
The originating revision and any separate permission held by the authors should
be recorded before public redistribution of those components. Public repository
visibility and academic citation are not substitutes for that confirmation.

## External dependencies and datasets

Packages installed from PyPI, Conda, or PyTorch distribution channels retain their
own licenses. Their licenses are not replaced by this repository's `LICENSE`.
Dataset files and trained weights are not included. Dataset use remains subject
to the terms established by the original dataset providers.

Verification date: 2026-11-02. This is a source-attribution inventory, not a claim
that every historical upstream revision has been identified or cleared.
