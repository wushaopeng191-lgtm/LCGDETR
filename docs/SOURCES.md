# Method and implementation sources

The retained code includes the full LCG-DETR model, its component ablations, and
the backbone and encoder comparisons in the manuscript. The following table
records the corresponding upstream methods. These are detector adaptations;
the comparison YAML files are not releases of the upstream authors' complete
compression, segmentation, restoration, or classification systems.

| Method | Reference | Upstream source | Role in this repository |
| --- | --- | --- | --- |
| Ultralytics / RT-DETR | Ultralytics 8.0.201; RT-DETR, Zhao et al., 2024 | [Ultralytics v8.0.201](https://github.com/ultralytics/ultralytics/tree/v8.0.201) | Base framework, RT-DETR detection pipeline, standard layers, training, and evaluation. |
| CGA | Aslam et al., *CGA: Curvature-Guided Attention for Remote Sensing Image Super-Resolution*, IEEE TGRS, 2026, [DOI](https://doi.org/10.1109/TGRS.2026.3661098) | [mwaleedaslam/CGA](https://github.com/mwaleedaslam/CGA) | LCGA attention used in the LCGA-only and LCGAConvGLU encoders. |
| TransNeXt | Shi, *TransNeXt: Robust Foveal Visual Perception for Vision Transformers*, CVPR, 2024 | [DaiShiResearch/TransNeXt](https://github.com/DaiShiResearch/TransNeXt) | Convolutional GLU mechanism; adapted hidden width and residual configuration in the detection encoder. The active class is `block.py:ConvolutionalGLU`. |
| FTIC | Li et al., *Frequency-Aware Transformer for Learned Image Compression*, ICLR, 2024 | [qingshi9974/ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC) | FAT backbone comparison, through `filc.py:FAT_Block` and `block.py:C2f_FAT`. |
| Mobile U-ViT | Tang et al., *Mobile U-ViT: Revisiting Large Kernel and U-Shaped ViT for Efficient Medical Image Segmentation*, ACM MM, 2025, [DOI](https://doi.org/10.1145/3746027.3755076) | [FengheTan9/Mobile-U-ViT](https://github.com/FengheTan9/Mobile-U-ViT) | LGLB backbone comparison, through `MobileUViT.py:LGLBlock` and `block.py:C2f_LGLB`. |
| HDNet | Xu et al., *HDNet: A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection*, IEEE TGRS, 2025, [DOI](https://doi.org/10.1109/TGRS.2025.3574962) | [iLearn-Lab/TGRS25-HDNet](https://github.com/iLearn-Lab/TGRS25-HDNet) | MAC backbone comparison, through `MAC.py:MAC` and `block.py:C2f_MAC`. The paper's `xumingzhu989/HDNet-TGRS` URL redirects to this repository. |
| LSNet | Wang et al., *LSNet: See Large, Focus Small*, CVPR, 2025 | [THU-MIG/lsnet](https://github.com/THU-MIG/lsnet) | LSNet-T backbone comparison via `backbone/lsnet.py:lsnet_t`, including SKA and Triton kernels. The paper's `jameslahm/lsnet` URL redirects here. |
| CrossFormer | Wang et al., *CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention*, ICLR, 2022 | [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer) | Dynamic position bias in the DPB encoder comparison and related local attention helpers. |
| ToST | Wu et al., *Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction*, ICLR, 2025 | [RobinWu218/ToST](https://github.com/RobinWu218/ToST) | TSSA encoder comparison through `attention.py:AttentionTSSA`. |
| CMTFNet | Wu et al., *CMTFNet: CNN and Multiscale Transformer Fusion Network for Remote-Sensing Image Semantic Segmentation*, IEEE TGRS, 2023 | [DrWuHonglin/CMTFNet](https://github.com/DrWuHonglin/CMTFNet) | Adapted multiscale multihead self-attention in the MSMHSA encoder comparison. |
| Histoformer | Sun et al., *Restoring Images in Adverse Weather Conditions via Histogram Transformer*, ECCV, 2024 | [sunshangquan/Histoformer](https://github.com/sunshangquan/Histoformer) | Histogram self-attention in the DHSA encoder comparison. |
| Swin Transformer | Liu et al., *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*, ICCV, 2021 | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | FTIC's source explicitly acknowledges Swin window-attention implementation; the same acknowledgment is retained in `filc.py`. |

`WCSPELAN`, `SpatialSepC3`, the integration of LCGA with reduced-width ConvGLU,
and the experiment-specific detector configurations are described in the
LCG-DETR manuscript. The legacy `DKCSPELAN` name is retained as a compatibility
alias in code and some YAML filenames. It refers to the manuscript's WCSPELAN
configuration and is not a separate paper method.

## Attribution scope and unresolved revisions

The table identifies method correspondence and official upstream sources. It
does not establish that every retained line was copied directly from the
current upstream branch. The local source does not record original commit
identifiers for all adapted components. The current public upstream license
statements were checked on 2026-11-02; exact license download URLs and SHA-256
hashes are recorded in [`../LICENSES/manifest.json`](../LICENSES/manifest.json).

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for complete license
text locations and the explicit unresolved permissions for FTIC, LSNet,
CMTFNet, and Histoformer. No license is inferred from the existence of a paper,
a citation, a public repository, or a model-hosting license tag.
