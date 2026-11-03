# LCG-DETR 源码说明

本仓库对应论文 *LCG-DETR: Curvature-Guided Attention and Convolutional Gating for Lightweight Underwater Object Detection*，作者为吴绍鹏、王泽峰、蔡绍滨。

完整安装、训练、评估、推理命令和18份模型配置见[英文 README](README.md)。

## 发布范围

提供完整模型、RT-DETR-R18基线、组件对比和消融配置，以及所需框架代码。数据图像、标注、权重、热力图和运行结果不随源码发布。使用者需要自行获取数据并训练，再加载自己的best.pt评估。

## 环境和数据

- [环境与依赖](docs/ENVIRONMENT.md)：Linux/WSL2、CUDA GPU、安装顺序及版本。
- [数据说明](docs/DATASETS.md)：获取来源、类别顺序、路径配置、DUO转换和测试集保留。
- [数据划分清单](splits/README.md)：只含相对文件路径，不含数据图像或标注。

核验的项目快照中，URPC2020划分为4434/554/555张，DUO划分为5336/1335/1111张，顺序均为训练/验证/测试。DUO保留官方1111张测试图像。

DUO原始训练与测试文件可能同名，必须保留目录限定路径。现有数据YAML与convert_duo.py保留原服务器路径，使用时按数据说明填写自己的路径。

## 实验入口

训练和验证分别统一为 train.py 与 val.py，不再提供数据集专用入口。

URPC2020训练：

~~~bash
python train.py --data URPC2020/data.yaml --project runs/urpc2020 --name lcg-detr
~~~

DUO训练：

~~~bash
python train.py --data DUO/data.yaml --project runs/duo --name lcg-detr
~~~

验证本地训练权重：

~~~bash
python val.py --weights runs/urpc2020/lcg-detr/weights/best.pt --data URPC2020/data.yaml
~~~

使用 --model 选择其他论文YAML，--seed 设置随机种子。历史URPC2020基线使用seed 3407，完整模型使用seed 0。其余参数和DUO验证命令见英文README。

验证默认使用测试集、batch 1、FP32，同一次评估汇总P、R和两项mAP，结果保存为paper_data.txt和metrics.json。--profile 可估算GFLOPs，--plots 可输出PR等验证图。JSON精度指标取值为0至1，表格使用百分数。

训练保留AdamW、250轮上限、batch 20、早停20轮及2000次迭代预热。统一入口的参数不会自动修改模型计算代码。

## 来源与许可

见[LICENSE](LICENSE)、[第三方声明](THIRD_PARTY_NOTICES.md)和[来源说明](docs/SOURCES.md)。各上游贡献的版权与许可声明应保留。

FTIC、LSNet、CMTFNet、Histoformer 的源码许可尚未查明，已在第三方声明中标出，公开分发这些对比实现前需核实。顶层许可证不替代缺失的上游授权。

