# Dataset preparation

The repository provides dataset configuration examples and image-membership lists. Images, annotations, and model weights must be obtained separately.

## Sources and dataset versions

- **URPC2020:** [URPC website](http://www.urpc.org.cn/) and the [OpenI URPC2020 dataset index](https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC2020_dataset/datasets).
- **DUO:** [upstream DUO repository](https://github.com/chongweiliu/DUO).

The URPC link on OpenI is an index/mirror. Different uploads can contain different files or annotation versions. Match the downloaded distribution against the supplied membership lists rather than assuming that every mirror is identical. Follow the data provider's terms when obtaining and using either dataset.

## Experimental split

The inspected dataset snapshot contains the following images:

| Dataset | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| URPC2020 | 4,434 | 554 | 555 | 5,543 |
| DUO | 5,336 | 1,335 | 1,111 | 7,782 |

For URPC2020, the supplied lists define the exact membership. The historical split algorithm and random seed are not established by the available records, so no seed or procedure is proposed for regenerating this split. The validation directory is named **`valid`**, while its membership-list filename is `val.txt`.

For DUO, the official training pool contains 6,671 images and the fixed official test set contains 1,111 images. The training pool is divided into 5,336 training images and 1,335 validation images. The official test set is retained separately and is not repartitioned. In the inspected snapshot, the converted training/validation filename union matches the official training filename set, and the converted test filenames match the official test filename set. The corresponding annotation-JSON checksums are recorded in `splits/manifest.json`.

Use the supplied membership lists for exact reproduction. The original converter uses `random.seed(0)`, shuffles the valid training images, and takes `int(N * 0.8)` images for training. Its result also depends on the order of images in the source JSON and on which images pass validation; the seed alone does not identify an exact split.

## Membership lists

The six lists are located at:

```text
splits/URPC2020/train.txt
splits/URPC2020/val.txt
splits/URPC2020/test.txt
splits/DUO/train.txt
splits/DUO/val.txt
splits/DUO/test.txt
```

Each entry is relative to the corresponding **prepared dataset root**, for example `valid/images/000010.jpg` under `URPC2020`, or `test/images/1.jpg` under `DUO`. Preserve the complete path. DUO's official training and test folders reuse some basenames, so `train/images/1.jpg` and `test/images/1.jpg` are separate directory-qualified entries. Do not merge or deduplicate these lists using only basenames.

The lists contain unique entries within each subset. Full relative-path intersections between the listed subsets are empty. These are membership checks, not image-content deduplication: no claim of content-hash deduplication is made.

The lists are membership records, not directly interchangeable with the directory paths in a training YAML. Arrange the files as below and point the data configuration to the image directories. See the [split-list documentation](../splits/README.md) for checksums and path conventions.

## Class order and label format

Prepared YOLO labels use this class order:

| Class | DUO COCO `category_id` | YOLO class ID |
|---|---:|---:|
| holothurian | 1 | 0 |
| echinus | 2 | 1 |
| scallop | 3 | 2 |
| starfish | 4 | 3 |

`convert_duo.py` applies `category_id - 1` to the DUO COCO annotations. Labels that are already in the prepared YOLO format use the zero-based IDs in the last column.

Each YOLO label line has the form:

```text
class_id center_x center_y width height
```

The four box coordinates are normalized by image width and height. Label files have the same basename as the corresponding images and use the `.txt` extension. Prepare matching YOLO annotations for the selected URPC2020 distribution; this release does not specify an unverified URPC conversion or random-splitting algorithm.

## Original DUO layout and conversion

The existing converter expects this upstream layout:

```text
DUO_raw/
├── annotations/
│   ├── instances_train.json
│   └── instances_test.json
└── images/
    ├── train/
    └── test/
```

Before running `python convert_duo.py`, edit **`BASE`** to the actual raw `DUO_raw` directory and **`OUT`** to the prepared `DUO` output directory. Keep the raw input and prepared output directories separate. The preserved script contains machine-specific absolute paths, so these settings must be checked first.

The converter verifies image readability, clips boxes to image boundaries, skips nonpositive boxes, and writes normalized YOLO labels. Corrupt images are skipped. It partitions only the valid official training images; the official test JSON is converted independently. Use the release membership lists to retain the exact experimental split, especially if the input JSON order or dataset distribution differs from the inspected version.

## Prepared directory layout

```text
URPC2020/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

DUO/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Data configuration examples

Update `URPC2020/data.yaml` for the prepared URPC2020 root:

```yaml
path: /absolute/path/to/URPC2020
train: train/images
val: valid/images
test: test/images
nc: 4
names: [holothurian, echinus, scallop, starfish]
```

Update `DUO/data.yaml` for the prepared DUO root:

```yaml
path: /absolute/path/to/DUO
train: train/images
val: val/images
test: test/images
nc: 4
names: [holothurian, echinus, scallop, starfish]
```

These examples change only the location of the prepared datasets. Keep the published train/validation/test membership and class order when reproducing the experiments.
