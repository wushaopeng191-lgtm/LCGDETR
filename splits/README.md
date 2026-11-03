# Experimental split lists

These files record image membership only. Images, annotations, and weights are not included.

| Dataset directory | `train.txt` | `val.txt` | `test.txt` |
|---|---:|---:|---:|
| `URPC2020` | 4,434 | 554 | 555 |
| `DUO` | 5,336 | 1,335 | 1,111 |

Each line is relative to the prepared dataset root. Preserve the complete path: URPC validation entries start with `valid/images/`, although the list is named `val.txt`. DUO validation entries start with `val/images/`.

DUO's official training and test folders reuse basenames. Keep directory-qualified identities such as `train/images/1.jpg` and `test/images/1.jpg`; basename-only merging or deduplication would change the split. The fixed official DUO test membership is retained.

The lists are authoritative for this experimental snapshot. No historical URPC split seed is asserted. DUO conversion uses seed 0 and an 80:20 division of the official training pool, but JSON order and image validity can affect a newly generated split.

Counts, list SHA-256 checksums, path conventions, and available DUO annotation checksums are in [manifest.json](manifest.json). Membership checks do not establish image-content deduplication.

For data sources, label mappings, conversion settings, and YAML path examples, see [Dataset preparation](../docs/DATASETS.md). These lists are membership references; use the prepared image-directory paths in the training data YAML.
