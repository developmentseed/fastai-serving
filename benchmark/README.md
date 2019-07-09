# Benchmarks to optimize prediction speed

`fastai` verison: `1.0.55.dev0`

## Classification
CPU:

| Method | ~ sec/image |
| --- | --- |
| `.predict` iteration | 0.24 |
| `.pred_batch` | 0.17 |
| `.pred_batch` + `apply_tfms` | 0.18 |


GPU (`p2.xlarge`):

| Method | ~ sec/image |
| --- | --- |
| `.predict` iteration | 0.019 |
| `.pred_batch` | 0.0061 |
| `.pred_batch` + `apply_tfms` | 0.0073 |

GPU (`p3.2xlarge`):

| Method | ~ sec/image |
| --- | --- |
| `.predict` iteration | 0.013 |
| `.pred_batch` | 0.0034 |
| `.pred_batch` + `apply_tfms` | 0.0038 |

## Segmentation
CPU:

| Method | ~ sec/image |
| --- | --- |
| `.predict` iteration | 8.3 |
| `.pred_batch` | 7.7 |
| `.pred_batch` + `apply_tfms` | 8.2 |

GPU (`p2.xlarge`):

| Method | ~ sec/image |
| --- | --- |
| `.predict` iteration | 0.45 |
| `.pred_batch` | 0.29 |
| `.pred_batch` + `apply_tfms` | 0.29 |

GPU (`p3.2xlarge`):

| Method | ~ sec/image |
| --- | --- |
| `.predict` iteration | 0.048 |
| `.pred_batch` | 0.032 |
| `.pred_batch` + `apply_tfms` | 0.033 |
