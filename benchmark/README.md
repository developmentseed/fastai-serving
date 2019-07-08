# Benchmarks to optimize prediction speed

`fastai` verison: `1.0.55.dev0`

## Classification
CPU:
- ~0.24 per image by iterating over images
- ~0.17 per image batch prediction over tensors
- ~0.18 per image batch prediction over tensors + transformations

GPU:
- ~0.018 per image by iterating over images
- ~0.0055 per image batch prediction over tensors

## Segmentation
CPU:
-

GPU:
