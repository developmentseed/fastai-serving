# Benchmarks to optimize prediction speed

CPU:
- ~0.17 per image by iterating over images
- ~0.12 per image batch prediction over tensors

GPU:
- ~0.018 per image by iterating over images
- ~0.0055 per image batch prediction over tensors
