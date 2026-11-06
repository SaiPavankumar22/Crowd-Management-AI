"""PET (Point quEry Transformer), vendored for inference.

Liu, Lu, Cao and Liu, "Point-Query Quadtree for Crowd Counting, Localization,
and More", ICCV 2023. Code: https://github.com/cxliu0/PET (commit 5b4dd7d),
MIT licence (see LICENSE); the authors ask that it be used for academic purposes.

Changes from upstream, all for running on current PyTorch and on CPU:
- imports made relative to this package
- the VGG backbone is built without ImageNet weights, which come from the checkpoint
- prediction moves query points to the model's device instead of forcing CUDA
- a CPU/GPU index mismatch in inference-time query selection is fixed
"""
