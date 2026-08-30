"""Large-volume YOLO segmentation pretraining and test-time augmentation.

The package initializer is deliberately inert so spawned worker processes do not
initialize inference runtimes until their backend entry point requests them.
"""

__version__ = "18.0.0"

__all__ = ("__version__",)
