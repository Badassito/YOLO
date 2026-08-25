"""Large-volume YOLO segmentation with test-time augmentation.

The package initializer is deliberately inert so spawned worker processes do not
initialize inference runtimes until their backend entry point requests them.
"""

__version__ = "17.1.4"

__all__ = ("__version__",)
