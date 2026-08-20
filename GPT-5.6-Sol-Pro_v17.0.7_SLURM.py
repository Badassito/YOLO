#!/usr/bin/env python3
"""Compatibility launcher for the packaged volume TTA pipeline.

The implementation now lives in :mod:`volume_tta`. Existing SLURM commands may keep
calling this filename unchanged.
"""

from volume_tta.__main__ import run


if __name__ == "__main__":
    run()
