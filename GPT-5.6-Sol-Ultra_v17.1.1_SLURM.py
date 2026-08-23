#!/usr/bin/env python3
"""Versioned compatibility launcher for the packaged volume TTA pipeline.

The implementation lives in :mod:`volume_tta`; the launcher filename remains versioned for
existing SLURM submission scripts.
"""

from volume_tta.__main__ import run


if __name__ == "__main__":
    run()
