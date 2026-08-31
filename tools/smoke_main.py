"""Exercise the packaged orchestration entry until its first deterministic I/O guard."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from smoke_import import ROOT, install_stubs


def main() -> None:
    install_stubs()
    os.environ["YOLO_TTA_TELEMETRY"] = "0"
    os.environ.pop("YOLO_TTA_CAPTURE_STDIO_PATH", None)
    missing = ROOT / "__XTA_missing_input__.mkv"
    if missing.exists():
        raise RuntimeError(f"reserved smoke-test path unexpectedly exists: {missing}")
    sys.argv = [
        "xta",
        "--input",
        str(missing),
        "--model",
        "gpu:unused.engine",
        "--device",
        "0",
    ]
    from XTA.pipeline import main as pipeline_main

    try:
        pipeline_main()
    except FileNotFoundError as exc:
        reported = Path(exc.filename) if exc.filename else Path(str(exc))
        if reported != missing:
            raise
        print("pipeline reached the expected missing-input guard")
        return
    raise RuntimeError("pipeline did not reject the deliberately missing input")


if __name__ == "__main__":
    main()
