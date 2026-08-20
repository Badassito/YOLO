"""CLI entry point for ``python -m volume_tta``."""

from __future__ import annotations

import sys

def run() -> None:
    # Keep discovery dependency-light: help/version and the no-argument argparse error
    # must not import OpenCV, SciPy, Ultralytics, or any accelerator runtime.
    if len(sys.argv) == 1 or any(token in {"-h", "--help", "--version"} for token in sys.argv[1:]):
        from .config import build_argparser

        build_argparser().parse_args()
        return

    from .media import abort_streaming_producers
    from .outputs import nrrd_layer_sink, set_nrrd_layer_sink
    from .pipeline import main

    try:
        main()
    except BaseException:
        abort_streaming_producers('fatal error in main()')
        fatal_layer_sink = nrrd_layer_sink()
        if fatal_layer_sink is not None:
            try:
                fatal_layer_sink.shutdown()
            except Exception:
                pass
            set_nrrd_layer_sink(None)
        raise

if __name__ == '__main__':
    run()
