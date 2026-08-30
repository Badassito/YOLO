"""Production TTA runner used by the unified CLI."""

from __future__ import annotations


def run() -> None:
    """Execute one already-validated TTA invocation from ``sys.argv``."""

    # Keep discovery dependency-light: validation and help happen in cli.py
    # before this production dependency surface is imported.
    from .media import abort_streaming_producers
    from .outputs import nrrd_layer_sink, set_nrrd_layer_sink
    from .pipeline import main

    try:
        main()
    except BaseException:
        abort_streaming_producers("fatal error in main()")
        fatal_layer_sink = nrrd_layer_sink()
        if fatal_layer_sink is not None:
            try:
                fatal_layer_sink.shutdown()
            except Exception:
                pass
            set_nrrd_layer_sink(None)
        raise


__all__ = ["run"]
