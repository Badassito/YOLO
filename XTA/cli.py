"""Dependency-light mode dispatcher for the unified PTA/TTA launcher."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator, Sequence

from .unification.context import activate_unified_launch


SCRIPT_VERSION = "18.0.1"
SCRIPT_BASENAME = "GPT-5.6-Sol-Ultra_v18.0.1_SLURM.py"
MODE_CHOICES = ("tta", "pta")


def build_argparser(*, dispatch_only: bool = False) -> argparse.ArgumentParser:
    """Build the dependency-free v18 launcher parser.

    The dispatch-only form deliberately omits ``-h/--help`` so help following a
    selected mode is handled by that mode's complete parser.
    """

    parser = argparse.ArgumentParser(
        prog=SCRIPT_BASENAME,
        usage=f"{SCRIPT_BASENAME} --mode {{tta,pta}} ...",
        description=(
            "Mode-aware launcher for volume pretraining augmentation and test-time "
            "augmentation. Select a mode, then supply only that mode's flags. Use "
            "'--mode tta --help' or '--mode pta --help' for mode-specific help."
        ),
        allow_abbrev=False,
        add_help=not bool(dispatch_only),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES)
    return parser


def _has_mode_argument(arguments: Sequence[str]) -> bool:
    return any(
        str(token) == "--mode" or str(token).startswith("--mode=")
        for token in arguments
    )


def _mode_argument_count(arguments: Sequence[str]) -> int:
    return sum(
        1
        for token in arguments
        if str(token) == "--mode" or str(token).startswith("--mode=")
    )


def _option_argument_count(arguments: Sequence[str], option: str) -> int:
    """Count exact scalar-option occurrences, including ``--flag=value`` form."""

    option_name = str(option)
    return sum(
        1
        for token in arguments
        if str(token) == option_name or str(token).startswith(f"{option_name}=")
    )


@contextlib.contextmanager
def _mode_sys_argv(arguments: Sequence[str]) -> Iterator[None]:
    """Expose mode-local arguments to the established TTA entry point."""

    previous = sys.argv
    sys.argv = [str(previous[0]), *(str(value) for value in arguments)]
    try:
        yield
    finally:
        sys.argv = previous


def _run_tta(arguments: Sequence[str]) -> None:
    """Validate with the existing TTA grammar, then run its production pipeline."""

    mode_arguments = [str(value) for value in arguments]
    # Validate before importing the production pipeline. This preserves argparse's
    # unavailable/foreign-flag errors on dependency-light hosts and keeps all v17 TTA
    # defaults and accepted flags authoritative.
    from .config import build_argparser as build_tta_argparser

    parser = build_tta_argparser()
    if _option_argument_count(mode_arguments, "--channel_format") > 1:
        parser.error("--channel_format may be provided only once in v18")
    parser.parse_args(mode_arguments)

    # The established entry point and pipeline consume sys.argv. Remove the v18 mode
    # selector for the duration of the call rather than teaching the v17 parser a new flag.
    from .tta_mode import run as run_tta

    with activate_unified_launch(
        version=SCRIPT_VERSION,
        launcher=SCRIPT_BASENAME,
        mode="tta",
        mode_arguments=mode_arguments,
    ):
        with _mode_sys_argv(mode_arguments):
            run_tta()


def _run_pta(arguments: Sequence[str]) -> None:
    """Load and invoke the PTA implementation only after PTA mode is selected."""

    from .pta_mode import run as run_pta

    mode_arguments = [str(value) for value in arguments]
    with activate_unified_launch(
        version=SCRIPT_VERSION,
        launcher=SCRIPT_BASENAME,
        mode="pta",
        mode_arguments=mode_arguments,
    ):
        run_pta(mode_arguments)


def run(argv: Sequence[str] | None = None) -> None:
    """Dispatch one v18 invocation.

    ``argv`` excludes the program name. Passing it explicitly keeps dispatch tests
    independent from process-global command-line state.
    """

    arguments = [str(value) for value in (sys.argv[1:] if argv is None else argv)]

    # Top-level discovery must not import OpenCV, SciPy, model runtimes, or either
    # mode implementation. Once a mode is present, its own --help remains authoritative.
    if not arguments or (
        not _has_mode_argument(arguments)
        and any(token in {"-h", "--help"} for token in arguments)
    ):
        build_argparser().parse_args(arguments)
        return

    dispatch_parser = build_argparser(dispatch_only=True)
    namespace, mode_arguments = dispatch_parser.parse_known_args(arguments)
    if _mode_argument_count(arguments) != 1:
        dispatch_parser.error("--mode must be specified exactly once")

    if namespace.mode == "tta":
        _run_tta(mode_arguments)
        return
    if namespace.mode == "pta":
        _run_pta(mode_arguments)
        return
    raise AssertionError(f"unhandled mode {namespace.mode!r}")


__all__ = [
    "MODE_CHOICES",
    "SCRIPT_BASENAME",
    "SCRIPT_VERSION",
    "build_argparser",
    "run",
]
