"""TTA run-resource ownership and successful-publication lifecycle.

The production pipeline retains the public/facade entry point.  This module owns the
mechanics that do not depend on scheduler-local state, with fallible filesystem and
publication operations supplied explicitly so the facade's established monkeypatch seams
remain usable by embedded callers and regression tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple


class PipelineRunResources:
    """Last-resort ownership for resources created during one TTA run."""

    def __init__(self) -> None:
        self.executors: List[object] = []
        self.output_managers: List[object] = []
        self.sinks: List[object] = []
        self.processes: List[object] = []
        self.queues: List[object] = []
        self.threads: List[Tuple[threading.Thread, Optional[threading.Event]]] = []
        self._seen: set[int] = set()

    def _add(self, collection: List[object], resource: object) -> object:
        if resource is not None and id(resource) not in self._seen:
            self._seen.add(id(resource))
            collection.append(resource)
        return resource

    def track_executor(self, resource: object) -> object:
        return self._add(self.executors, resource)

    def track_output_manager(self, resource: object) -> object:
        return self._add(self.output_managers, resource)

    def track_sink(self, resource: object) -> object:
        return self._add(self.sinks, resource)

    def track_process(self, resource: object) -> object:
        return self._add(self.processes, resource)

    def track_queue(self, resource: object) -> object:
        return self._add(self.queues, resource)

    def track_thread(
        self,
        thread: threading.Thread,
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        if id(thread) not in self._seen:
            self._seen.add(id(thread))
            self.threads.append((thread, stop_event))
        return thread

    def close(self, *, failed: bool) -> None:
        for _thread, stop_event in self.threads:
            if stop_event is not None:
                stop_event.set()

        # A process left here escaped the scheduler's cooperative sentinel path. Terminate
        # it before closing queues or waiting on parent thread pools that may depend on it.
        for proc in reversed(self.processes):
            try:
                proc.join(timeout=0.0 if failed else 0.25)  # type: ignore[attr-defined]
                if proc.is_alive():  # type: ignore[attr-defined]
                    proc.terminate()  # type: ignore[attr-defined]
            except Exception:
                pass
        for proc in reversed(self.processes):
            try:
                proc.join(timeout=2.0)  # type: ignore[attr-defined]
                if proc.is_alive() and hasattr(proc, "kill"):  # type: ignore[attr-defined]
                    proc.kill()  # type: ignore[attr-defined]
                    proc.join(timeout=1.0)  # type: ignore[attr-defined]
            except Exception:
                pass

        for manager in reversed(self.output_managers):
            try:
                manager.wait()  # type: ignore[attr-defined]
            except Exception:
                pass
        for sink in reversed(self.sinks):
            try:
                sink.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass
        for executor in reversed(self.executors):
            try:
                executor.shutdown(wait=True, cancel_futures=bool(failed))  # type: ignore[attr-defined]
            except TypeError:
                try:
                    executor.shutdown(wait=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass
        for thread, _stop_event in reversed(self.threads):
            try:
                if thread is not threading.current_thread():
                    thread.join(timeout=5.0)
            except Exception:
                pass
        for process_queue in reversed(self.queues):
            if failed:
                try:
                    process_queue.cancel_join_thread()  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                process_queue.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            if not failed:
                try:
                    process_queue.join_thread()  # type: ignore[attr-defined]
                except Exception:
                    pass


def cleanup_tta_selected_run_scratch(
    *,
    temp_dir: Path,
    out_dir: Path,
    release_memfd_owners_under: Callable[[Path], int],
    remove_tree: Callable[[Path], object],
) -> None:
    """Strictly retire scratch owned by the selected run."""

    temp_dir = Path(temp_dir)
    out_dir = Path(out_dir)
    try:
        released_memfd_files = release_memfd_owners_under(temp_dir)
        if int(released_memfd_files) > 0:
            print(f"Released {int(released_memfd_files)} memfd-backed scratch payload(s).")

        for child in list(temp_dir.iterdir()):
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                remove_tree(child)
            else:
                child.unlink(missing_ok=True)

        default_temp_dir = out_dir / "temp"
        if temp_dir != default_temp_dir:
            temp_dir.rmdir()
            temp_link = default_temp_dir
            if temp_link.is_symlink():
                temp_link.unlink()
            elif temp_link.exists():
                marker = temp_link / "SCRATCH_LOCATION.txt"
                if (
                    not marker.is_file()
                    or marker.read_text(encoding="utf-8").strip() != str(temp_dir)
                ):
                    raise RuntimeError(
                        "scratch exposure path is not the selected run marker: "
                        f"{temp_link}"
                    )
                marker.unlink()
                temp_link.rmdir()
    except Exception as exc:
        raise RuntimeError(
            "TTA selected-run scratch cleanup failed; refusing a complete manifest: "
            f"{temp_dir}"
        ) from exc


def publish_complete_tta_manifest(
    *,
    path: Path,
    manifest: Mapping[str, object],
    artifact_identities: Mapping[str, object],
    assert_artifacts_unchanged: Callable[[Mapping[str, object]], None],
    write_manifest: Callable[[Path, Mapping[str, object]], Path],
) -> Path:
    """Revalidate authoritative artifacts and atomically commit TTA success."""

    if str(manifest.get("status", "")) != "complete":
        raise ValueError("TTA complete-manifest publisher requires status=complete")
    # Keep this check immediately adjacent to the atomic write. In particular, no cleanup,
    # summary generation, or manifest construction may be inserted between these calls.
    assert_artifacts_unchanged(artifact_identities)
    return write_manifest(path, manifest)


def finalize_tta_selected_run_and_publish(
    *,
    temp_dir: Path,
    out_dir: Path,
    keep_temp_artifacts: bool,
    manifest_path: Path,
    manifest: Mapping[str, object],
    artifact_identities: Mapping[str, object],
    cleanup_scratch: Callable[..., None],
    publish_manifest: Callable[..., Path],
) -> Path:
    """Finish selected-run cleanup before publishing the complete manifest."""

    if not bool(keep_temp_artifacts):
        cleanup_scratch(temp_dir=temp_dir, out_dir=out_dir)
    return publish_manifest(
        path=manifest_path,
        manifest=manifest,
        artifact_identities=artifact_identities,
    )


__all__ = [
    "PipelineRunResources",
    "cleanup_tta_selected_run_scratch",
    "finalize_tta_selected_run_and_publish",
    "publish_complete_tta_manifest",
]
