"""Verify preservation of the monolith's top-level executable statement inventory."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "volume_tta"
MANIFEST = PACKAGE / "_refactor_manifest.json"

# These definitions received small, intentional seam fixes after physical extraction.
INTENTIONALLY_CHANGED = {
    ("config", "build_argparser"),
    ("config", "resolve_backend_batches"),
    ("config", "resolve_backend_precisions"),
    ("media", "abort_streaming_producers"),
    ("media", "decode_video_to_memmap_gray8_streaming"),
    ("media", "resize_volume_to_processing_cube_gray8_streaming"),
    ("outputs", "_publish_staged_file_atomically"),
    ("outputs", "_try_gpu_downbin_volume"),
    ("outputs", "_try_gpu_downbin_volume_on_device"),
    ("outputs", "NrrdLayerSink"),
    ("outputs", "write_binary_tiff_sequence_from_pattern"),
    ("outputs", "write_layer_nrrd_with_low_quality_mirrors"),
    ("outputs", "write_single_layer_nrrd_from_ref"),
    ("outputs", "write_view_images"),
    ("outputs", "write_yolo_labels_from_pattern"),
    ("pipeline", "main"),
    ("runtime", "_GpuWorkerAuxInterpolationPool"),
    ("runtime", "_materialize_worker_task_memfd_paths"),
    ("runtime", "interpolate_view_volume_pass_maybe_process"),
    ("runtime", "interpolation_process_start_method"),
    ("workers", "run_prediction_volume_in_worker"),
    ("topology", "_try_label_slices_stage_a_gpu"),
    ("interpolation", "interpolation_planning_backend_name"),
    ("finalization", "assemble_view_volumes_and_projected_layers_fused"),
}

# The public wrapper now owns the full-run cleanup boundary; the preserved orchestration
# body moved intact (plus reviewed v17.0.7 fixes) behind this private implementation name.
INTENTIONALLY_RENAMED_CHANGED = {
    ("pipeline", "main"): "_main_impl",
}

INTENTIONALLY_VERSIONED = {
    ("config", "451b35336c86c625bd71b77e55c8a09bef571c75405977484e6e0e6debadcd51"):
        "SCRIPT_VERSION",
    ("config", "bbaeec59e08232583950d10ce19229b162f82f5f41ec60afe2dff19fc2e9c6b2"):
        "SCRIPT_VERSION_COMPACT",
}

INTENTIONALLY_RELOCATED = {
    ("runtime", "70e22341666e8e63ad2a0a0239676cd85eb4aba6e256a55379ec7958cbc35799"):
        "workers",
    ("runtime", "a6ecc26570ba0d1a5feda101d96bfa587013159b1bf5370b5875aff9ed3ff212"):
        "workers",
    ("inference", "bf2ffc53f405ceff8a38bc5f846d09608a4ac82575a58d8938ceb0204d8bc99b"):
        "backprojection",
    ("inference", "fcc88a83030aa0dac3b506d345ee748d01c4cdc9e0e3b08aa9549cbfde0d44ca"):
        "backprojection",
}


def digest(node: ast.AST) -> str:
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_path = ROOT / str(manifest["source"])
    if not source_path.is_file():
        raise RuntimeError(f"refactor source is missing: {source_path}")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != str(manifest["source_sha256"]):
        raise RuntimeError(
            f"refactor source checksum mismatch: {source_sha256} != {manifest['source_sha256']}"
        )

    available: dict[str, Counter[str]] = {}
    top_level: dict[str, list[ast.stmt]] = {}
    for module in {str(item["module"]) for item in manifest["statements"]}:
        tree = ast.parse(
            (PACKAGE / f"{module}.py").read_text(encoding="utf-8"),
            filename=str(PACKAGE / f"{module}.py"),
        )
        top_level[module] = list(tree.body)
        available[module] = Counter(digest(node) for node in tree.body)

    missing_changed = [
        (module, name)
        for module, name in sorted(INTENTIONALLY_CHANGED)
        if sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == INTENTIONALLY_RENAMED_CHANGED.get((module, name), name)
            for node in top_level[module]
        ) != 1
    ]
    if missing_changed:
        raise RuntimeError(f"missing or duplicate reviewed seam functions: {missing_changed!r}")
    for (module, public_name), implementation_name in INTENTIONALLY_RENAMED_CHANGED.items():
        public_matches = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == public_name
            for node in top_level[module]
        )
        if public_matches != 1:
            raise RuntimeError(
                f"missing or duplicate public wrapper for {module}.{implementation_name}: "
                f"{module}.{public_name}"
            )

    missing_versions: list[tuple[str, str]] = []
    for (module, _original_hash), variable_name in INTENTIONALLY_VERSIONED.items():
        matches = 0
        for node in top_level[module]:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            matches += sum(isinstance(target, ast.Name) and target.id == variable_name for target in targets)
        if matches != 1:
            missing_versions.append((module, variable_name))
    if missing_versions:
        raise RuntimeError(f"missing or duplicate version declarations: {missing_versions!r}")

    missing: list[dict[str, object]] = []
    preserved = 0
    changed = 0
    for item in manifest["statements"]:
        module = str(item["module"])
        name = item.get("name")
        statement_hash = str(item["sha256"])
        if (
            (module, name) in INTENTIONALLY_CHANGED
            or (module, statement_hash) in INTENTIONALLY_VERSIONED
        ):
            changed += 1
            continue
        destination = INTENTIONALLY_RELOCATED.get((module, statement_hash), module)
        if available[destination][statement_hash] < 1:
            missing.append(item)
            continue
        available[destination][statement_hash] -= 1
        preserved += 1

    if missing:
        raise RuntimeError(f"missing {len(missing)} preserved statements: {missing!r}")
    expected = int(manifest["statement_count"])
    if preserved + changed != expected:
        raise RuntimeError(
            f"statement accounting mismatch: {preserved} preserved + {changed} changed != {expected}"
        )
    print(
        f"accounted for {preserved} unchanged original top-level statements and "
        f"{changed} reviewed original-statement changes ({expected} original statements total)"
    )


if __name__ == "__main__":
    main()
