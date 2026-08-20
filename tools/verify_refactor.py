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
    ("workers", "run_prediction_volume_in_worker"),
    ("topology", "_try_label_slices_stage_a_gpu"),
    ("interpolation", "interpolation_planning_backend_name"),
    ("finalization", "assemble_view_volumes_and_projected_layers_fused"),
}

INTENTIONALLY_VERSIONED = {
    ("config", "451b35336c86c625bd71b77e55c8a09bef571c75405977484e6e0e6debadcd51"),
    ("config", "bbaeec59e08232583950d10ce19229b162f82f5f41ec60afe2dff19fc2e9c6b2"),
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
    available: dict[str, Counter[str]] = {}
    for module in {str(item["module"]) for item in manifest["statements"]}:
        tree = ast.parse(
            (PACKAGE / f"{module}.py").read_text(encoding="utf-8"),
            filename=str(PACKAGE / f"{module}.py"),
        )
        available[module] = Counter(digest(node) for node in tree.body)

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
        raise RuntimeError(f"missing {len(missing)} preserved statements: {missing[:12]!r}")
    expected = int(manifest["statement_count"])
    if preserved + changed != expected:
        raise RuntimeError(
            f"statement accounting mismatch: {preserved} preserved + {changed} changed != {expected}"
        )
    print(
        f"verified {preserved} unchanged top-level statements and "
        f"{changed} reviewed seam changes ({expected} total)"
    )


if __name__ == "__main__":
    main()
