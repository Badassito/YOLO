"""Report mutable module globals read across subsystem boundaries."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "volume_tta"
IGNORED = {"_deps", "__init__", "__main__"}


def assigned_names(target: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def module_bindings(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(assigned_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(assigned_names(node.target))
        elif isinstance(node, ast.If):
            for child in (*node.body, *node.orelse):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(child.name)
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        names.update(assigned_names(target))
    return names


def main() -> None:
    trees: dict[str, ast.Module] = {}
    owners: defaultdict[str, set[str]] = defaultdict(set)
    for path in sorted(PACKAGE.glob("*.py")):
        if path.stem in IGNORED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        trees[path.stem] = tree
        for name in module_bindings(tree):
            owners[name].add(path.stem)

    rebounds: defaultdict[str, set[str]] = defaultdict(set)
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                rebounds[module].update(node.names)

    findings: set[tuple[str, str, str]] = set()
    for consumer, tree in trees.items():
        loads = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for name in loads:
            for owner in owners.get(name, ()):
                if owner != consumer and name in rebounds[owner]:
                    findings.add((consumer, owner, name))
    for consumer, owner, name in sorted(findings):
        print(f"{consumer:16} <- {owner:16} {name}")


if __name__ == "__main__":
    main()
