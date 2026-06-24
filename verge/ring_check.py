"""Ring-enforcement guard — rings are enforced in code, not convention (§8).

Walks the import graph of the `verge` package starting from the self-improvement
layers and **fails the build** if any reachable module imports a *mutable* Ring-0
handle — i.e. anything from `verge.ring0` outside the read-only allowlist
(`verge.ring0.RING0_READONLY_EXPORTS`). Private internals (the append-only ledger
object `_LEDGER`, any future setter) and whole-module `import verge.ring0.*` are
violations; the curated read-only names (the verifier, the ladder, append-only
`record`/`read_ledger`, the frozen `THRESHOLDS`) are allowed.

Used by `tests/test_ring_enforcement.py` and importable for CI:
    python -m verge.ring_check    # exits non-zero on any violation
"""
from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass

# Layers on a self-improvement / autonomy path (Ring 1 + the learned core, Ring 2).
# Ring 0, the latent contract, the bus, the eval harness and configs are NOT here.
SELF_IMPROVEMENT_ROOTS = (
    "verge.l1_perception",
    "verge.l2_world_model",
    "verge.l3_reasoner",
    "verge.l4_plasticity",
    "verge.l5_social",
    "verge.l6_motivation",
    "verge.l7_router",
    "verge.memory",
)

_RING0_PREFIX = "verge.ring0"


@dataclass(frozen=True)
class Violation:
    module: str
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f"{self.module}:{self.lineno}: {self.detail}"


def _allowlist() -> frozenset[str]:
    # Imported lazily so the guard has no import-time dependency on Ring 0 internals.
    from verge.ring0 import RING0_READONLY_EXPORTS

    return RING0_READONLY_EXPORTS


def ring0_import_violations(source: str, *, module_name: str) -> list[Violation]:
    """Apply the Ring-0 import rule to a single module's source. Pure; used directly by
    the bad-fixture test to prove the guard fails on a violation."""
    allow = _allowlist()
    tree = ast.parse(source, filename=module_name)
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == _RING0_PREFIX or mod.startswith(_RING0_PREFIX + "."):
                for alias in node.names:
                    if alias.name == "*":
                        out.append(Violation(module_name, node.lineno,
                                             f"wildcard import from {mod} (too broad)"))
                    elif alias.name not in allow:
                        out.append(Violation(
                            module_name, node.lineno,
                            f"imports mutable/non-allowlisted Ring-0 handle "
                            f"'{alias.name}' from {mod}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _RING0_PREFIX or alias.name.startswith(_RING0_PREFIX + "."):
                    out.append(Violation(
                        module_name, node.lineno,
                        f"whole-module 'import {alias.name}' (must import specific "
                        f"read-only names, not a Ring-0 namespace)"))
    return out


def _module_name(path: str, package_root: str) -> str:
    rel = os.path.relpath(path, os.path.dirname(package_root))
    rel = rel[:-3] if rel.endswith(".py") else rel
    parts = [p for p in rel.split(os.sep) if p != "__init__"]
    return ".".join(parts)


def _verge_imports(source: str, module_name: str) -> set[str]:
    """The set of `verge.*` modules this module imports (for the reachability walk)."""
    edges: set[str] = set()
    tree = ast.parse(source, filename=module_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("verge"):
            edges.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("verge"):
                    edges.add(alias.name)
    return edges


def _load_tree(package_root: str) -> dict[str, tuple[str, str]]:
    """module_name -> (filepath, source) for every .py under verge/."""
    out: dict[str, tuple[str, str]] = {}
    for dirpath, _dirs, files in os.walk(package_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            name = _module_name(path, package_root)
            with open(path, encoding="utf-8") as f:
                out[name] = (path, f.read())
    return out


def _reachable(modules: dict[str, tuple[str, str]]) -> set[str]:
    """Modules reachable from the self-improvement roots via the import graph."""
    seen: set[str] = set()
    stack = [m for m in modules if any(
        m == r or m.startswith(r + ".") for r in SELF_IMPROVEMENT_ROOTS)]
    while stack:
        m = stack.pop()
        if m in seen or m not in modules:
            continue
        seen.add(m)
        for dep in _verge_imports(modules[m][1], m):
            # normalize package imports to module nodes we know about
            for cand in (dep, dep.rsplit(".", 1)[0]):
                if cand in modules and cand not in seen:
                    stack.append(cand)
    return seen


def scan(package_root: str | None = None) -> list[Violation]:
    """Walk the import graph from the self-improvement roots and return all Ring-0
    violations. Empty list == clean tree."""
    if package_root is None:
        package_root = os.path.dirname(os.path.abspath(__file__))
    modules = _load_tree(package_root)
    reachable = _reachable(modules)
    violations: list[Violation] = []
    for name in sorted(reachable):
        # Ring 0 is allowed to reference its own internals — it is not a self-improvement
        # path. The guard polices everything reachable that is NOT Ring 0 itself.
        if name == _RING0_PREFIX or name.startswith(_RING0_PREFIX + "."):
            continue
        _path, src = modules[name]
        violations.extend(ring0_import_violations(src, module_name=name))
    return violations


def reachable_modules(package_root: str | None = None) -> set[str]:
    """Exposed so the test can assert the walk actually traverses (e.g. reaches
    verge.l3_reasoner.reward, which legitimately imports the read-only verifier)."""
    if package_root is None:
        package_root = os.path.dirname(os.path.abspath(__file__))
    return _reachable(_load_tree(package_root))


def main() -> int:
    violations = scan()
    if violations:
        print("RING-0 ENFORCEMENT FAILED — self-improvement path imports a mutable handle:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("ring-0 enforcement: clean (no self-improvement path imports a mutable Ring-0 handle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
