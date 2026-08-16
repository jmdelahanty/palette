#!/usr/bin/env python3
"""Reject guessed string literals assigned to observation-valued metadata keys."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


OBSERVATION_SUFFIXES = ("_observed", "_measured", "_detected")
# Metric/display names can end with the same English word without claiming an
# observed metadata authority. Keep this semantic exclusion deliberately tiny.
NON_OBSERVATION_KEYS = frozenset({"fraction_of_detected"})
SCHEMA_TYPE_LITERALS = frozenset(
    {
        "bool",
        "boolean",
        "bytes",
        "float",
        "float32",
        "float64",
        "int",
        "int32",
        "int64",
        "str",
        "string",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    key: str
    value: str


def _observation_key(value: ast.AST | None) -> str | None:
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    key = value.value
    return (
        key
        if key.endswith(OBSERVATION_SUFFIXES) and key not in NON_OBSERVATION_KEYS
        else None
    )


def _literal_string(value: ast.AST | None) -> str | None:
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return value.value


def _is_schema_type_literal(value: str) -> bool:
    return value.strip().lower() in SCHEMA_TYPE_LITERALS


def _scan(path: Path) -> list[Violation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[Violation] = []
    for node in ast.walk(tree):
        candidates: list[tuple[ast.AST | None, ast.AST | None, int]] = []
        if isinstance(node, ast.Dict):
            candidates.extend(
                (key, value, int(getattr(key, "lineno", node.lineno)))
                for key, value in zip(node.keys, node.values)
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript):
                    candidates.append((target.slice, value, int(node.lineno)))
        for raw_key, raw_value, line in candidates:
            key = _observation_key(raw_key)
            value = _literal_string(raw_value)
            if key is None or value is None or _is_schema_type_literal(value):
                continue
            violations.append(Violation(path=path, line=line, key=key, value=value))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject literal values for *_observed/*_measured/*_detected metadata."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("src/fisheye")],
    )
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for root in args.roots:
        if root.is_file() and root.suffix == ".py":
            paths.append(root)
        elif root.is_dir():
            paths.extend(root.rglob("*.py"))
    violations = [
        violation
        for path in sorted(set(paths))
        for violation in _scan(path)
    ]
    if violations:
        print("Observed metadata must come from runtime evidence, not literals:")
        for item in violations:
            print(f"- {item.path}:{item.line}: {item.key}={item.value!r}")
        return 1
    print(f"Checked {len(set(paths))} Python file(s); no observed metadata literals found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
