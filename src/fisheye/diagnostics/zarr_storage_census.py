"""Read-only static census of Palette Zarr schemas and writer surfaces.

The census intentionally reports uncertainty instead of trying to execute
writer code.  It combines two independent views:

* logical declarations made with ``ArraySpec``; and
* physical creation sites, including calls routed through local/imported
  writer helpers and the encoded-mask path that writes Zarr metadata itself.

Run from the repository root with::

    scripts/py -m fisheye.diagnostics.zarr_storage_census --write

No archive is opened and no production data is modified.  ``--write`` only
replaces the checked-in diagnostic artifacts owned by this module.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from fisheye.diagnostics.zarr_detection_schema_inventory import (
    DETECTION_OUTPUT,
    DETECTION_SUMMARY_OUTPUT,
    build_detection_document,
    render_detection_inventory,
)

SCHEMA_OUTPUT = Path("docs/diagnostics/zarr_array_schema_census.json")
WRITER_OUTPUT = Path("docs/diagnostics/zarr_production_writer_census.json")
SUMMARY_OUTPUT = Path("docs/diagnostics/zarr_storage_census_summary.md")

ARRAY_SPEC_FILES = (
    Path("src/fisheye/shared/zarr/stage_arrays.py"),
    Path("src/fisheye/shared/zarr/analysis_stage_arrays.py"),
)

DIRECT_ARRAY_METHODS = frozenset(
    {
        "create_array",
        "create_dataset",
        "require_array",
        "require_dataset",
        "to_zarr",
    }
)

GROUP_METHODS = frozenset({"create_group", "require_group"})

NAME_PARAMETER_CANDIDATES = (
    "name",
    "array_name",
    "dataset_name",
    "key",
)
DATA_PARAMETER_CANDIDATES = ("data", "values", "value", "array")

MULTI_ARRAY_WRAPPER_NAMES = frozenset(
    {
        "copy_completed_run_to_sharded",
        "write_columnar_dataset",
        "write_bitpacked_mask_store_from_dense",
        "write_component_rle_mask_store_from_dense",
        "write_encoded_component_rle_mask_store",
        "refresh_bitpacked_mask_store_from_dense",
        "refresh_component_rle_mask_store_from_dense",
    }
)

_LEGACY_STAGE_NAMES = frozenset({"eye_masks", "refined_eye_masks"})


@dataclass(frozen=True)
class ModuleInfo:
    """Parsed source module and the static information needed by the census."""

    path: Path
    relative_path: str
    module_name: str
    tree: ast.Module
    constants: Mapping[str, object]
    module_assignments: Mapping[str, ast.AST]
    imports: Mapping[str, str]
    has_zarr_import: bool


@dataclass(frozen=True)
class FunctionInfo:
    """One function/method definition addressable by wrapper-call analysis."""

    symbol: str
    module_name: str
    relative_path: str
    local_qualname: str
    leaf_name: str
    parameters: tuple[str, ...]
    node: ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class CallInfo:
    """One call expression with its enclosing symbol and static scope."""

    module: ModuleInfo
    node: ast.Call
    enclosing_symbol: str
    enclosing_local_qualname: str | None
    assignments: Mapping[str, ast.AST]


def _source_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return None


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _module_name(relative_path: Path) -> str:
    if relative_path.parts and relative_path.parts[0] == "src":
        parts = list(relative_path.with_suffix("").parts[1:])
    else:
        parts = list(relative_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _literal(
    node: ast.AST | None,
    constants: Mapping[str, object],
    *,
    seen: frozenset[str] = frozenset(),
) -> object | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_literal(item, constants, seen=seen) for item in node.elts]
        if any(value is None for value in values):
            return None
        return tuple(values)
    if isinstance(node, ast.Dict):
        keys = [_literal(item, constants, seen=seen) for item in node.keys]
        values = [_literal(item, constants, seen=seen) for item in node.values]
        if any(key is None for key in keys) or any(value is None for value in values):
            return None
        return dict(zip(keys, values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand, constants, seen=seen)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value
    if isinstance(node, ast.Name) and node.id not in seen and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal(node.left, constants, seen=seen)
        right = _literal(node.right, constants, seen=seen)
        if isinstance(left, (str, tuple)) and isinstance(right, type(left)):
            return left + right
    return None


def _top_level_bindings(
    tree: ast.Module,
) -> tuple[dict[str, object], dict[str, ast.AST]]:
    constants: dict[str, object] = {}
    assignments: dict[str, ast.AST] = {}
    pending: list[tuple[str, ast.AST]] = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if value is None:
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
                pending.append((target.id, value))

    changed = True
    while changed:
        changed = False
        for name, value_node in pending:
            if name in constants:
                continue
            value = _literal(value_node, constants)
            if value is not None:
                constants[name] = value
                changed = True
    return constants, assignments


def _imports(tree: ast.Module, module_name: str) -> tuple[dict[str, str], bool]:
    aliases: dict[str, str] = {}
    has_zarr = False
    package_parts = module_name.split(".")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name
                if alias.name == "zarr" or alias.name.startswith("zarr."):
                    has_zarr = True
        elif isinstance(node, ast.ImportFrom):
            imported_module = node.module or ""
            if node.level:
                prefix = package_parts[: max(0, len(package_parts) - node.level)]
                imported_module = ".".join([*prefix, imported_module]).strip(".")
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                aliases[local] = f"{imported_module}.{alias.name}".strip(".")
            if imported_module == "zarr" or imported_module.startswith("zarr."):
                has_zarr = True
    return aliases, has_zarr


def _source_has_direct_census_surface(source: str) -> bool:
    return (
        "ArraySpec(" in source
        or any(f".{method}(" in source for method in DIRECT_ARRAY_METHODS)
        or ("zarr.json" in source and "_atomic_write_json(" in source)
    )


def _parse_module(repo_root: Path, path: Path, source: str) -> ModuleInfo:
    relative = path.relative_to(repo_root)
    try:
        tree = ast.parse(source, filename=relative.as_posix())
    except SyntaxError as exc:
        raise SyntaxError(f"Cannot parse census source {relative}: {exc}") from exc
    constants, assignments = _top_level_bindings(tree)
    name = _module_name(relative)
    import_map, has_zarr = _imports(tree, name)
    return ModuleInfo(
        path=path,
        relative_path=relative.as_posix(),
        module_name=name,
        tree=tree,
        constants=constants,
        module_assignments=assignments,
        imports=import_map,
        has_zarr_import=has_zarr,
    )


def _load_modules(repo_root: Path) -> tuple[list[ModuleInfo], int]:
    all_paths = sorted(
        [*repo_root.glob("src/**/*.py"), *repo_root.glob("scripts/**/*.py")],
        key=lambda item: item.relative_to(repo_root).as_posix(),
    )
    source_by_path: dict[Path, str] = {}
    candidate_paths: set[Path] = set()
    scanned_count = 0
    for path in all_paths:
        relative = path.relative_to(repo_root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        scanned_count += 1
        source = path.read_text(encoding="utf-8")
        source_by_path[path] = source
        if _source_has_direct_census_surface(source):
            candidate_paths.add(path)

    parsed: dict[Path, ModuleInfo] = {
        path: _parse_module(repo_root, path, source_by_path[path])
        for path in sorted(candidate_paths)
    }
    # Resolve callers of writer helpers without retaining ASTs for the hundreds
    # of source modules that only read arrays.  Iterate because one helper may
    # wrap another while preserving a name parameter.
    while True:
        functions, calls = _collect_functions_and_calls(list(parsed.values()))
        writer_symbols, _ = _writer_function_symbols(functions, calls)
        by_symbol = {function.symbol: function for function in functions}
        wrapper_leaves = {
            by_symbol[symbol].leaf_name
            for symbol in writer_symbols
            if symbol in by_symbol
            and (
                any(
                    parameter in NAME_PARAMETER_CANDIDATES
                    for parameter in by_symbol[symbol].parameters
                )
                or by_symbol[symbol].leaf_name in MULTI_ARRAY_WRAPPER_NAMES
            )
        }
        new_paths = {
            path
            for path, source in source_by_path.items()
            if path not in parsed
            and any(f"{leaf}(" in source for leaf in wrapper_leaves)
        }
        if not new_paths:
            break
        for path in sorted(new_paths):
            parsed[path] = _parse_module(repo_root, path, source_by_path[path])

    return (
        sorted(parsed.values(), key=lambda module: module.relative_path),
        scanned_count,
    )


def _collect_scope_assignments(node: ast.AST) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}

    def visit(current: ast.AST) -> None:
        if current is not node and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        if (
            isinstance(current, (ast.Assign, ast.AnnAssign))
            and current.value is not None
        ):
            targets = (
                current.targets if isinstance(current, ast.Assign) else [current.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = current.value
        for child in ast.iter_child_nodes(current):
            visit(child)

    visit(node)
    return assignments


class _DefinitionAndCallCollector(ast.NodeVisitor):
    def __init__(self, module: ModuleInfo) -> None:
        self.module = module
        self.qualname_stack: list[str] = []
        self.assignment_stack: list[Mapping[str, ast.AST]] = [module.module_assignments]
        self.functions: list[FunctionInfo] = []
        self.calls: list[CallInfo] = []

    def _visit_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.qualname_stack.append(node.name)
        local_qualname = ".".join(self.qualname_stack)
        symbol = f"{self.module.module_name}:{local_qualname}"
        positional = [*node.args.posonlyargs, *node.args.args]
        parameters = tuple(
            [argument.arg for argument in positional]
            + [argument.arg for argument in node.args.kwonlyargs]
        )
        self.functions.append(
            FunctionInfo(
                symbol=symbol,
                module_name=self.module.module_name,
                relative_path=self.module.relative_path,
                local_qualname=local_qualname,
                leaf_name=node.name,
                parameters=parameters,
                node=node,
            )
        )
        local_assignments = dict(self.assignment_stack[-1])
        local_assignments.update(_collect_scope_assignments(node))
        self.assignment_stack.append(local_assignments)
        for statement in node.body:
            self.visit(statement)
        self.assignment_stack.pop()
        self.qualname_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.qualname_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.qualname_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        local = ".".join(self.qualname_stack) or None
        symbol = (
            f"{self.module.module_name}:{local}"
            if local is not None
            else f"{self.module.module_name}:<module>"
        )
        self.calls.append(
            CallInfo(
                module=self.module,
                node=node,
                enclosing_symbol=symbol,
                enclosing_local_qualname=local,
                assignments=self.assignment_stack[-1],
            )
        )
        self.generic_visit(node)


def _collect_functions_and_calls(
    modules: Sequence[ModuleInfo],
) -> tuple[list[FunctionInfo], list[CallInfo]]:
    functions: list[FunctionInfo] = []
    calls: list[CallInfo] = []
    for module in modules:
        collector = _DefinitionAndCallCollector(module)
        collector.visit(module.tree)
        functions.extend(collector.functions)
        calls.extend(collector.calls)
    return functions, calls


def _attribute_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _call_leaf(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_direct_array_creation(call: CallInfo) -> bool:
    leaf = _call_leaf(call.node)
    if leaf not in DIRECT_ARRAY_METHODS:
        return False
    if leaf == "create_dataset":
        # ``create_dataset`` is also the dominant h5py API in legacy scripts.
        # Only retain it when the module itself imports Zarr.
        return call.module.has_zarr_import
    if leaf == "to_zarr":
        return True
    return leaf in {"create_array", "require_array", "require_dataset"}


def _resolve_callee(
    call: CallInfo,
    functions_by_symbol: Mapping[str, FunctionInfo],
    functions_by_module_leaf: Mapping[tuple[str, str], Sequence[FunctionInfo]],
    functions_by_import_path: Mapping[str, Sequence[FunctionInfo]],
) -> str | None:
    func = call.node.func
    if isinstance(func, ast.Name):
        imported = call.module.imports.get(func.id)
        if imported:
            candidates = functions_by_import_path.get(imported, ())
            return candidates[0].symbol if len(candidates) == 1 else None
        candidates = functions_by_module_leaf.get(
            (call.module.module_name, func.id), ()
        )
        if len(candidates) == 1:
            return candidates[0].symbol
        if call.enclosing_local_qualname:
            pieces = call.enclosing_local_qualname.split(".")[:-1]
            while pieces:
                candidate = f"{call.module.module_name}:{'.'.join([*pieces, func.id])}"
                if candidate in functions_by_symbol:
                    return candidate
                pieces.pop()
        return None
    if isinstance(func, ast.Attribute):
        chain = _attribute_chain(func)
        if chain is None:
            return None
        parts = chain.split(".")
        imported = call.module.imports.get(parts[0])
        if imported:
            target = ".".join([imported, *parts[1:]])
            candidates = functions_by_import_path.get(target, ())
            return candidates[0].symbol if len(candidates) == 1 else None
        if isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"}:
            if call.enclosing_local_qualname:
                class_parts = call.enclosing_local_qualname.split(".")[:-1]
                candidate = (
                    f"{call.module.module_name}:{'.'.join([*class_parts, func.attr])}"
                )
                if candidate in functions_by_symbol:
                    return candidate
    return None


def _writer_function_symbols(
    functions: Sequence[FunctionInfo], calls: Sequence[CallInfo]
) -> tuple[set[str], dict[int, str | None]]:
    by_symbol = {function.symbol: function for function in functions}
    by_module_leaf: dict[tuple[str, str], list[FunctionInfo]] = {}
    for function in functions:
        by_module_leaf.setdefault(
            (function.module_name, function.leaf_name), []
        ).append(function)
    by_import_path: dict[str, list[FunctionInfo]] = {}
    for function in functions:
        by_import_path.setdefault(
            f"{function.module_name}.{function.leaf_name}", []
        ).append(function)
    for values in by_module_leaf.values():
        values.sort(key=lambda item: item.symbol)
    for values in by_import_path.values():
        values.sort(key=lambda item: item.symbol)
    resolved: dict[int, str | None] = {
        id(call.node): _resolve_callee(call, by_symbol, by_module_leaf, by_import_path)
        for call in calls
    }
    writers = {
        call.enclosing_symbol
        for call in calls
        if _is_direct_array_creation(call)
        and not call.enclosing_symbol.endswith(":<module>")
    }
    changed = True
    while changed:
        changed = False
        for call in calls:
            callee = resolved[id(call.node)]
            if callee in writers and not call.enclosing_symbol.endswith(":<module>"):
                caller = by_symbol.get(call.enclosing_symbol)
                exposes_array_identity = caller is not None and (
                    any(
                        parameter in NAME_PARAMETER_CANDIDATES
                        for parameter in caller.parameters
                    )
                    or caller.leaf_name in MULTI_ARRAY_WRAPPER_NAMES
                )
                if exposes_array_identity and call.enclosing_symbol not in writers:
                    writers.add(call.enclosing_symbol)
                    changed = True
    return writers, resolved


def _argument_for_parameter(
    call: ast.Call,
    function: FunctionInfo | None,
    candidates: Sequence[str],
) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg in candidates:
            return keyword.value
    if function is None:
        return None
    for candidate in candidates:
        if candidate not in function.parameters:
            continue
        index = function.parameters.index(candidate)
        if index < len(call.args):
            return call.args[index]
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _direct_name_node(call: ast.Call) -> ast.AST | None:
    named = _keyword(call, "name")
    if named is not None:
        return named
    if call.args:
        return call.args[0]
    return None


def _resolve_name(node: ast.AST | None, call: CallInfo) -> tuple[str | None, bool]:
    value = _literal(node, call.module.constants)
    if isinstance(value, str):
        return value, False
    return None, node is not None


def _dtype_token(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    text = _source_text(node)
    if text is None:
        return None
    token = text.replace("numpy.", "np.").split(".")[-1]
    if token.startswith("dtype("):
        token = token[6:].rstrip(")").strip("'\"")
    known = (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
    return token if token in known else None


def _infer_dtype(
    node: ast.AST | None,
    call: CallInfo,
    *,
    seen: frozenset[str] = frozenset(),
) -> str | None:
    if node is None:
        return None
    direct = _dtype_token(node)
    if direct is not None:
        return direct
    if isinstance(node, ast.Name) and node.id not in seen:
        assigned = call.assignments.get(node.id)
        if assigned is not None:
            return _infer_dtype(assigned, call, seen=seen | {node.id})
    if isinstance(node, ast.Call):
        leaf = _call_leaf(node)
        dtype_node = _keyword(node, "dtype")
        inferred = _dtype_token(dtype_node)
        if inferred is not None:
            return inferred
        if isinstance(node.func, ast.Attribute) and node.func.attr == "astype":
            if node.args:
                inferred = _dtype_token(node.args[0])
                if inferred is not None:
                    return inferred
        if leaf in {"asarray", "array", "zeros", "ones", "empty", "full", "arange"}:
            return _dtype_token(dtype_node)
    if isinstance(node, ast.Attribute) and node.attr == "dtype":
        return _infer_dtype(node.value, call, seen=seen)
    return None


def _infer_shape_expression(
    node: ast.AST | None,
    call: CallInfo,
    *,
    seen: frozenset[str] = frozenset(),
) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name) and node.id not in seen:
        assigned = call.assignments.get(node.id)
        if assigned is not None:
            return _infer_shape_expression(assigned, call, seen=seen | {node.id})
    if isinstance(node, ast.Call):
        leaf = _call_leaf(node)
        if leaf in {"zeros", "ones", "empty", "full"} and node.args:
            return _source_text(node.args[0])
        shape_node = _keyword(node, "shape")
        if shape_node is not None:
            return _source_text(shape_node)
        if leaf in {"asarray", "array"} and node.args:
            return f"shape({_source_text(node.args[0])})"
        if isinstance(node.func, ast.Attribute) and node.func.attr == "astype":
            return f"shape({_source_text(node.func.value)})"
    if isinstance(node, ast.Subscript):
        return f"shape({_source_text(node)})"
    return f"shape({_source_text(node)})"


def _path_expression(
    node: ast.AST | None,
    call: CallInfo,
    *,
    seen: frozenset[str] = frozenset(),
) -> str:
    if node is None:
        return "<unresolved-parent>"
    if isinstance(node, ast.Name):
        if node.id not in seen and node.id in call.assignments:
            assigned = call.assignments[node.id]
            if isinstance(assigned, (ast.Subscript, ast.BinOp, ast.Call)):
                return _path_expression(assigned, call, seen=seen | {node.id})
        return f"<{node.id}>"
    if isinstance(node, ast.Subscript):
        parent = _path_expression(node.value, call, seen=seen)
        value = _literal(node.slice, call.module.constants)
        segment = (
            str(value)
            if isinstance(value, (str, int))
            else f"<{_source_text(node.slice)}>"
        )
        return f"{parent.rstrip('/')}/{segment}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parent = _path_expression(node.left, call, seen=seen)
        value = _literal(node.right, call.module.constants)
        segment = (
            str(value)
            if isinstance(value, (str, int))
            else f"<{_source_text(node.right)}>"
        )
        return f"{parent.rstrip('/')}/{segment}"
    if isinstance(node, ast.Call):
        leaf = _call_leaf(node)
        if leaf in GROUP_METHODS and isinstance(node.func, ast.Attribute):
            parent = _path_expression(node.func.value, call, seen=seen)
            name_node = _keyword(node, "name") or (node.args[0] if node.args else None)
            value = _literal(name_node, call.module.constants)
            segment = (
                str(value)
                if isinstance(value, (str, int))
                else f"<{_source_text(name_node)}>"
            )
            return f"{parent.rstrip('/')}/{segment}"
        chain = _attribute_chain(node.func)
        if chain in {"zarr.open_group", "zarr.group", "zarr.open_array"}:
            target = node.args[0] if node.args else _keyword(node, "store")
            return f"<zarr-root:{_source_text(target)}>"
        return f"<{_source_text(node)}>"
    return f"<{_source_text(node)}>"


def _surface_class(relative_path: str) -> tuple[str, str, str]:
    lower = relative_path.lower()
    filename = Path(relative_path).name.lower()
    if "/diagnostics/benchmark_" in lower or filename.startswith("benchmark_"):
        return "benchmark", "unclear", "filename identifies a benchmark surface"
    if "/diagnostics/" in lower:
        return "diagnostic", "unclear", "module is under diagnostics"
    if "/training/" in lower or "training" in filename:
        return (
            "training",
            "current",
            "module is a training reader/writer or training export",
        )
    if any(token in lower for token in ("sharded_copy", "compact", "rechunk")):
        return (
            "compaction",
            "current",
            "module performs sharding, compaction, or rechunking",
        )
    if any(
        token in lower for token in ("publish", "publication", "promotion", "finalize")
    ):
        return (
            "publication",
            "current",
            "module name identifies publication/finalization",
        )
    if any(
        token in lower
        for token in ("migrate", "migration", "backfill", "repair", "legacy", "patch_")
    ):
        return (
            "migration_or_maintenance",
            "unclear",
            "module name identifies compatibility maintenance",
        )
    if relative_path.startswith("scripts/"):
        return (
            "script",
            "unclear",
            "standalone script; production status requires review",
        )
    if relative_path.startswith("src/") and not relative_path.startswith(
        "src/fisheye/"
    ):
        return (
            "legacy",
            "unclear",
            "top-level pre-package location suggests legacy, but use is not statically proven",
        )
    return (
        "production",
        "current",
        "packaged Palette module without a maintenance marker",
    )


def _infer_stage(*, path_pattern: str, relative_path: str) -> tuple[str | None, str]:
    lower_path = path_pattern.lower()
    if "/quality_reports/" in lower_path:
        return "detect_quality", "literal path contains a detection quality report"
    path_families = (
        ("refined_subject_masks_runs", "refined_subject_masks"),
        ("subject_mask_runs", "subject_masks"),
        ("refined_eye_masks_runs", "refined_eye_masks"),
        ("eye_masks_runs", "eye_masks"),
        ("refined_keypoints_runs", "refined_keypoints"),
        ("keypoints_runs", "keypoints"),
        ("refined_detect_runs", "refined_detect"),
        ("detect_quality_runs", "detect_quality"),
        ("detection_artifact_runs", "detect"),
        ("detect_runs", "detect"),
        ("crop_runs", "crop"),
        ("tracking_runs", "tracking"),
        ("stimulus_runs", "stimulus"),
    )
    for family, stage in path_families:
        if family in lower_path:
            return stage, f"literal family path contains {family}"
    lower_file = relative_path.lower()
    if "refined_subject_mask" in lower_file:
        return (
            "refined_subject_masks",
            "writer module is specific to refined subject masks",
        )
    if "subject_mask" in lower_file and "training" not in lower_file:
        return "subject_masks", "writer module is specific to subject masks"
    if "refine_keypoint" in lower_file:
        return "refined_keypoints", "writer module is specific to refined keypoints"
    if "refined_detect" in lower_file:
        return "refined_detect", "writer module is specific to refined detections"
    if "/refinement/detect_quality" in lower_file:
        return "detect_quality", "writer module is specific to detection quality"
    if "/detection/" in lower_file or lower_file.endswith("/predict_detections.py"):
        return "detect", "writer module is specific to raw detections"
    return None, "stage is not statically bound at this writer call"


def _semantic_defaults(
    *,
    name: str | None,
    stage: str | None,
    surface_class: str,
) -> tuple[str, str, str, str]:
    lower_name = (name or "").lower()
    lower_stage = (stage or "").lower()
    if any(
        token in lower_name
        for token in ("masks_roi", "mask_probs_roi", "roi_images", "images_")
    ):
        access = "per_row"
        access_basis = "pixel payload name"
    elif (
        lower_name in {"ptr", "len", "points_xy", "counts", "values"}
        or "contour" in lower_name
        or "rle" in lower_name
    ):
        access = "indexed"
        access_basis = "flat payload/index naming"
    elif any(
        token in lower_name
        for token in (
            "keypoint",
            "frame_",
            "row_id",
            "row_index",
            "indices",
            "coordinates",
            "timeline",
            "time_seconds",
        )
    ):
        access = "windowed"
        access_basis = "row/timeline/coordinate naming"
    else:
        access = "unclassified"
        access_basis = "no high-confidence static access rule"

    if surface_class in {"publication", "compaction"}:
        lifecycle = "immutable_publication"
        lifecycle_basis = "publication/compaction surface"
    elif surface_class == "training":
        lifecycle = "training_build_then_immutable"
        lifecycle_basis = "training artifact surface"
    elif surface_class == "migration_or_maintenance":
        lifecycle = "maintenance_mutation"
        lifecycle_basis = "maintenance surface"
    elif lower_stage in {"refined_keypoints", "refined_subject_masks"}:
        lifecycle = "editable_random_update"
        lifecycle_basis = "declared editable authority stage"
    elif surface_class == "production":
        lifecycle = "append_or_build_then_publish"
        lifecycle_basis = "production output; exact write mode unverified"
    else:
        lifecycle = "unclassified"
        lifecycle_basis = "surface does not imply archive lifecycle"
    return access, access_basis, lifecycle, lifecycle_basis


def _consumer(name: str | None, surface_class: str) -> tuple[str, str]:
    lower = (name or "").lower()
    if surface_class == "training":
        return "palette_training", "training writer/reader surface"
    if any(
        token in lower
        for token in (
            "keypoint",
            "mask",
            "contour",
            "frame_counts",
            "frame_offsets",
            "frame_row_offsets",
            "frame_alignment",
        )
    ):
        return (
            "palette_and_probable_crimson",
            "name is in the known viewer-facing archive families; verify per path",
        )
    return "unclassified", "consumer is not provable from static creation/declaration"


_CONTRACT_SIGNATURES: Mapping[str, tuple[str, tuple[object, ...], str]] = {
    "frame_counts": ("int32", ("n_frames",), "palette.array.frame_counts@1"),
    "frame_offsets": ("int64", ("n_frame_offsets",), "palette.array.frame_offsets@1"),
    "frame_row_offsets": (
        "int64",
        ("n_frame_offsets",),
        "palette.array.frame_offsets@1",
    ),
    "keypoints_roi": (
        "float64",
        ("n_rois", "n_keypoints", 2),
        "palette.array.keypoints_roi@1",
    ),
    "keypoints_img": (
        "float64",
        ("n_rois", "n_keypoints", 2),
        "palette.array.keypoints_img@1",
    ),
    "keypoints_norm": (
        "float64",
        ("n_rois", "n_keypoints", 2),
        "palette.array.keypoints_norm@1",
    ),
    "masks_roi": (
        "uint8",
        ("n_rois", "n_channels", "H", "W"),
        "palette.array.subject_masks_roi_dense@1",
    ),
    "points_xy": ("float32", ("n_points", 2), "palette.array.contour_points_xy@1"),
}


def _shape_compatible(observed: Sequence[object], expected: Sequence[object]) -> bool:
    if len(observed) != len(expected):
        return False
    for actual, wanted in zip(observed, expected):
        if isinstance(wanted, int) and actual != wanted:
            return False
        if isinstance(wanted, str) and isinstance(actual, int):
            continue
        if isinstance(wanted, str) and isinstance(actual, str):
            if wanted in {"H", "W"} and actual != wanted:
                return False
    return True


def _contract_mapping(
    name: str | None,
    dtype: str | None,
    shape: Sequence[object] | None,
    *,
    stage: str | None,
) -> dict[str, object]:
    if name not in _CONTRACT_SIGNATURES:
        return {
            "canonical_contract_id": None,
            "canonical_contract_version": None,
            "contract_mapping_status": "unmapped",
            "contract_mapping_basis": "no initial catalog candidate for this leaf name",
        }
    expected_dtype, expected_shape, contract = _CONTRACT_SIGNATURES[name]
    contract_id, version_text = contract.rsplit("@", 1)
    if name == "masks_roi" and stage not in {
        "subject_masks",
        "refined_subject_masks",
        None,
    }:
        return {
            "canonical_contract_id": contract_id,
            "canonical_contract_version": int(version_text),
            "contract_mapping_status": "candidate",
            "contract_mapping_basis": "dense mask leaf matches, but this is not a canonical subject-mask stage",
        }
    if (
        dtype == expected_dtype
        and shape is not None
        and _shape_compatible(shape, expected_shape)
    ):
        return {
            "canonical_contract_id": contract_id,
            "canonical_contract_version": int(version_text),
            "contract_mapping_status": "exact",
            "contract_mapping_basis": "leaf name, exact dtype, rank, and fixed axes match initial contract",
        }
    return {
        "canonical_contract_id": contract_id,
        "canonical_contract_version": int(version_text),
        "contract_mapping_status": "candidate",
        "contract_mapping_basis": "leaf name matches, but dtype or shape is unresolved/different",
    }


def _array_spec_calls(
    expression: ast.AST | None,
    assignments: Mapping[str, ast.AST],
    *,
    seen: frozenset[str] = frozenset(),
) -> list[ast.Call]:
    if expression is None:
        return []
    if isinstance(expression, ast.Name) and expression.id not in seen:
        target = assignments.get(expression.id)
        if target is not None:
            return _array_spec_calls(target, assignments, seen=seen | {expression.id})
    if isinstance(expression, ast.Starred):
        return _array_spec_calls(expression.value, assignments, seen=seen)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return [
            *_array_spec_calls(expression.left, assignments, seen=seen),
            *_array_spec_calls(expression.right, assignments, seen=seen),
        ]
    if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
        return [
            call
            for item in expression.elts
            for call in _array_spec_calls(item, assignments, seen=seen)
        ]
    if isinstance(expression, ast.Call) and _call_leaf(expression) == "ArraySpec":
        return [expression]
    return []


def _call_argument(call: ast.Call, index: int, keyword: str) -> ast.AST | None:
    value = _keyword(call, keyword)
    if value is not None:
        return value
    return call.args[index] if index < len(call.args) else None


def _parse_array_spec(call: ast.Call, module: ModuleInfo) -> dict[str, object]:
    name_node = _call_argument(call, 0, "name")
    dtype_node = _call_argument(call, 1, "dtype")
    shape_node = _call_argument(call, 2, "shape_template")
    required_node = _call_argument(call, 3, "required")
    description_node = _call_argument(call, 4, "description")
    name = _literal(name_node, module.constants)
    dtype = _literal(dtype_node, module.constants)
    shape = _literal(shape_node, module.constants)
    required = _literal(required_node, module.constants)
    description = _literal(description_node, module.constants)
    return {
        "array_name": name if isinstance(name, str) else None,
        "array_name_expression": _source_text(name_node),
        "dynamic_name": not isinstance(name, str),
        "dtype": dtype if isinstance(dtype, str) else None,
        "dtype_expression": _source_text(dtype_node),
        "shape_template": list(shape) if isinstance(shape, tuple) else None,
        "shape_expression": _source_text(shape_node),
        "required": True if required is None else bool(required),
        "description": description if isinstance(description, str) else "",
    }


def _stage_spec_bindings(
    module: ModuleInfo,
) -> tuple[list[dict[str, object]], set[int]]:
    records: list[dict[str, object]] = []
    bound_call_ids: set[int] = set()
    for assignment_name, expression in module.module_assignments.items():
        if (
            not isinstance(expression, ast.Call)
            or _call_leaf(expression) != "StageSpec"
        ):
            continue
        stage_node = _call_argument(expression, 0, "stage_name")
        group_node = _call_argument(expression, 1, "zarr_group")
        specs_node = _call_argument(expression, 2, "specs")
        subgroups_node = _call_argument(expression, 3, "subgroups")
        stage = _literal(stage_node, module.constants)
        group = _literal(group_node, module.constants)
        stage_name = stage if isinstance(stage, str) else assignment_name
        group_path = group if isinstance(group, str) else f"<stage:{stage_name}>/"

        bindings: list[tuple[str | None, ast.Call]] = [
            (None, call)
            for call in _array_spec_calls(specs_node, module.module_assignments)
        ]
        subgroup_expr = subgroups_node
        if isinstance(subgroup_expr, ast.Name):
            subgroup_expr = module.module_assignments.get(
                subgroup_expr.id, subgroup_expr
            )
        if isinstance(subgroup_expr, ast.Dict):
            for key, value in zip(subgroup_expr.keys, subgroup_expr.values):
                subgroup = _literal(key, module.constants)
                subgroup_text = (
                    subgroup if isinstance(subgroup, str) else f"<{_source_text(key)}>"
                )
                bindings.extend(
                    (subgroup_text, call)
                    for call in _array_spec_calls(value, module.module_assignments)
                )
        for subgroup, call in bindings:
            bound_call_ids.add(id(call))
            parsed = _parse_array_spec(call, module)
            name = parsed["array_name"]
            path = group_path.rstrip("/")
            if subgroup:
                path = f"{path}/{subgroup.strip('/')}"
            path = f"{path}/{name or '<dynamic-array-name>'}"
            description = str(parsed["description"] or "")
            dtype = parsed["dtype"] if isinstance(parsed["dtype"], str) else None
            status = "current"
            status_basis = "current StageSpec declaration"
            if stage_name in _LEGACY_STAGE_NAMES or "legacy" in description.lower():
                status = "legacy_only"
                status_basis = "legacy compatibility stage/description"
            elif dtype and "/" in dtype:
                status = "unclear"
                status_basis = (
                    "compatibility dtype union is not one exact current contract"
                )
            access, access_basis, lifecycle, lifecycle_basis = _semantic_defaults(
                name=name if isinstance(name, str) else None,
                stage=stage_name,
                surface_class="production",
            )
            consumer, consumer_basis = _consumer(
                name if isinstance(name, str) else None, "production"
            )
            record = {
                "occurrence_id": _stable_id(
                    "schema",
                    module.relative_path,
                    call.lineno,
                    assignment_name,
                    subgroup,
                ),
                "source_kind": "array_spec_stage_binding",
                "file": module.relative_path,
                "line": int(call.lineno),
                "declaration_symbol": assignment_name,
                "declaring_stage": stage_name,
                "declaring_stage_basis": "bound by StageSpec",
                "stage_spec_symbol": assignment_name,
                "subgroup": subgroup,
                "path_pattern": path,
                **parsed,
                "status": status,
                "status_basis": status_basis,
                "access_pattern": access,
                "access_pattern_basis": access_basis,
                "write_lifecycle": lifecycle,
                "write_lifecycle_basis": lifecycle_basis,
                "consumer": consumer,
                "consumer_basis": consumer_basis,
            }
            record.update(
                _contract_mapping(
                    (
                        record["array_name"]
                        if isinstance(record["array_name"], str)
                        else None
                    ),
                    dtype,
                    (
                        record["shape_template"]
                        if isinstance(record["shape_template"], list)
                        else None
                    ),
                    stage=stage_name,
                )
            )
            records.append(record)
    return records, bound_call_ids


def _declaration_owners(module: ModuleInfo) -> dict[int, str]:
    owners: dict[int, str] = {}
    for name, expression in module.module_assignments.items():
        for call in _array_spec_calls(expression, module.module_assignments):
            owners.setdefault(id(call), name)
    return owners


def _schema_declarations(modules: Sequence[ModuleInfo]) -> list[dict[str, object]]:
    by_path = {module.relative_path: module for module in modules}
    records: list[dict[str, object]] = []
    for relative in ARRAY_SPEC_FILES:
        module = by_path.get(relative.as_posix())
        if module is None:
            raise FileNotFoundError(f"Missing ArraySpec source: {relative}")
        stage_records, bound = _stage_spec_bindings(module)
        records.extend(stage_records)
        owners = _declaration_owners(module)
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Call) or _call_leaf(node) != "ArraySpec":
                continue
            if id(node) in bound:
                continue
            parsed = _parse_array_spec(node, module)
            owner = owners.get(id(node), "<inline>")
            name = (
                parsed["array_name"] if isinstance(parsed["array_name"], str) else None
            )
            dtype = parsed["dtype"] if isinstance(parsed["dtype"], str) else None
            description = str(parsed["description"] or "")
            status = "legacy_only" if "legacy" in description.lower() else "current"
            status_basis = (
                "description marks compatibility/legacy"
                if status == "legacy_only"
                else "active shared logical declaration without a StageSpec path binding"
            )
            if dtype and "/" in dtype:
                status = "unclear"
                status_basis = (
                    "compatibility dtype union is not one exact current contract"
                )
            access, access_basis, lifecycle, lifecycle_basis = _semantic_defaults(
                name=name,
                stage=None,
                surface_class="production",
            )
            consumer, consumer_basis = _consumer(name, "production")
            path = f"<declaration:{owner}>/{name or '<dynamic-array-name>'}"
            record = {
                "occurrence_id": _stable_id(
                    "schema", module.relative_path, node.lineno, owner, "standalone"
                ),
                "source_kind": "array_spec_unbound_declaration",
                "file": module.relative_path,
                "line": int(node.lineno),
                "declaration_symbol": owner,
                "declaring_stage": None,
                "declaring_stage_basis": "standalone ArraySpec tuple has no StageSpec path binding",
                "stage_spec_symbol": None,
                "subgroup": None,
                "path_pattern": path,
                **parsed,
                "status": status,
                "status_basis": status_basis,
                "access_pattern": access,
                "access_pattern_basis": access_basis,
                "write_lifecycle": lifecycle,
                "write_lifecycle_basis": lifecycle_basis,
                "consumer": consumer,
                "consumer_basis": consumer_basis,
            }
            record.update(
                _contract_mapping(
                    name,
                    dtype,
                    (
                        record["shape_template"]
                        if isinstance(record["shape_template"], list)
                        else None
                    ),
                    stage=None,
                )
            )
            records.append(record)
    return sorted(
        records,
        key=lambda row: (str(row["file"]), int(row["line"]), str(row["path_pattern"])),
    )


def _required_array_path_and_stage(
    declaration_name: str, array_name: str
) -> tuple[str, str | None]:
    lower = declaration_name.lower()
    if "/" in array_name:
        stage, _ = _infer_stage(
            path_pattern=array_name,
            relative_path="",
        )
        return array_name, stage
    if "crop" in lower:
        return f"crop_runs/<run>/{array_name}", "crop"
    if "keypoint" in lower or "_kp_" in lower:
        return f"keypoints_runs/<run>/{array_name}", "keypoints"
    if "mask" in lower:
        return f"subject_mask_runs/<run>/{array_name}", "subject_masks"
    if "source" in lower and "detection" not in lower and "path" not in lower:
        return f"source_index/{array_name}", None
    return f"<required-array-set:{declaration_name}>/{array_name}", None


def _required_array_declarations(
    modules: Sequence[ModuleInfo],
) -> list[dict[str, object]]:
    """Collect reader/validator array-name tuples, notably training schemas."""

    records: list[dict[str, object]] = []
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            for declaration_name in names:
                lower = declaration_name.lower()
                if "required" not in lower or "array" not in lower:
                    continue
                values = _literal(node.value, module.constants)
                if not isinstance(values, tuple) or not values:
                    continue
                if not all(isinstance(value, str) for value in values):
                    continue
                surface, status, status_basis = _surface_class(module.relative_path)
                for array_name in values:
                    assert isinstance(array_name, str)
                    path, stage = _required_array_path_and_stage(
                        declaration_name, array_name
                    )
                    stage_basis = (
                        "required-array declaration name identifies a stage family"
                    )
                    if stage is None:
                        inferred_stage, inferred_basis = _infer_stage(
                            path_pattern=path,
                            relative_path=module.relative_path,
                        )
                        stage = inferred_stage
                        stage_basis = inferred_basis
                    access, access_basis, lifecycle, lifecycle_basis = (
                        _semantic_defaults(
                            name=array_name,
                            stage=stage,
                            surface_class=surface,
                        )
                    )
                    consumer, consumer_basis = _consumer(array_name, surface)
                    record: dict[str, object] = {
                        "occurrence_id": _stable_id(
                            "schema-required",
                            module.relative_path,
                            node.lineno,
                            declaration_name,
                            array_name,
                        ),
                        "source_kind": "reader_required_array_declaration",
                        "file": module.relative_path,
                        "line": int(node.lineno),
                        "declaration_symbol": declaration_name,
                        "declaring_stage": stage,
                        "declaring_stage_basis": stage_basis,
                        "stage_spec_symbol": None,
                        "subgroup": None,
                        "path_pattern": path,
                        "array_name": array_name,
                        "array_name_expression": repr(array_name),
                        "dynamic_name": False,
                        "dtype": None,
                        "dtype_expression": None,
                        "shape_template": None,
                        "shape_expression": None,
                        "required": True,
                        "description": "Reader/validator required-array name; dtype and shape are not declared here.",
                        "status": status,
                        "status_basis": status_basis,
                        "access_pattern": access,
                        "access_pattern_basis": access_basis,
                        "write_lifecycle": lifecycle,
                        "write_lifecycle_basis": lifecycle_basis,
                        "consumer": consumer,
                        "consumer_basis": consumer_basis,
                    }
                    record.update(
                        _contract_mapping(
                            array_name,
                            None,
                            None,
                            stage=stage,
                        )
                    )
                    records.append(record)
    return sorted(
        records,
        key=lambda row: (
            str(row["file"]),
            int(row["line"]),
            str(row["path_pattern"]),
        ),
    )


def _wrapper_name_data_shape(
    call: CallInfo,
    function: FunctionInfo | None,
) -> tuple[ast.AST | None, ast.AST | None, ast.AST | None, ast.AST | None]:
    name_node = _argument_for_parameter(call.node, function, NAME_PARAMETER_CANDIDATES)
    data_node = _argument_for_parameter(call.node, function, DATA_PARAMETER_CANDIDATES)
    shape_node = _argument_for_parameter(call.node, function, ("shape",))
    dtype_node = _argument_for_parameter(call.node, function, ("dtype",))
    return name_node, data_node, shape_node, dtype_node


def _receiver_node(call: ast.Call) -> ast.AST | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.value
    return None


def _writer_record(
    call: CallInfo,
    *,
    call_kind: str,
    writer_symbol: str | None,
    function: FunctionInfo | None,
) -> dict[str, object]:
    if call_kind == "direct_zarr_api":
        name_node = _direct_name_node(call.node)
        data_node = _keyword(call.node, "data")
        shape_node = _keyword(call.node, "shape")
        dtype_node = _keyword(call.node, "dtype")
        receiver = _receiver_node(call.node)
        parent_path = _path_expression(receiver, call)
    else:
        name_node, data_node, shape_node, dtype_node = _wrapper_name_data_shape(
            call, function
        )
        if function is not None and function.parameters:
            parent_candidates = (
                "group",
                "parent",
                "run",
                "run_group",
                "root",
                "destination_root",
            )
            parent_node = _argument_for_parameter(
                call.node, function, parent_candidates
            )
        else:
            parent_node = call.node.args[0] if call.node.args else None
        parent_path = _path_expression(parent_node, call)

    name, dynamic_name = _resolve_name(name_node, call)
    name_expression = _source_text(name_node)
    if name:
        path_pattern = f"{parent_path.rstrip('/')}/{name.strip('/')}"
    elif name_node is not None:
        path_pattern = f"{parent_path.rstrip('/')}/<{name_expression}>"
    else:
        path_pattern = f"{parent_path.rstrip('/')}/<dynamic-arrays>"

    explicit_dtype = _dtype_token(dtype_node)
    dtype = explicit_dtype or _infer_dtype(data_node, call)
    explicit_shape = _literal(shape_node, call.module.constants)
    shape_template = list(explicit_shape) if isinstance(explicit_shape, tuple) else None
    shape_expression = (
        _source_text(shape_node)
        if shape_node is not None
        else _infer_shape_expression(data_node, call)
    )
    surface, status, status_basis = _surface_class(call.module.relative_path)
    stage, stage_basis = _infer_stage(
        path_pattern=path_pattern,
        relative_path=call.module.relative_path,
    )
    access, access_basis, lifecycle, lifecycle_basis = _semantic_defaults(
        name=name,
        stage=stage,
        surface_class=surface,
    )
    consumer, consumer_basis = _consumer(name, surface)
    method = _call_leaf(call.node)
    record: dict[str, object] = {
        "site_id": _stable_id(
            "writer",
            call.module.relative_path,
            call.node.lineno,
            call_kind,
            writer_symbol,
            name_expression,
        ),
        "file": call.module.relative_path,
        "line": int(call.node.lineno),
        "enclosing_symbol": call.enclosing_symbol,
        "call_kind": call_kind,
        "api_method": method,
        "writer_symbol": writer_symbol,
        "receiver_expression": _source_text(_receiver_node(call.node)),
        "parent_path_pattern": parent_path,
        "path_pattern": path_pattern,
        "declaring_stage": stage,
        "declaring_stage_basis": stage_basis,
        "array_name": name,
        "array_name_expression": name_expression,
        "dynamic_name": dynamic_name or name_node is None,
        "data_expression": _source_text(data_node),
        "dtype": dtype,
        "dtype_expression": _source_text(dtype_node),
        "shape_template": shape_template,
        "shape_expression": shape_expression,
        "chunks_expression": _source_text(_keyword(call.node, "chunks")),
        "shards_expression": _source_text(_keyword(call.node, "shards")),
        "compressor_expression": _source_text(_keyword(call.node, "compressor")),
        "compressors_expression": _source_text(_keyword(call.node, "compressors")),
        "filters_expression": _source_text(_keyword(call.node, "filters")),
        "serializer_expression": _source_text(_keyword(call.node, "serializer")),
        "zarr_format_expression": _source_text(_keyword(call.node, "zarr_format")),
        "surface_class": surface,
        "status": status,
        "status_basis": status_basis,
        "access_pattern": access,
        "access_pattern_basis": access_basis,
        "write_lifecycle": lifecycle,
        "write_lifecycle_basis": lifecycle_basis,
        "consumer": consumer,
        "consumer_basis": consumer_basis,
    }
    record.update(_contract_mapping(name, dtype, shape_template, stage=stage))
    return record


def _manual_metadata_records(calls: Sequence[CallInfo]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for call in calls:
        leaf = _call_leaf(call.node)
        if leaf not in {"_atomic_write_json", "write_text"}:
            continue
        arguments = [*call.node.args, *(item.value for item in call.node.keywords)]
        texts = [_source_text(argument) or "" for argument in arguments]
        if not any("zarr.json" in text for text in texts):
            continue
        path_node = call.node.args[0] if call.node.args else None
        path_pattern = _path_expression(path_node, call)
        name = (
            "encoded_global_masks_roi"
            if "refined_subject_mask_encoded_chunks.py" in call.module.relative_path
            else None
        )
        surface, status, status_basis = _surface_class(call.module.relative_path)
        access, access_basis, lifecycle, lifecycle_basis = _semantic_defaults(
            name=name, stage="refined_subject_masks", surface_class=surface
        )
        if name == "encoded_global_masks_roi":
            lifecycle = "build_then_immutable_derived_cache"
            lifecycle_basis = (
                "encoded clip-package payload is a derived immutable cache"
            )
        consumer, consumer_basis = _consumer(name, surface)
        record: dict[str, object] = {
            "site_id": _stable_id(
                "writer",
                call.module.relative_path,
                call.node.lineno,
                "manual_zarr_metadata",
            ),
            "file": call.module.relative_path,
            "line": int(call.node.lineno),
            "enclosing_symbol": call.enclosing_symbol,
            "call_kind": "manual_zarr_metadata",
            "api_method": leaf,
            "writer_symbol": call.enclosing_symbol,
            "receiver_expression": None,
            "parent_path_pattern": path_pattern.removesuffix("/zarr.json"),
            "path_pattern": path_pattern.removesuffix("/zarr.json"),
            "declaring_stage": "refined_subject_masks",
            "declaring_stage_basis": "manual payload copies refined_subject_masks/masks_roi chunks",
            "array_name": name,
            "array_name_expression": "ENCODED_MASK_PAYLOAD_NAME" if name else None,
            "dynamic_name": name is None,
            "data_expression": (
                _source_text(call.node.args[1]) if len(call.node.args) > 1 else None
            ),
            "dtype": "uint8" if name else None,
            "dtype_expression": "manual Zarr v3 metadata data_type",
            "shape_template": ["n_rois", "n_channels", "H", "W"] if name else None,
            "shape_expression": "payload metadata copied from masks_roi and global grid",
            "chunks_expression": (
                "dense_mask_chunks from global-grid manifest" if name else None
            ),
            "shards_expression": None,
            "compressor_expression": None,
            "compressors_expression": (
                "bytes + zstd(level=0, checksum=False)" if name else None
            ),
            "filters_expression": None,
            "serializer_expression": None,
            "zarr_format_expression": "3" if name else None,
            "surface_class": surface,
            "status": status,
            "status_basis": status_basis,
            "access_pattern": access,
            "access_pattern_basis": access_basis,
            "write_lifecycle": lifecycle,
            "write_lifecycle_basis": lifecycle_basis,
            "consumer": consumer,
            "consumer_basis": consumer_basis,
            "manual_writer_note": "Creates an ordinary Zarr array by writing zarr.json before encoded chunk objects.",
        }
        record.update(
            _contract_mapping(
                name,
                "uint8" if name else None,
                (
                    record["shape_template"]
                    if isinstance(record["shape_template"], list)
                    else None
                ),
                stage="refined_subject_masks",
            )
        )
        records.append(record)
    return records


def _writer_records(
    modules: Sequence[ModuleInfo],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    functions, calls = _collect_functions_and_calls(modules)
    by_symbol = {function.symbol: function for function in functions}
    writer_symbols, resolved = _writer_function_symbols(functions, calls)
    records: list[dict[str, object]] = []
    for call in calls:
        if _is_direct_array_creation(call):
            records.append(
                _writer_record(
                    call,
                    call_kind="direct_zarr_api",
                    writer_symbol=call.enclosing_symbol,
                    function=None,
                )
            )
        callee = resolved[id(call.node)]
        callee_function = by_symbol.get(callee) if callee is not None else None
        reportable_wrapper = callee_function is not None and (
            any(
                parameter in NAME_PARAMETER_CANDIDATES
                for parameter in callee_function.parameters
            )
            or callee_function.leaf_name in MULTI_ARRAY_WRAPPER_NAMES
        )
        if callee in writer_symbols and reportable_wrapper:
            records.append(
                _writer_record(
                    call,
                    call_kind="writer_wrapper_call",
                    writer_symbol=callee,
                    function=callee_function,
                )
            )
    records.extend(_manual_metadata_records(calls))

    unique: dict[str, dict[str, object]] = {}
    for record in records:
        unique[str(record["site_id"])] = record
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            str(row["file"]),
            int(row["line"]),
            str(row["call_kind"]),
            str(row["path_pattern"]),
        ),
    )
    metadata = {
        "function_definition_count": len(functions),
        "writer_function_count": len(writer_symbols),
        "writer_functions": sorted(writer_symbols),
    }
    return ordered, metadata


def _writer_schema_occurrences(
    writers: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for writer in writers:
        name = writer.get("array_name")
        if not isinstance(name, str):
            continue
        record = {
            "occurrence_id": _stable_id("schema-writer", writer["site_id"]),
            "source_kind": "writer_observed_schema",
            "file": writer["file"],
            "line": writer["line"],
            "declaration_symbol": writer["enclosing_symbol"],
            "declaring_stage": writer["declaring_stage"],
            "declaring_stage_basis": writer["declaring_stage_basis"],
            "stage_spec_symbol": None,
            "subgroup": None,
            "path_pattern": writer["path_pattern"],
            "array_name": name,
            "array_name_expression": writer["array_name_expression"],
            "dynamic_name": writer["dynamic_name"],
            "dtype": writer["dtype"],
            "dtype_expression": writer["dtype_expression"],
            "shape_template": writer["shape_template"],
            "shape_expression": writer["shape_expression"],
            "required": None,
            "description": "Schema evidence recovered from a physical writer call site.",
            "status": writer["status"],
            "status_basis": writer["status_basis"],
            "access_pattern": writer["access_pattern"],
            "access_pattern_basis": writer["access_pattern_basis"],
            "write_lifecycle": writer["write_lifecycle"],
            "write_lifecycle_basis": writer["write_lifecycle_basis"],
            "consumer": writer["consumer"],
            "consumer_basis": writer["consumer_basis"],
            "canonical_contract_id": writer["canonical_contract_id"],
            "canonical_contract_version": writer["canonical_contract_version"],
            "contract_mapping_status": writer["contract_mapping_status"],
            "contract_mapping_basis": writer["contract_mapping_basis"],
            "writer_site_id": writer["site_id"],
        }
        records.append(record)
    return records


def _signature(record: Mapping[str, object]) -> tuple[str, str, str]:
    dtype = str(record.get("dtype") or "<unresolved>")
    shape = json.dumps(
        record.get("shape_template")
        or record.get("shape_expression")
        or "<unresolved>",
        sort_keys=True,
    )
    return str(record.get("array_name") or "<dynamic>"), dtype, shape


def _ambiguities(
    records: Sequence[Mapping[str, object]],
    *,
    source_kinds: frozenset[str] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[tuple[str, str], list[str]]] = {}
    for record in records:
        if source_kinds is not None and record.get("source_kind") not in source_kinds:
            continue
        name = record.get("array_name")
        if not isinstance(name, str):
            continue
        _, dtype, shape = _signature(record)
        grouped.setdefault(name, {}).setdefault((dtype, shape), []).append(
            str(record["occurrence_id"])
        )
    result: list[dict[str, object]] = []
    for name, signatures in sorted(grouped.items()):
        if len(signatures) <= 1:
            continue
        result.append(
            {
                "array_name": name,
                "signature_count": len(signatures),
                "review_status": "unresolved",
                "review_question": "same semantics, different semantics, or legacy variation?",
                "signatures": [
                    {
                        "dtype": dtype,
                        "shape": (
                            json.loads(shape) if shape.startswith(("[", '"')) else shape
                        ),
                        "occurrence_ids": sorted(ids),
                    }
                    for (dtype, shape), ids in sorted(signatures.items())
                ],
            }
        )
    return result


def _counts_by(records: Iterable[Mapping[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "<none>")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_census(repo_root: str | Path) -> dict[str, object]:
    """Build the deterministic in-memory census without modifying the repo."""

    root = Path(repo_root).expanduser().resolve()
    modules, scanned_module_count = _load_modules(root)
    declared = _schema_declarations(modules)
    required_array_declarations = _required_array_declarations(modules)
    writers, writer_metadata = _writer_records(modules)
    writer_schemas = _writer_schema_occurrences(writers)
    schemas = sorted(
        [*declared, *required_array_declarations, *writer_schemas],
        key=lambda row: (
            str(row["file"]),
            int(row["line"]),
            str(row["source_kind"]),
            str(row["path_pattern"]),
        ),
    )

    declared_names = {
        str(record["array_name"])
        for record in declared
        if isinstance(record.get("array_name"), str)
    }
    writer_names = {
        str(record["array_name"])
        for record in writers
        if isinstance(record.get("array_name"), str)
    }
    array_spec_ambiguities = _ambiguities(
        declared,
        source_kinds=frozenset(
            {"array_spec_stage_binding", "array_spec_unbound_declaration"}
        ),
    )
    all_ambiguities = _ambiguities(schemas)
    coverage = {
        "declared_leaf_name_count": len(declared_names),
        "writer_leaf_name_count": len(writer_names),
        "declared_and_written_leaf_name_count": len(declared_names & writer_names),
        "declared_without_resolved_writer_leaf_names": sorted(
            declared_names - writer_names
        ),
        "writer_without_array_spec_leaf_names": sorted(writer_names - declared_names),
        "note": "Leaf-name coverage is orientation evidence only; dynamic paths and semantic reuse require manual review.",
    }
    summary = {
        "scanned_python_module_count": scanned_module_count,
        "parsed_writer_candidate_module_count": len(modules),
        "array_spec_declaration_count": len(declared),
        "array_spec_unique_leaf_name_count": len(declared_names),
        "array_spec_unique_signature_count": len({_signature(row) for row in declared}),
        "array_spec_ambiguous_leaf_name_count": len(array_spec_ambiguities),
        "reader_required_array_declaration_count": len(required_array_declarations),
        "writer_site_count": len(writers),
        "direct_zarr_api_site_count": sum(
            row["call_kind"] == "direct_zarr_api" for row in writers
        ),
        "writer_wrapper_call_site_count": sum(
            row["call_kind"] == "writer_wrapper_call" for row in writers
        ),
        "manual_zarr_metadata_site_count": sum(
            row["call_kind"] == "manual_zarr_metadata" for row in writers
        ),
        "resolved_writer_leaf_name_count": len(writer_names),
        "dynamic_or_unresolved_writer_site_count": sum(
            not isinstance(row.get("array_name"), str) for row in writers
        ),
        "writer_site_with_stage_classification_count": sum(
            isinstance(row.get("declaring_stage"), str) for row in writers
        ),
        "writer_site_without_dtype_evidence_count": sum(
            row.get("dtype") is None and row.get("dtype_expression") is None
            for row in writers
        ),
        "writer_site_without_shape_evidence_count": sum(
            row.get("shape_template") is None and row.get("shape_expression") is None
            for row in writers
        ),
        "writer_site_with_chunks_expression_count": sum(
            row.get("chunks_expression") is not None for row in writers
        ),
        "writer_site_with_shards_expression_count": sum(
            row.get("shards_expression") is not None for row in writers
        ),
        "writer_site_with_codec_expression_count": sum(
            any(
                row.get(key) is not None
                for key in (
                    "compressor_expression",
                    "compressors_expression",
                    "filters_expression",
                    "serializer_expression",
                )
            )
            for row in writers
        ),
        "writer_sites_by_surface": _counts_by(writers, "surface_class"),
        "writer_sites_by_status": _counts_by(writers, "status"),
        "schema_occurrences_by_source_kind": _counts_by(schemas, "source_kind"),
        "schema_occurrences_by_contract_mapping": _counts_by(
            schemas, "contract_mapping_status"
        ),
    }
    schema_document = {
        "schema_id": "palette.zarr_array_schema_census",
        "schema_version": 1,
        "generation_policy": "deterministic static AST census; no source execution",
        "scope": ["src/**/*.py", "scripts/**/*.py", "excluding tests"],
        "summary": summary,
        "coverage": coverage,
        "array_spec_ambiguities": array_spec_ambiguities,
        "all_observed_ambiguities": all_ambiguities,
        "occurrences": schemas,
    }
    writer_document = {
        "schema_id": "palette.zarr_production_writer_census",
        "schema_version": 1,
        "generation_policy": "deterministic static AST census; no source execution",
        "scope": [
            "direct Zarr array creation",
            "resolved writer-wrapper calls",
            "manual encoded-mask Zarr metadata creation",
            "production, training, publication, compaction, migration, diagnostic, benchmark, and legacy surfaces",
        ],
        "summary": summary,
        "wrapper_analysis": writer_metadata,
        "coverage": coverage,
        "sites": writers,
    }
    return {
        "schema_document": schema_document,
        "writer_document": writer_document,
        "detection_document": build_detection_document(
            schema_document,
            writer_document,
        ),
    }


def _markdown_table(
    rows: Sequence[Sequence[object]], headers: Sequence[str]
) -> list[str]:
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(escape(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(escape(item) for item in row) + " |" for row in rows
    )
    return output


def render_summary(census: Mapping[str, object]) -> str:
    """Render the review-oriented human summary for the generated JSON files."""

    schema = census["schema_document"]
    writer = census["writer_document"]
    assert isinstance(schema, Mapping) and isinstance(writer, Mapping)
    summary = schema["summary"]
    coverage = schema["coverage"]
    ambiguities = schema["array_spec_ambiguities"]
    assert isinstance(summary, Mapping) and isinstance(coverage, Mapping)
    assert isinstance(ambiguities, Sequence)
    lines = [
        "# Palette Zarr Schema and Writer Census",
        "",
        "Status: generated review baseline; contradictions intentionally unresolved",
        "",
        "This report is generated by `fisheye.diagnostics.zarr_storage_census`. "
        "It is a read-only static census: a record marked `unclassified`, `unclear`, "
        "or `candidate` is a prompt for review, not an inferred decision.",
        "",
        "## Ground Truth Surfaces",
        "",
        f"- `{summary['array_spec_declaration_count']}` `ArraySpec` occurrences, "
        f"`{summary['array_spec_unique_leaf_name_count']}` leaf names, and "
        f"`{summary['array_spec_unique_signature_count']}` `(name, dtype, shape)` signatures;",
        f"- `{summary['array_spec_ambiguous_leaf_name_count']}` `ArraySpec` leaf names with multiple signatures;",
        f"- `{summary['reader_required_array_declaration_count']}` additional reader/validator required-array declarations, including training validation schemas;",
        f"- `{summary['writer_site_count']}` writer sites: "
        f"`{summary['direct_zarr_api_site_count']}` direct APIs, "
        f"`{summary['writer_wrapper_call_site_count']}` resolved wrapper calls, and "
        f"`{summary['manual_zarr_metadata_site_count']}` manual metadata-created arrays;",
        f"- `{summary['dynamic_or_unresolved_writer_site_count']}` writer sites retain a dynamic or unresolved array name;",
        f"- `{coverage['writer_without_array_spec_leaf_names'].__len__()}` writer-observed leaf names have no `ArraySpec` declaration.",
        "",
        "The two JSON artifacts retain every occurrence and writer site. Repeated "
        "rows are intentional because a shared helper and each concrete caller are "
        "different migration and benchmark surfaces.",
        "",
        "## What `ArraySpec` Represents Today",
        "",
        "`ArraySpec` is existing logical validation evidence, not a complete storage "
        "contract. It carries a leaf name, free-form dtype string, symbolic shape "
        "template, required flag, and description. `StageSpec` can bind those fields "
        "to one stage path and fixed subgroups. Runtime validation checks presence, "
        "rank/dimensions, and broad dtype kind.",
        "",
        "It does not identify a canonical schema ID/version, exact cross-language "
        "dtype, named axes, units, fill/null semantics, physical chunks/shards/codecs, "
        "lifecycle, consumers, or the writer that creates the array. Compatibility "
        "dtype strings such as `float16/float32/uint8` are intentionally non-exact. "
        "Standalone analysis tuple declarations also lack a concrete archive path. "
        "For those reasons this census never treats an `ArraySpec` occurrence as "
        "proof that an array is currently written.",
        "",
        "## First Review Queue",
        "",
        f"- `{summary['schema_occurrences_by_source_kind']['array_spec_unbound_declaration']}` declarations are standalone tuples without a `StageSpec` path binding.",
        f"- `{summary['dynamic_or_unresolved_writer_site_count']}` writer sites still have a dynamic/unresolved array name.",
        f"- `{summary['writer_site_without_dtype_evidence_count']}` writer sites retain neither an inferred exact dtype nor an explicit dtype expression; `{summary['writer_site_without_shape_evidence_count']}` retain no static shape evidence.",
        f"- Only `{summary['writer_site_with_stage_classification_count']}` writer sites can currently be assigned to a stage from a literal family path or stage-specific module.",
        f"- Raw physical arguments are visible at `{summary['writer_site_with_chunks_expression_count']}` chunk sites, `{summary['writer_site_with_shards_expression_count']}` shard sites, and `{summary['writer_site_with_codec_expression_count']}` codec sites. These are call-site counts, not distinct arrays or effective runtime layouts.",
        "- The initial seven-contract catalog produces only "
        f"`{summary['schema_occurrences_by_contract_mapping'].get('exact', 0)}` exact occurrence mappings; this is expected until contradictions are reviewed and more contracts are accepted.",
        "",
        "## Writer Surface Counts",
        "",
    ]
    surface = summary["writer_sites_by_surface"]
    assert isinstance(surface, Mapping)
    lines.extend(
        _markdown_table(
            [(key, value) for key, value in surface.items()],
            ("Surface", "Writer sites"),
        )
    )
    lines.extend(
        [
            "",
            "## Declaration-to-Writer Coverage",
            "",
            f"- Declared leaf names: `{coverage['declared_leaf_name_count']}`",
            f"- Writer-observed leaf names: `{coverage['writer_leaf_name_count']}`",
            f"- Names present in both views: `{coverage['declared_and_written_leaf_name_count']}`",
            f"- Declared names without a statically resolved writer name: `{len(coverage['declared_without_resolved_writer_leaf_names'])}`",
            f"- Writer names absent from `ArraySpec`: `{len(coverage['writer_without_array_spec_leaf_names'])}`",
            "",
            "Dynamic helpers mean the first gap list contains false gaps until a "
            "human binds caller paths. Conversely, writer-only names are exactly the "
            "surface that a declaration-only migration would miss.",
            "",
            "## Ambiguous `ArraySpec` Leaf Names",
            "",
            "Each row is deliberately unresolved. The next design pass should label "
            "it as same semantics, different semantics, or legacy variation.",
            "",
        ]
    )
    ambiguity_rows: list[tuple[object, ...]] = []
    for item in ambiguities:
        assert isinstance(item, Mapping)
        signatures = item["signatures"]
        assert isinstance(signatures, Sequence)
        rendered = "; ".join(
            f"{signature['dtype']} {signature['shape']}"
            for signature in signatures
            if isinstance(signature, Mapping)
        )
        ambiguity_rows.append((item["array_name"], item["signature_count"], rendered))
    lines.extend(
        _markdown_table(
            ambiguity_rows,
            ("Leaf name", "Signatures", "Observed dtype/shape"),
        )
    )
    lines.extend(
        [
            "",
            "## Review Checklist",
            "",
            "- [ ] Review the ambiguous leaf-name rows one at a time; do not assign contract IDs by leaf name alone.",
            "- [ ] Bind standalone analysis tuple declarations to concrete path templates.",
            "- [ ] Review every `writer_without_array_spec_leaf_names` entry and decide whether it needs a canonical logical contract, a stage-local binding, or an explicit derived-cache/legacy designation.",
            "- [ ] Review dynamic writer sites and bind helper parameters to concrete callers.",
            "- [ ] Confirm consumers and access patterns against Crimson and Palette call sites; current heuristic labels include their basis.",
            "- [ ] Confirm editable, append-only, and immutable lifecycle per concrete path before applying `StoragePlan`.",
            "- [ ] Require every current writer path to have a benchmark case before enforcing the shared array factory.",
            "",
            "## Machine-Readable Artifacts",
            "",
            f"- `{SCHEMA_OUTPUT.as_posix()}` — declarations, writer-observed schemas, ambiguities, and coverage gaps.",
            f"- `{WRITER_OUTPUT.as_posix()}` — direct APIs, wrapper call sites, physical arguments, lifecycle/access/consumer classifications, and unresolved fields.",
            f"- `{DETECTION_OUTPUT.as_posix()}` — detection-owned variants, downstream lineage, conflicts, and dated physical observations.",
            f"- `{DETECTION_SUMMARY_OUTPUT.as_posix()}` — human review view of the detection-family inventory.",
            "",
        ]
    )
    return "\n".join(lines)


def write_census(repo_root: str | Path) -> dict[str, object]:
    """Regenerate the checked-in census artifacts and return the census."""

    root = Path(repo_root).expanduser().resolve()
    census = build_census(root)
    outputs = (
        (SCHEMA_OUTPUT, census["schema_document"]),
        (WRITER_OUTPUT, census["writer_document"]),
        (DETECTION_OUTPUT, census["detection_document"]),
    )
    for relative, payload in outputs:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary_path = root / SUMMARY_OUTPUT
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_summary(census), encoding="utf-8")
    detection_summary_path = root / DETECTION_SUMMARY_OUTPUT
    detection_summary_path.parent.mkdir(parents=True, exist_ok=True)
    detection_summary_path.write_text(
        render_detection_inventory(census["detection_document"]),
        encoding="utf-8",
    )
    return census


def _repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_repo_root_from_module())
    parser.add_argument(
        "--write",
        action="store_true",
        help="write deterministic JSON and Markdown artifacts under docs/diagnostics",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in artifacts differ from freshly generated content",
    )
    args = parser.parse_args(argv)
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")

    census = build_census(args.repo_root)
    if args.write:
        write_census(args.repo_root)
    elif args.check:
        expected = {
            SCHEMA_OUTPUT: json.dumps(
                census["schema_document"], indent=2, sort_keys=True
            )
            + "\n",
            WRITER_OUTPUT: json.dumps(
                census["writer_document"], indent=2, sort_keys=True
            )
            + "\n",
            SUMMARY_OUTPUT: render_summary(census),
            DETECTION_OUTPUT: json.dumps(
                census["detection_document"], indent=2, sort_keys=True
            )
            + "\n",
            DETECTION_SUMMARY_OUTPUT: render_detection_inventory(
                census["detection_document"]
            ),
        }
        stale = [
            relative.as_posix()
            for relative, content in expected.items()
            if not (args.repo_root / relative).is_file()
            or (args.repo_root / relative).read_text(encoding="utf-8") != content
        ]
        if stale:
            raise SystemExit("stale Zarr census artifacts: " + ", ".join(stale))
    else:
        print(
            json.dumps(census["schema_document"]["summary"], indent=2, sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
