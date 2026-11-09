# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
On-demand audit: stub parameters that are narrower than the binding accepts.

A ``.pyi`` parameter annotated ``list[T]`` rejects a tuple; ``dict[K, V]`` rejects
any other mapping. nanobind's STL casters accept both -- ``std::vector<T>`` casts
from any sequence except ``str``/``bytes``, and ``std::map`` casts through
``PyMapping_Items``. Where the stub says ``list`` and the binding means "any
sequence", the type checker rejects code that runs, and the author gets no hint
that the stub rather than their call is wrong.

This audit reads the signature **nanobind itself renders into each overload's
__doc__**, where ``std::vector<T>`` appears as ``collections.abc.Sequence[T]``
and a genuine ``nb::list`` parameter appears as plain ``list``. That is the
binding's own account of what it accepts, so the comparison needs no C++ parsing
and does not depend on the ``nanobind.stubgen`` version. The distinction matters:
``IRBuilder.emit_reg(kwargs)`` really is bound as ``const nb::dict&``, so ``dict``
is correct there and this audit must not flag it.

Run it against a built tree::

    PYTHONPATH=build/python/bindings python tests/lint/audit_stub_container_params.py

It exits non-zero when it finds anything, so it can be run manually or in a
one-off sweep. It is deliberately **not** a pre-commit hook: the repository
already carries 13 custom parity gates, and this failure mode is a latent
type-checker annoyance rather than a runtime defect, so it does not earn a
place on every commit. Re-run it after a binding change that alters a container
parameter.
"""

import argparse
import ast
import importlib
import re
import sys
from pathlib import Path

# Abstract type nanobind renders -> the concrete spelling that is too narrow.
# `Iterable` is included because a few casters render that way; the concrete
# counterpart a stub author would reach for is still `list`.
NARROWING = {
    "Sequence": "list",
    "Mapping": "dict",
    "Set": "set",
    "Iterable": "list",
}

# A rendered nanobind signature line, e.g.
#   `__init__(self, stmts: collections.abc.Sequence[...], span: ...) -> None`
# Overloads are numbered, so an optional leading `N. ` is allowed.
SIGNATURE_RE = re.compile(r"^\s*(?:\d+\.\s+)?(\w+)\((.*)\)\s*->\s*.+$")

DEFAULT_MODULES = ("ir", "passes", "arith", "sync", "codegen", "backend", "testing")


def split_top_level(text: str) -> list[str]:
    """Split a parameter list on commas that are not inside brackets."""
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts]


def rendered_overloads(member: object, name: str) -> list[dict[str, str]]:
    """Return ``[{parameter: rendered_type}]``, one dict per rendered overload."""
    overloads: list[dict[str, str]] = []
    for line in (getattr(member, "__doc__", "") or "").splitlines():
        match = SIGNATURE_RE.match(line)
        if match is None or match.group(1) != name:
            continue
        params: dict[str, str] = {}
        for part in split_top_level(match.group(2)):
            part = part.split("=")[0].strip()
            if ":" not in part:
                continue  # `self`, or a positional-only marker
            param, annotation = part.split(":", 1)
            params[param.strip()] = annotation.strip()
        overloads.append(params)
    return overloads


def import_core(name: str) -> object:
    """Import ``pypto_core.<name>``, however the extension is reachable.

    An installed tree exposes it as ``pypto.pypto_core``; a bare build tree on
    ``PYTHONPATH`` exposes it top-level as ``pypto_core``. Try the installed
    spelling first so the audit matches what consumers actually import.
    """
    for prefix in ("pypto.pypto_core", "pypto_core"):
        try:
            return importlib.import_module(f"{prefix}.{name}")
        except ImportError:
            continue
    raise ImportError(f"cannot import pypto_core.{name} under either prefix")


def resolve(module: object, qualname: str) -> object | None:
    """Walk a dotted stub qualname to the runtime object that implements it."""
    current = module
    for part in qualname.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def audit(module_name: str, stub_path: Path) -> list[tuple[str, str, str, str, int]]:
    """Return (qualname, parameter, stub annotation, rendered type, line)."""
    module = import_core(module_name)
    findings: list[tuple[str, str, str, str, int]] = []

    def visit(node: ast.ClassDef | ast.Module, prefix: str) -> None:
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef):
                inspect_function(child, prefix)

    def inspect_function(node: ast.FunctionDef, prefix: str) -> None:
        qualname = prefix + node.name
        member = resolve(module, qualname)
        if member is None:
            return  # stub-only symbol (TYPE_CHECKING alias, re-export, ...)
        for params in rendered_overloads(member, node.name):
            for arg in node.args.posonlyargs + node.args.args:
                if arg.annotation is None or arg.arg not in params:
                    continue
                rendered = params[arg.arg]
                declared = ast.unparse(arg.annotation)
                for abstract, concrete in NARROWING.items():
                    if f"{abstract}[" not in rendered and not rendered.endswith(abstract):
                        continue
                    # The stub is narrow whether the concrete container is the
                    # whole annotation or one member of a union.
                    if re.search(rf"\b{concrete}\[", declared) or declared == concrete:
                        findings.append((qualname, arg.arg, declared, rendered, arg.annotation.lineno))
                    break

    visit(ast.parse(stub_path.read_text(encoding="utf-8")), "")

    # One report per (qualname, parameter): a parameter repeated across overloads
    # is one place to fix, not several.
    seen: set[tuple[str, str]] = set()
    unique = [f for f in findings if not (f[:2] in seen or seen.add(f[:2]))]
    return sorted(unique, key=lambda f: f[4])


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2], help="Repository root"
    )
    parser.add_argument(
        "--modules", nargs="*", default=list(DEFAULT_MODULES), help="pypto_core submodules to audit"
    )
    args = parser.parse_args()

    try:
        import_core(args.modules[0])
    except ImportError:
        print(
            "ERROR: `pypto_core` is not importable. This audit reads signatures from the built "
            "extension, so point PYTHONPATH at a build, e.g.\n"
            "  PYTHONPATH=build/python/bindings python tests/lint/audit_stub_container_params.py",
            file=sys.stderr,
        )
        return 2

    total = 0
    for name in args.modules:
        stub = args.root / "python" / "pypto" / "pypto_core" / f"{name}.pyi"
        if not stub.is_file():
            continue
        findings = audit(name, stub)
        total += len(findings)
        for qualname, param, declared, rendered, line in findings:
            short = rendered.replace("collections.abc.", "").replace(f"pypto_core.{name}.", "")
            print(f"{stub.relative_to(args.root)}:{line}: {qualname}({param})")
            print(f"    stub says    : {declared}")
            print(f"    binding takes: {short}")

    if total:
        print(
            f"\n{total} stub parameter(s) are narrower than the binding accepts. Widen each to the "
            f"abstract type the binding reports; leave return types and attributes alone, because "
            f"nanobind really does hand back a concrete list/dict there.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no stub parameter is narrower than its binding ({len(args.modules)} module(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
