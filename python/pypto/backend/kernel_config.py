# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``kernel_config.py`` generation shared by every kernel compiler route.

The config names the orchestration entry, every kernel source, and optional
per-kernel metadata (signatures, external include dirs, extra link objects).
Keeping it in one module means a new route (for example cce-mlir's linked
device objects) adds a field without forking the generator.
"""

from __future__ import annotations

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes

__all__ = ["format_signature", "generate_kernel_config"]


def format_signature(directions: list[str]) -> str:
    """Render a runtime ArgDirection name list as a ``[_D.IN, _D.OUT, ...]`` body.

    Used for both the per-kernel ``KERNELS`` signature and the ``ORCHESTRATION``
    signature so the ``_D.{name}`` token format lives in one place.
    """
    return ", ".join(f"_D.{d}" for d in directions)


def generate_kernel_config(  # noqa: PLR0913 — arguments are the config dimensions
    orch_func_name: str,
    func_name_to_id: dict[str, int],
    func_name_to_core_type: dict[str, _ir_core.CoreType],
    func_name_to_signature: dict[str, list[str]] | None = None,
    orchestration_signature: list[str] | None = None,
    func_name_to_external_source: dict[str, str] | None = None,
    func_name_to_external_include_dirs: dict[str, tuple[str, ...]] | None = None,
    func_name_to_link_objects: dict[str, list[str]] | None = None,
    enable_sdma: bool = False,
    runtime: _passes.RuntimeKind = _passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    host_tensor_indices: list[int] | None = None,
) -> str:
    """Generate kernel_config.py content.

    ``func_name_to_signature`` maps each kernel name to its runtime
    ``ArgDirection`` names ("IN"/"OUT"/"INOUT") for its tensor args, in
    task-payload (tensors-first) order. Scalars are excluded: the CoreCallable
    signature array is sized to CORE_MAX_TENSOR_ARGS (a per-tensor-arg list), so
    each entry lines up 1:1 with a payload tensor. When present, each KERNELS
    entry gains a ``"signature"`` field of ``ArgDirection`` members so the
    runtime builds a non-empty CoreCallable signature — required for the tensor
    dump to match the task payload tensor_count. Kernels without an entry fall
    back to an empty signature (the pre-existing behavior).

    ``orchestration_signature`` is the orchestration entry's per-tensor
    ``ArgDirection`` names ("IN"/"OUT"/"INOUT"), in ``orch_args`` tensor order
    (scalars excluded). When present, the ``ORCHESTRATION`` dict gains a
    ``"signature"`` field so the runtime builds a non-empty ChipCallable
    signature, indexed by the orch tensor index in
    ``bind_callable_to_runtime_impl``. This lets read-only IN tensors skip the
    wasteful D2H copy-back and pure-OUT tensors take the on-device memset fast
    path. Without it the signature is empty and every tensor is conservatively
    copied back (the pre-existing behavior).

    ``func_name_to_external_include_dirs`` maps external kernel names to their
    ordered CCEC include search paths. Non-external kernels ignore this map.

    ``func_name_to_link_objects`` maps a kernel to artifact-relative object
    files linked after its generated source under ``kernel_entry``. The cce-mlir
    route uses it for the compiled device object.

    ``enable_sdma`` records that at least one emitted kernel consumes the
    runtime-owned SDMA workspace.

    ``runtime`` selects the Simpler runtime ABI. Its wire name is written to
    ``RUNTIME_CONFIG["runtime"]`` — the value ``CompiledProgram.runtime_name``
    reports and the one the ``ChipWorker`` reuse lookup matches against, so a
    program compiled for one runtime never binds to a worker running another.

    Returns:
        The ``kernel_config.py`` source.
    """
    runtime_name = _passes.runtime_kind_to_name(runtime)
    func_name_to_signature = func_name_to_signature or {}
    func_name_to_external_source = func_name_to_external_source or {}
    func_name_to_external_include_dirs = func_name_to_external_include_dirs or {}
    orchestration_signature = orchestration_signature or []
    has_signatures = any(func_name_to_signature.values()) or bool(orchestration_signature)

    runtime_lines = [
        "RUNTIME_CONFIG = {",
        f'\t"runtime": "{runtime_name}",',
        '\t"aicpu_thread_num": 0,',
    ]
    if host_tensor_indices:
        runtime_lines.append(f'\t"host_tensor_indices": {host_tensor_indices},')
    if enable_sdma:
        runtime_lines.append('\t"enable_sdma": True,')
    runtime_lines.append("}\n")

    header = [
        "# Kernel and Orchestration Configuration\n",
        "from pathlib import Path\n",
    ]
    # ArgDirection is only imported when at least one kernel has a signature,
    # so configs for kernels without directions stay import-free.
    if has_signatures:
        header.append("from simpler.task_interface import ArgDirection as _D\n")
    header.append("_ROOT_DIR = Path(__file__).parent\n")

    lines = [
        *header,
        f"# Runtime configuration for {runtime_name}.",
        "# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).",
        *runtime_lines,
        "ORCHESTRATION = {",
        f'\t"source": str(_ROOT_DIR / "orchestration" / "{orch_func_name}.cpp"),',
        '\t"function_name": "aicpu_orchestration_entry",',
    ]
    if orchestration_signature:
        lines.append(f'\t"signature": [{format_signature(orchestration_signature)}],')
    lines += [
        "}\n",
        "KERNELS = [",
    ]

    for name, func_id in sorted(func_name_to_id.items(), key=lambda x: x[1]):
        core_type = func_name_to_core_type[name]
        ct_str = "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"
        # External kernels are referenced in place at their original path so the
        # entry .cpp keeps its sibling files (relative #include "../..." resolve);
        # DSL kernels are generated at kernels/<ct>/<name>.cpp under the artifact.
        ext_source = func_name_to_external_source.get(name)
        if ext_source is not None:
            source_expr = repr(ext_source)
        else:
            source_expr = f'str(_ROOT_DIR / "kernels" / "{ct_str}" / "{name}.cpp")'
        entry = (
            f'\t{{"func_id": {func_id}, "name": "{name}", "source": {source_expr}, "core_type": "{ct_str}"'
        )
        if ext_source is not None:
            entry += ', "external": True'
            include_dirs = func_name_to_external_include_dirs.get(name)
            if include_dirs:
                entry += f', "extra_include_dirs": {list(include_dirs)!r}'
        signature = func_name_to_signature.get(name)
        if signature:
            entry += f', "signature": [{format_signature(signature)}]'
        link_objects = (func_name_to_link_objects or {}).get(name)
        if link_objects:
            rendered = ", ".join(f'str(_ROOT_DIR / "{path}")' for path in link_objects)
            entry += f', "link_objects": [{rendered}]'
        entry += "},"
        lines.append(entry)

    lines.append("]")
    return "\n".join(lines) + "\n"
