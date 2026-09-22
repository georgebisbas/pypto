"""Strict, read-only export of PyPTO task metadata to TaskTile JSON.

This module deliberately exports only semantics explicitly attached to the IR.
It never reconstructs dependencies from printed IR or opaque serialization.
Passes that want to participate must attach ``tasktile_*`` attributes at the
selected pass boundary; missing required fields are reported as an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pypto.pypto_core import ir as _ir


class TaskTileExportError(ValueError):
    """Raised when a required TaskTile semantic field is unavailable."""


def _required(attrs: Any, key: str, *, task: str) -> Any:
    value = attrs.get(key) if hasattr(attrs, "get") else None
    if value is None:
        raise TaskTileExportError(
            f"Submit {task!r} is missing required TaskTile attribute {key!r}"
        )
    return value


def _as_int(value: Any, *, key: str, task: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskTileExportError(
            f"Submit {task!r} attribute {key!r} must be an integer"
        )
    return value


def export_tasktile_checkpoint(
    program: _ir.Program,
    *,
    pass_name: str,
    revision: str,
    output: Path,
) -> Path:
    """Write a deterministic TaskTile checkpoint without mutating ``program``.

    Tile, buffer, and communication records are collected only when a
    structured IR node carries the corresponding ``tasktile_*`` attribute.
    The exporter never infers them from operation names or printed IR.
    """

    if not pass_name or not revision:
        raise ValueError("pass_name and revision must be non-empty")
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")

    tasks: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    buffers: list[dict[str, Any]] = []
    communication: list[dict[str, Any]] = []

    class Collector(_ir.IRVisitor):
        def visit_submit(self, op: _ir.Submit) -> None:
            attrs = op.attrs
            task_id = _required(attrs, "tasktile_id", task="<unknown>")
            if not isinstance(task_id, str) or not task_id:
                raise TaskTileExportError("tasktile_id must be a non-empty string")
            deps = _required(attrs, "tasktile_dependencies", task=task_id)
            if isinstance(deps, str):
                deps = [deps]
            if not isinstance(deps, (list, tuple)) or not all(
                isinstance(dep, str) and dep for dep in deps
            ):
                raise TaskTileExportError(
                    f"Submit {task_id!r} tasktile_dependencies must be string IDs"
                )
            duration = _as_int(
                _required(attrs, "tasktile_duration", task=task_id),
                key="tasktile_duration",
                task=task_id,
            )
            engine = attrs.get("tasktile_engine", "aic")
            if not isinstance(engine, str) or not engine:
                raise TaskTileExportError(
                    f"Submit {task_id!r} tasktile_engine must be a string"
                )
            tasks.append(
                {
                    "id": task_id,
                    "duration": duration,
                    "dependencies": list(deps),
                    "engine": engine,
                    "start": attrs.get("tasktile_start"),
                    "witness": attrs.get("tasktile_witness"),
                    "reads": attrs.get("tasktile_reads", []),
                    "writes": attrs.get("tasktile_writes", []),
                    "shape": attrs.get("tasktile_shape", []),
                }
            )
            if "tasktile_stage_id" in attrs:
                stages.append(
                    {
                        "id": attrs["tasktile_stage_id"],
                        "task": attrs["tasktile_stage_task"],
                        "duration": attrs["tasktile_stage_duration"],
                        "buffer": attrs.get("tasktile_stage_buffer"),
                        "slot": attrs.get("tasktile_stage_slot", 0),
                        "start": attrs.get("tasktile_stage_start"),
                        "witness": attrs.get("tasktile_stage_witness"),
                    }
                )
            if "tasktile_buffer_id" in attrs:
                buffers.append(
                    {
                        "id": attrs["tasktile_buffer_id"],
                        "bytes": attrs["tasktile_buffer_bytes"],
                        "slots": attrs.get("tasktile_buffer_slots", 1),
                        "memory_space": attrs.get("tasktile_buffer_memory_space", "ub"),
                        "witness": attrs.get("tasktile_buffer_witness"),
                    }
                )
            if "tasktile_comm_id" in attrs:
                ranks = attrs["tasktile_comm_ranks"]
                if isinstance(ranks, str):
                    ranks = [int(rank) for rank in ranks.split(",") if rank]
                communication.append(
                    {
                        "id": attrs["tasktile_comm_id"],
                        "task": attrs["tasktile_comm_task"],
                        "ranks": ranks,
                        "bytes": attrs["tasktile_comm_bytes"],
                        "synchronization": attrs.get("tasktile_comm_synchronization", "fifo"),
                        "witness": attrs.get("tasktile_comm_witness"),
                    }
                )
            super().visit_submit(op)

        def visit_call(self, op: _ir.Call) -> None:
            attrs = op.attrs
            for key, destination in (
                ("tasktile_stage", stages),
                ("tasktile_buffer", buffers),
                ("tasktile_communication", communication),
            ):
                record = attrs.get(key)
                if record is not None:
                    if not isinstance(record, dict):
                        raise TaskTileExportError(
                            f"Call attribute {key!r} must be a mapping"
                        )
                    destination.append(dict(record))
            if "tasktile_stage_id" in attrs:
                stages.append(
                    {
                        "id": attrs["tasktile_stage_id"],
                        "task": attrs["tasktile_stage_task"],
                        "duration": attrs["tasktile_stage_duration"],
                        "buffer": attrs.get("tasktile_stage_buffer"),
                        "slot": attrs.get("tasktile_stage_slot", 0),
                        "start": attrs.get("tasktile_stage_start"),
                        "witness": attrs.get("tasktile_stage_witness"),
                    }
                )
            if "tasktile_buffer_id" in attrs:
                buffers.append(
                    {
                        "id": attrs["tasktile_buffer_id"],
                        "bytes": attrs["tasktile_buffer_bytes"],
                        "slots": attrs.get("tasktile_buffer_slots", 1),
                        "memory_space": attrs.get("tasktile_buffer_memory_space", "ub"),
                        "witness": attrs.get("tasktile_buffer_witness"),
                    }
                )
            if "tasktile_comm_id" in attrs:
                ranks = attrs["tasktile_comm_ranks"]
                if isinstance(ranks, str):
                    try:
                        ranks = [int(rank) for rank in ranks.split(",") if rank]
                    except ValueError as error:
                        raise TaskTileExportError(
                            "tasktile_comm_ranks string must contain comma-separated integers"
                        ) from error
                communication.append(
                    {
                        "id": attrs["tasktile_comm_id"],
                        "task": attrs["tasktile_comm_task"],
                        "ranks": ranks,
                        "bytes": attrs["tasktile_comm_bytes"],
                        "synchronization": attrs.get("tasktile_comm_synchronization", "fifo"),
                        "witness": attrs.get("tasktile_comm_witness"),
                    }
                )
            super().visit_call(op)

    Collector().visit_program(program)

    payload = {
        "source": {"repository": "pypto", "revision": revision, "pass": pass_name},
        "tasktile": {
            "schema_version": 1,
            "tasks": sorted(tasks, key=lambda item: item["id"]),
            "stages": stages,
            "buffers": buffers,
            "communication": communication,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


__all__ = ["TaskTileExportError", "export_tasktile_checkpoint"]
