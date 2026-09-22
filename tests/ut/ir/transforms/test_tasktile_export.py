"""Regression tests for the strict PyPTO-to-TaskTile exporter."""

import json
from pathlib import Path

import pypto.language as pl
from pypto import ir


def _program():
    @pl.program
    class TwoTasks:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self, x: pl.Tensor[[1], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[1], pl.FP32]) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                _, first = pl.submit(
                    self.kernel,
                    x,
                    attrs={
                        "tasktile_id": "producer",
                        "tasktile_dependencies": [],
                        "tasktile_duration": 7,
                        "tasktile_engine": "aic",
                        "tasktile_witness": "source:producer",
                        "tasktile_stage_id": "stage0",
                        "tasktile_stage_task": "producer",
                        "tasktile_stage_duration": 3,
                        "tasktile_stage_buffer": "buf0",
                        "tasktile_buffer_id": "buf0",
                        "tasktile_buffer_bytes": 4096,
                        "tasktile_buffer_slots": 2,
                        "tasktile_buffer_memory_space": "ub",
                        "tasktile_comm_id": "comm0",
                        "tasktile_comm_task": "producer",
                        "tasktile_comm_ranks": "0,1",
                        "tasktile_comm_bytes": 1024,
                        "tasktile_comm_synchronization": "fifo",
                    },
                )
                _, second = pl.submit(
                    self.kernel,
                    x,
                    deps=[first],
                    attrs={
                        "tasktile_id": "consumer",
                        # The DSL's attrs round-trip accepts scalar strings;
                        # native pass code may provide a list programmatically.
                        "tasktile_dependencies": "producer",
                        "tasktile_duration": 11,
                        "tasktile_engine": "aiv",
                        "tasktile_witness": "source:consumer",
                    },
                )
            return second

    return TwoTasks


def _fanout_program():
    @pl.program
    class Fanout:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self, x: pl.Tensor[[1], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[1], pl.FP32]) -> pl.Scalar[pl.TASK_ID]:
            with pl.manual_scope():
                _, root = pl.submit(
                    self.kernel,
                    x,
                    attrs={
                        "tasktile_id": "root",
                        "tasktile_dependencies": [],
                        "tasktile_duration": 2,
                        "tasktile_engine": "mte",
                    },
                )
                _, left = pl.submit(
                    self.kernel,
                    x,
                    deps=[root],
                    attrs={
                        "tasktile_id": "left",
                        "tasktile_dependencies": "root",
                        "tasktile_duration": 3,
                        "tasktile_engine": "aic",
                        "tasktile_stage_id": "left-stage",
                        "tasktile_stage_task": "left",
                        "tasktile_stage_duration": 3,
                        "tasktile_stage_buffer": "shared",
                        "tasktile_stage_slot": 0,
                        "tasktile_buffer_id": "shared",
                        "tasktile_buffer_bytes": 2048,
                        "tasktile_buffer_slots": 2,
                    },
                )
                _, right = pl.submit(
                    self.kernel,
                    x,
                    deps=[root],
                    attrs={
                        "tasktile_id": "right",
                        "tasktile_dependencies": "root",
                        "tasktile_duration": 4,
                        "tasktile_engine": "aiv",
                        "tasktile_stage_id": "right-stage",
                        "tasktile_stage_task": "right",
                        "tasktile_stage_duration": 4,
                        "tasktile_stage_buffer": "shared",
                        "tasktile_stage_slot": 1,
                        "tasktile_comm_id": "allgather",
                        "tasktile_comm_task": "right",
                        "tasktile_comm_ranks": "0,1,2,3",
                        "tasktile_comm_bytes": 8192,
                    },
                )
            return right

    return Fanout


def test_export_two_submit_fixture(tmp_path: Path) -> None:
    output = tmp_path / "two-submit.tasktile.json"
    ir.export_tasktile_checkpoint(
        _program(), pass_name="task-dependencies", revision="fixture-rev", output=output
    )
    payload = json.loads(output.read_text())
    tasks = payload["tasktile"]["tasks"]
    assert tasks[0]["id"] == "consumer"
    assert tasks[0]["dependencies"] == ["producer"]
    assert tasks[1]["id"] == "producer"
    assert payload["tasktile"]["stages"] == [
        {
            "buffer": "buf0",
            "bytes": None,
            "duration": 3,
            "id": "stage0",
            "slot": 0,
            "start": None,
            "task": "producer",
            "witness": None,
        }
    ]
    assert payload["tasktile"]["buffers"] == [
        {
            "bytes": 4096,
            "id": "buf0",
            "memory_space": "ub",
            "slots": 2,
            "witness": None,
        }
    ]
    assert all(
        task["reads"] == [] and task["writes"] == [] and task["shape"] == []
        for task in payload["tasktile"]["tasks"]
    )
    assert payload["tasktile"]["communication"] == [
        {
            "bytes": 1024,
            "id": "comm0",
            "ranks": [0, 1],
            "synchronization": "fifo",
            "task": "producer",
            "witness": None,
        }
    ]


def test_default_pipeline_exposes_export_boundary() -> None:
    names = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default).pass_names
    assert names[42] == "AutoDeriveTaskDependencies"
    assert names[50] == "MaterializeRuntimeScopes"
    assert names[52] == "InsertCommFence"
    assert names[53] == "MaterializeValidShapeSymbols"


def test_export_is_repeatable_and_read_only(tmp_path: Path) -> None:
    program = _program()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    ir.export_tasktile_checkpoint(
        program,
        pass_name="post:MaterializeValidShapeSymbols",
        revision="fixture-rev",
        output=first,
    )
    ir.export_tasktile_checkpoint(
        program,
        pass_name="post:MaterializeValidShapeSymbols",
        revision="fixture-rev",
        output=second,
    )
    assert first.read_bytes() == second.read_bytes()


def test_export_fanout_collective_fixture(tmp_path: Path) -> None:
    output = tmp_path / "fanout.tasktile.json"
    ir.export_tasktile_checkpoint(
        _fanout_program(),
        pass_name="post:MaterializeValidShapeSymbols",
        revision="fixture-rev",
        output=output,
    )
    payload = json.loads(output.read_text())
    assert [task["id"] for task in payload["tasktile"]["tasks"]] == [
        "left",
        "right",
        "root",
    ]
    assert [stage["id"] for stage in payload["tasktile"]["stages"]] == [
        "left-stage",
        "right-stage",
    ]
    assert payload["tasktile"]["communication"][0]["ranks"] == [0, 1, 2, 3]
