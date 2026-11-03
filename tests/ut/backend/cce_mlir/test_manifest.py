# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manifest provenance tests (no cce-mlir dependency)."""

from pypto.backend.cce_mlir.manifest import _queue_task_id, object_digest


def test_queue_task_id_prefers_the_explicit_variable(monkeypatch):
    monkeypatch.setenv("TASKQUEUE_TASK_ID", "task_explicit")
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", "/var/lib/taskqueue/logs/task_from_log.ascend")

    assert _queue_task_id() == "task_explicit"


def test_queue_task_id_is_derived_from_the_device_log_path(monkeypatch):
    monkeypatch.delenv("TASKQUEUE_TASK_ID", raising=False)
    monkeypatch.setenv(
        "ASCEND_PROCESS_LOG_PATH",
        "/var/lib/taskqueue/logs/task_20261103_142518_39102456125.ascend",
    )

    assert _queue_task_id() == "task_20261103_142518_39102456125"


def test_queue_task_id_is_empty_outside_the_queue(monkeypatch):
    monkeypatch.delenv("TASKQUEUE_TASK_ID", raising=False)
    monkeypatch.delenv("ASCEND_PROCESS_LOG_PATH", raising=False)

    assert _queue_task_id() == ""


def test_object_digest_is_content_addressed():
    assert object_digest(b"abc") == object_digest(b"abc")
    assert object_digest(b"abc") != object_digest(b"abd")
