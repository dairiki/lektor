from __future__ import annotations

import os
import re
import sys
from typing import Any

import pytest

from lektor.builder import ArtifactId
from lektor.buildfailures import BuildFailure
from lektor.buildfailures import FailureController


@pytest.fixture
def output_path(tmp_path):
    return tmp_path.__fspath__()


@pytest.fixture
def failure_controller(pad, output_path):
    return FailureController(pad, output_path)


def test_BuildFailure_from_exc_info():
    def throw_exception():
        x: dict[str, Any] = {}
        try:
            x["somekey"]
        except KeyError:
            raise RuntimeError("test error")  # pylint: disable=raise-missing-from

    artifact_id = ArtifactId("test_artifact")
    failure = None
    try:
        throw_exception()
    except Exception:
        exc_info = sys.exc_info()
        assert exc_info[1] is not None
        failure = BuildFailure.from_exc_info(artifact_id, exc_info)

    assert failure
    assert failure.data["artifact"] == artifact_id
    assert failure.data["exception"] == "RuntimeError: test error"
    traceback = failure.data["traceback"]
    print(traceback)
    patterns = [
        r'x\["somekey"\]',
        r"KeyError: .somekey.",
        r"During handling of the above exception, another exception occurred",
        r"throw_exception\(\)",
        r'raise RuntimeError\("test error"\)',
        r"RuntimeError: test error",
    ]
    for pattern in patterns:
        assert re.search(pattern, traceback)


def test_failure_controller(failure_controller):
    try:
        raise RuntimeError("test exception")
    except Exception:
        failure_controller.store_failure("artifact_id", sys.exc_info())

    failure = failure_controller.lookup_failure("artifact_id")
    assert failure.data["exception"] == "RuntimeError: test exception"

    failure_controller.clear_failure("artifact_id")
    assert failure_controller.lookup_failure("artifact_id") is None


def test_failure_controller_clear_lookup_missing(failure_controller):
    assert failure_controller.lookup_failure("missing_artifact") is None


def test_failure_controller_clear_missing(failure_controller):
    failure_controller.clear_failure("missing_artifact")
    assert failure_controller.lookup_failure("missing_artifact") is None


def test_failure_controller_fs_exceptions(failure_controller):
    # Create a directory in the location that FailureController wants
    # a file stored to trigger unexpected OSErrors.
    filename = failure_controller.get_filename("broken")
    os.makedirs(filename)

    with pytest.raises(OSError):
        failure_controller.lookup_failure("broken")
    with pytest.raises(OSError):
        failure_controller.clear_failure("broken")
    with pytest.raises(OSError):
        try:
            raise RuntimeError("test exception")
        except Exception:
            failure_controller.store_failure("broken", sys.exc_info())
