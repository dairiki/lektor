from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import sys
from pathlib import Path
from traceback import TracebackException
from typing import TYPE_CHECKING

from marshmallow_dataclass import class_schema

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from _typeshed import StrPath
    from lektor.builder import ArtifactId
    from lektor.db import Pad
    from lektor.typing import ExcInfo


@dataclasses.dataclass
class BuildFailure:
    artifact: str  # XXX: should be ArtifactId, but circdep causes issues
    exception: str  # formatted exception name
    traceback: str  # formatted traceback

    @classmethod
    def from_exc_info(
        cls: type[Self], artifact_id: ArtifactId, exc_info: ExcInfo
    ) -> Self:
        te = TracebackException(*exc_info)
        # NB: we have dropped werkzeug's support for Paste's __traceback_hide__
        # frame local.
        return cls(
            artifact_id,
            exception="".join(te.format_exception_only()).strip(),
            traceback="".join(te.format()).strip(),
        )

    def to_json(self) -> dict[str, str]:
        return dataclasses.asdict(self)

    @property
    def data(self) -> dict[str, str]:
        return self.to_json()


BuildFailureSchema = class_schema(BuildFailure)


class FailureController:
    def __init__(self, pad: Pad, destination_path: StrPath):
        destination_path = Path(pad.db.env.root_path, destination_path).resolve()
        self.pad = pad
        self.path = destination_path / ".lektor/failures"

    def get_path(self, artifact_id: ArtifactId) -> Path:
        namehash = hashlib.md5(artifact_id.encode("utf-8")).hexdigest()
        return self.path / f"{namehash}.json"

    def get_filename(self, artifact_id: ArtifactId) -> str:
        # b/c
        return os.fspath(self.get_path(artifact_id))

    def lookup_failure(self, artifact_id: ArtifactId) -> BuildFailure | None:
        """Looks up a failure for the given artifact name."""
        fn = self.get_filename(artifact_id)
        try:
            with open(fn, encoding="utf-8") as f:
                schema = BuildFailureSchema()
                return schema.load(json.load(f))  # type: ignore[no-any-return]
        except OSError as ex:
            if ex.errno == errno.ENOENT:
                return None
            raise

    def clear_failure(self, artifact_id: ArtifactId) -> None:
        """Clears a stored failure."""
        try:
            os.unlink(self.get_filename(artifact_id))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def store_failure(self, artifact_id: ArtifactId, exc_info: ExcInfo) -> None:
        """Stores a failure from an exception info tuple."""
        fn = self.get_filename(artifact_id)
        failure = BuildFailure.from_exc_info(artifact_id, exc_info)
        try:
            os.makedirs(os.path.dirname(fn))
        except OSError:
            pass
        with open(fn, mode="w", encoding="utf-8") as fp:
            print(json.dumps(failure.to_json()), file=fp)
