from __future__ import annotations

import base64
import builtins
import hashlib
import io
import mimetypes
import os
from contextlib import suppress
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import IO
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING

from lektor.utils import deprecated

if TYPE_CHECKING:
    from _typeshed import StrPath


class FileContents:
    @deprecated(name="FileContents", version="3.4.0")
    def __init__(self, filename: StrPath):
        self.filename = filename
        self._path = Path(filename)
        self._md5 = None
        self._sha1 = None
        self._integrity = None
        self._mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    @dataclass
    class _Hashes:
        md5: str
        sha1: str
        integrity: str

    @property
    def sha1(self) -> str:
        return self._hashes.sha1

    @property
    def md5(self) -> str:
        return self._hashes.md5

    @property
    def integrity(self) -> str:
        return self._hashes.integrity

    @property
    def mimetype(self) -> str:
        return self._mimetype

    @property
    def bytes(self) -> int:
        with suppress(OSError):
            return os.stat(self.filename).st_size
        return 0

    def as_data_url(self, mediatype: str | None = None) -> str:
        if mediatype is None:
            mediatype = self.mimetype
        return f"data:{mediatype};base64,{self.as_base64()}"

    def as_text(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def as_bytes(self) -> builtins.bytes:
        return self._path.read_bytes()

    def as_base64(self) -> str:
        return base64.b64encode(self.as_bytes()).decode("ascii")

    @overload
    def open(
        self, mode: Literal["r"] = "r", encoding: str | None = None
    ) -> io.TextIOWrapper:
        ...

    @overload
    def open(self, mode: Literal["rb"], encoding: None = None) -> io.BufferedReader:
        ...

    def open(
        self, mode: Literal["r", "rb"] = "r", encoding: str | None = None
    ) -> IO[Any]:
        if mode not in {"rb", "r"}:
            raise TypeError("Can only open files for reading")
        return open(self.filename, mode, encoding=encoding)

    @cached_property
    def _hashes(self) -> _Hashes:
        with self.open("rb") as f:
            md5 = hashlib.md5()
            sha1 = hashlib.sha1()
            sha384 = hashlib.sha384()
            while chunk := f.read(16384):
                md5.update(chunk)
                sha1.update(chunk)
                sha384.update(chunk)

            integrity = "sha384-" + base64.b64encode(sha384.digest()).decode("ascii")

            return FileContents._Hashes(
                md5=md5.hexdigest(),
                sha1=sha1.hexdigest(),
                integrity=integrity,
            )

    def __repr__(self) -> str:
        return f"<FileContents {self.filename!r} md5={self.md5!r}>"
