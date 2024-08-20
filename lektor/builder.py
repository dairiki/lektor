from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import warnings
import weakref
from collections import deque
from collections import namedtuple
from collections.abc import Sized
from contextlib import AbstractContextManager
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from itertools import chain
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Generic
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import NewType
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import click

from lektor.build_programs import BuildProgram
from lektor.build_programs import builtin_build_programs
from lektor.build_programs import SourceInfo
from lektor.buildfailures import FailureController
from lektor.compat import itertools_batched as batched
from lektor.constants import PRIMARY_ALT
from lektor.context import Context
from lektor.db import Pad
from lektor.environment import Environment
from lektor.environment import TemplateValuesType
from lektor.environment.config import Config
from lektor.reporter import reporter
from lektor.sourceobj import SourceObject
from lektor.sourceobj import VirtualSourceObject
from lektor.sourcesearch import find_files
from lektor.sourcesearch import FindFileResult
from lektor.typing import ExcInfo
from lektor.utils import deprecated
from lektor.utils import DeprecatedWarning
from lektor.utils import process_extra_flags
from lektor.utils import prune_file_and_folder

if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

if sys.version_info >= (3, 10):
    from typing import TypeAlias
    from typing import TypeGuard
else:
    from typing_extensions import TypeAlias
    from typing_extensions import TypeGuard

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath
    from _typeshed import StrPath

# Key use for a source file in the SQL build database
#
# This is the path to a source file relative to the environments root_path,
# with path separators converted to forward slashes.
SourceId = NewType("SourceId", str)

# Key used for an artifact in the SQL build database.
#
# This is the path to the artifact (output file) relative to the build destination path,
# with path separators converted to forward slashes.
ArtifactId = NewType("ArtifactId", str)

# VirtualSource path and alt packed into a single string.
#
# XXX: This is hack to store both the path and the alt in
# the `source` column of the `artifacts` table.
PackedVirtualSourcePath = NewType("PackedVirtualSourcePath", str)


_SourceObj = TypeVar("_SourceObj", bound=SourceObject)

# Maximum number of place-holders allowed in an SQLite statement.
# The default values is 999 for sqlite < 3.32 or 32766 for sqlite >= 3.32
# There seems to be no easy way to determine the actual value at runtime.
# See https://www.sqlite.org/limits.html#max_variable_number
SQLITE_MAX_VARIABLE_NUMBER = 999  # Default SQLITE_MAX_VARIABLE_NUMBER.

_BUILDSTATE_SCHEMA = (resources.files("lektor") / "buildstate_schema.sql").read_text()
if sqlite3.sqlite_version_info < (3, 8, 2):
    # old versions of libsqlite3 do not support WITHOUT ROWID
    _BUILDSTATE_SCHEMA = re.sub(r"(?i)\s+WITHOUT ROWID\b", "", _BUILDSTATE_SCHEMA)


_SqlParameters: TypeAlias = Union[Sequence[Any], Mapping[str, Any]]


_T = TypeVar("_T")


class ThreadLocal(threading.local, Generic[_T]):
    """Helper to store a thread-local Connection.

    This calls the finalizer when the thread exit.
    It works with value types that are not weakrefable (like sqlite3.Connection).
    """

    value: _T
    sentinel: set[None]

    def get(self) -> _T | None:
        if hasattr(self, "value"):
            return self.value
        return None

    def set(self, value: _T, finalizer: Callable[[], None]) -> None:
        self.value = value
        self.sentinel = set()  # simple weakrefable object
        weakref.finalize(self.sentinel, finalizer)


@dataclass(frozen=True)
class SqliteConnectionPool(AbstractContextManager[sqlite3.Connection, None]):
    """Maintain a pool of connections, one per thread, to a sqlite3 database.

    The pool can be used as a contextmanager to execute a complete transaction,
    comitting on success, or rolling-back if the context body raises an exception.
    E.g.

        with pool as conn:
            conn.execute("INSERT INTO ...", data)

    """

    database: StrOrBytesPath
    timeout: float = 10.0

    _con: ThreadLocal[sqlite3.Connection] = field(
        init=False, default_factory=ThreadLocal
    )

    def connect(self) -> sqlite3.Connection:
        """Get cached (thread-local) database connection.

        A new Connection instance will be created if one does not already exist for the
        current thread.

        """
        con = self._con.get()
        if con is None or not self._connection_appears_usable(con):
            con = self._connect()
            self._con.set(con, con.close)
        return con

    def _connect(self) -> sqlite3.Connection:
        """Unconditionally create a new connection."""
        con = sqlite3.connect(self.database, timeout=self.timeout)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    @staticmethod
    def _connection_appears_usable(con: sqlite3.Connection | None) -> bool:
        """Check that a connection appears usuable.

        In particular, if the connection has been closed, it should test as being unusable.
        """
        if not isinstance(con, sqlite3.Connection):
            return False
        try:
            # check that connection is usable (e.g. that it hasn't been closed)
            con.total_changes  # pylint: disable=pointless-statement
        except sqlite3.ProgrammingError:
            return False
        return True

    def __enter__(self) -> sqlite3.Connection:
        connection = self.connect()
        return connection.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        con = self._con.get()
        if con is None:
            raise ValueError("__exit__ called before __enter__")
        con.__exit__(exc_type, exc_value, traceback)

    def execute(self, sql: str, parameters: _SqlParameters = (), /) -> sqlite3.Cursor:
        """Create a new Cursor object and call execute() on it.

        Returns the new Cursor object.
        """
        return self.connect().execute(sql, parameters)

    def executemany(
        self, sql: str, parameters: Iterable[_SqlParameters], /
    ) -> sqlite3.Cursor:
        """Create a new Cursor object and call executemany() on it.

        Returns the new Cursor object.
        """
        return self.connect().executemany(sql, parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        """Create a new Cursor object and call executescript() on it.

        Returns the new Cursor object.
        """
        return self.connect().executescript(sql_script)


def _placeholders(values: Sized) -> str:
    """Return SQL placeholders for values."""
    return ",".join(["?"] * len(values))


class BuildState:
    def __init__(self, builder: Builder, path_cache: PathCache):
        self.builder = builder

        self.updated_artifacts = []
        self.failed_artifacts = []
        self.path_cache = path_cache

    updated_artifacts: list[Artifact]
    failed_artifacts: list[Artifact]

    @property
    def build_db(self) -> SqliteConnectionPool:  # FIXME: delete?
        """The (sqlite) build state database."""
        return self.builder.build_db

    @property
    def pad(self) -> Pad:
        """The pad for this buildstate."""
        return self.builder.pad

    @property
    def env(self) -> Environment:
        """The environment backing this buildstate."""
        return self.builder.env

    @property
    def config(self) -> Config:
        """The config for this buildstate."""
        return self.builder.pad.db.config

    def notify_failure(self, artifact: Artifact, exc_info: ExcInfo) -> None:
        """Notify about a failure.  This marks a failed artifact and stores
        a failure.
        """
        self.failed_artifacts.append(artifact)
        self.builder.failure_controller.store_failure(artifact.artifact_id, exc_info)
        reporter.report_failure(artifact, exc_info)

    def get_file_info(self, filename: StrPath) -> FileInfo:
        if not filename:
            raise ValueError("bad filename: {filename!r}")
        return self.path_cache.get_file_info(filename)

    def to_source_id(self, filename: StrPath) -> SourceId:
        return self.path_cache.to_source_id(filename)

    @deprecated("renamed to to_source_id", version="3.4.0")
    def to_source_filename(self, filename: StrPath) -> SourceId:
        return self.to_source_id(filename)

    def get_virtual_source_info(
        self, virtual_source_path: str, alt: str | None = None
    ) -> VirtualSourceInfo:
        virtual_source = self.pad.get(virtual_source_path, alt=alt)
        if virtual_source is not None:
            if not isinstance(virtual_source, VirtualSourceObject):
                msg = f"not a virtual source path: {virtual_source_path!r}"
                raise ValueError(msg)
            float_mtime = virtual_source.get_mtime(self.path_cache)
            mtime = None if float_mtime is None else int(float_mtime)
            checksum = virtual_source.get_checksum(self.path_cache)
        else:
            mtime = checksum = None
        return VirtualSourceInfo(virtual_source_path, alt, mtime, checksum)

    def get_destination_filename(self, artifact_id: ArtifactId | str) -> str:
        """Returns the destination filename for an artifact name."""
        return os.path.join(
            self.builder.destination_path,
            artifact_id.strip("/").replace("/", os.path.sep),
        )

    def artifact_id_from_destination_filename(self, filename: StrPath) -> ArtifactId:
        """Returns the artifact name for a destination filename."""
        dst = self.builder.destination_path
        filename_ = os.path.join(dst, filename)
        if filename_.startswith(dst):
            filename_ = filename_[len(dst) :].lstrip(os.path.sep)
            if os.path.altsep:
                filename_ = filename_.lstrip(os.path.altsep)
        return ArtifactId(filename_.replace(os.path.sep, "/"))

    @deprecated("renamed to artifact_id_from_destination_filename", version="3.4.0")
    def artifact_name_from_destination_filename(self, filename: StrPath) -> ArtifactId:
        return self.artifact_id_from_destination_filename(filename)

    def new_artifact(
        self,
        artifact_name: str,
        sources: Collection[str] | None = None,
        source_obj: SourceObject | None = None,
        extra: Any | None = None,  # XXX: appears unused?
        config_hash: str | None = None,
    ) -> Artifact:
        """Creates a new artifact and returns it."""
        dst_filename = self.get_destination_filename(artifact_name)
        artifact_id = self.artifact_id_from_destination_filename(dst_filename)
        return Artifact(
            self,
            artifact_id,
            dst_filename,
            sources or (),
            source_obj=source_obj,
            extra=extra,
            config_hash=config_hash,
        )

    @deprecated(version="3.4.0")
    def artifact_exists(self, artifact_id: ArtifactId) -> bool:
        """Given an artifact name this checks if it was already produced."""
        dst_filename = self.get_destination_filename(artifact_id)
        return os.path.exists(dst_filename)

    _DependencyInfo = Union[
        Tuple[SourceId, "FileInfo"],
        Tuple[PackedVirtualSourcePath, "VirtualSourceInfo"],
        Tuple[SourceId, None],
    ]

    @deprecated(version="3.4.0")
    def get_artifact_dependency_infos(
        self, artifact_id: ArtifactId, sources: Iterable[StrPath]
    ) -> list[_DependencyInfo]:
        to_source_id = self.to_source_id
        dependency_infos: list[BuildState._DependencyInfo] = []
        known_source_ids = set()

        for info in self._iter_artifact_source_infos(artifact_id):
            if isinstance(info, VirtualSourceInfo):
                packed_vpath = _pack_virtual_source_path(info.path, info.alt)
                dependency_infos.append((packed_vpath, info))
            else:
                assert isinstance(info, FileInfo)
                source_id = to_source_id(info.filename)
                known_source_ids.add(source_id)
                dependency_infos.append((source_id, info))

        # In any case we also iterate over our direct sources, even if the
        # build state does not know about them yet.  This can be caused by
        # an initial build or a change in original configuration.
        dependency_infos.extend(
            (source_id, None)
            for source in sources
            if (source_id := to_source_id(source)) not in known_source_ids
        )

        return dependency_infos

    def _iter_artifact_source_infos(
        self, artifact_id: ArtifactId
    ) -> Iterator[_ArtifactSourceInfo]:
        """Get saved artifact source information."""

        for path, mtime, size, checksum, is_dir in self.build_db.execute(
            """
            SELECT source, source_mtime, source_size, source_checksum, is_dir
            FROM artifacts
            WHERE artifact = ?
            """,
            [artifact_id],
        ):
            if _is_packed_virtual_source_path(path):
                vpath, alt = _unpack_virtual_source_path(path)
                yield VirtualSourceInfo(vpath, alt, mtime, checksum)
            else:
                yield FileInfo(self.env, path, mtime, size, checksum, bool(is_dir))

    def write_source_info(self, info: SourceInfo) -> None:
        """Writes the source info into the database.  The source info is
        an instance of :class:`lektor.build_programs.SourceInfo`.
        """
        reporter.report_write_source_info(info)
        source_id = self.to_source_id(info.filename)

        with self.build_db as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO source_info (path, alt, lang, type, source, title)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    [info.path, info.alt, lang, info.type, source_id, title]
                    for lang, title in info.title_i18n.items()
                ),
            )

    def prune_source_infos(self) -> None:
        """Remove all source infos of files that no longer exist."""

        root_path = Path(self.env.root_path)
        to_clean = (
            source
            for (source,) in self.build_db.execute(
                "SELECT DISTINCT source FROM source_info"
            )
            if not root_path.joinpath(source).exists()
        )
        with self.build_db as con:
            for batch in batched(to_clean, SQLITE_MAX_VARIABLE_NUMBER):
                con.execute(
                    f"DELETE FROM source_info WHERE SOURCE in ({_placeholders(batch)})",
                    batch,
                )

        for source in to_clean:
            reporter.report_prune_source_info(source)

    def remove_artifact(self, artifact_id: ArtifactId) -> None:
        """Removes an artifact from the build state."""
        with self.build_db as con:
            con.execute("DELETE FROM artifacts WHERE artifact = ?", [artifact_id])

    def _any_sources_are_dirty(self, sources: Collection[str]) -> bool:
        """Given a list of sources this checks if any of them are marked
        as dirty.
        """
        if not sources:
            return False

        cur = self.build_db.execute(
            f"""
            SELECT EXISTS(
                SELECT 1 FROM dirty_sources WHERE source in ({_placeholders(sources)})
            )
            """,
            [self.to_source_id(src) for src in sources],
        )
        return bool(cur.fetchone()[0])

    def _get_artifact_config_hash(self, artifact_id: ArtifactId) -> str | None:
        """Returns the artifact's config hash."""
        cur = self.build_db.execute(
            "SELECT config_hash FROM artifact_config_hashes WHERE artifact = ?",
            [artifact_id],
        )
        if (row := cur.fetchone()) is not None:
            return str(row[0])
        return None

    # FIXME: rename parameter to artifact_id
    def check_artifact_is_current(
        self,
        artifact_id: ArtifactId,
        sources: Collection[str],
        config_hash: str | None,
    ) -> bool:
        # The artifact config changed
        if config_hash != self._get_artifact_config_hash(artifact_id):
            return False

        # If one of our source files is explicitly marked as dirty in the
        # build state, we are not current.
        if self._any_sources_are_dirty(sources):
            return False

        # Read saved dependency info from build db
        source_infos = list(self._iter_artifact_source_infos(artifact_id))

        to_source_id = self.to_source_id
        known_source_ids = {
            to_source_id(info.filename)
            for info in source_infos
            if isinstance(info, FileInfo)
        }

        # If we are missing saved dependency info for any of our direct
        # sources, that means we need to rebuild.
        if any(to_source_id(source) not in known_source_ids for source in sources):
            return False

        # If any of the sources for which we have saved dependency info has
        # changed, a rebuild is needed.
        if any(info.is_changed(self) for info in source_infos):
            return False

        return True

    def iter_existing_artifacts(self) -> Iterator[ArtifactId]:
        """Scan output directory for artifacts.

        Returns an iterable of the artifact_ids for artifacts found.
        """
        is_ignored = self.env.is_ignored_artifact

        def _unignored(filenames: Iterable[str]) -> Iterator[str]:
            return (fn for fn in filenames if not is_ignored(fn))

        dst = self.builder.destination_path
        for dirpath, dirnames, filenames in os.walk(dst):
            dirnames[:] = _unignored(dirnames)
            for filename in _unignored(filenames):
                full_path = os.path.join(dst, dirpath, filename)
                yield self.artifact_id_from_destination_filename(full_path)

    def iter_unreferenced_artifacts(
        self, all: bool | None = None
    ) -> Iterator[ArtifactId]:
        """Finds all unreferenced artifacts in the build folder and yields
        them.
        """
        if all is not None:
            warnings.warn(
                DeprecatedWarning(
                    "all",
                    reason=(
                        "The use of the `all` parameter is deprecated. "
                        "Use the iter_existing_artifacts method to find all existing "
                        "artifacts."
                    ),
                    version="3.4.0",
                ),
                stacklevel=2,
            )

        def _is_unreferenced(artifact_id: ArtifactId) -> bool:
            # Check whether any of the primary sources for the artifact
            # exist and — if the source can be resolved to a record —
            # correspond to non-hidden records.
            cur = self.build_db.execute(
                """
                SELECT DISTINCT source, path, alt
                FROM artifacts LEFT JOIN source_info USING(source)
                WHERE artifact = ?
                    AND is_primary_source""",
                [artifact_id],
            )
            for source, path, alt in cur:
                if self.get_file_info(source).exists:
                    if path is None:
                        return False  # no record to check
                    record = self.pad.get(path, alt)
                    if record is None:
                        # I'm not sure this should happen, but be safe
                        return False
                    if record.is_visible:
                        return False
            # no sources exist, or those that do belong to hidden records
            return True

        existing_artifacts = self.iter_existing_artifacts()
        if all:
            yield from existing_artifacts
        else:
            yield from filter(_is_unreferenced, existing_artifacts)

    @deprecated(version="3.4.0")
    def iter_artifacts(self) -> Iterator[tuple[ArtifactId, FileInfo]]:
        """Iterates over all artifact and their file infos.."""
        for (artifact_id,) in self.build_db.execute(
            "SELECT DISTINCT artifact FROM artifacts ORDER BY artifact"
        ):
            path = self.get_destination_filename(artifact_id)
            info = FileInfo(self.builder.env, path)
            if info.exists:
                yield artifact_id, info

    def vacuum(self) -> None:
        """Vacuums the build db."""
        self.build_db.execute("VACUUM")


def _describe_fs_path_for_checksum(path: Path) -> bytes:
    """Given a file system path this returns a basic description of what
    this is.  This is used for checksum hashing on directories.
    """
    # This is not entirely correct as it does not detect changes for
    # contents from alternatives.  However for the moment it's good
    # enough.
    if path.is_file():
        return b"\x01"
    if path.joinpath("contents.lr").is_file():
        return b"\x02"
    if path.is_dir():
        return b"\x03"
    return b"\x00"


class _ArtifactSourceInfo:
    """Base for classes that contain freshness data about artifact sources.

    Concrete subclasses include FileInfo and VirtualSourceInfo.
    """

    def is_changed(self, build_state: BuildState) -> bool:
        """Determine whether source has changed."""
        raise NotImplementedError()


class FileInfo(_ArtifactSourceInfo):
    """A file info object holds metainformation of a file so that changes
    can be detected easily.
    """

    def __init__(
        self,
        env: Environment,
        path: StrPath,
        mtime: int | None = None,
        size: int | None = None,
        checksum: str | None = None,
        is_dir: bool | None = None,
    ):
        self.env = env
        self.path = Path(path)
        cache = self.__dict__
        if mtime is not None and size is not None and is_dir is not None:
            cache["_stat"] = mtime, size, is_dir
        if checksum is not None:
            cache["checksum"] = checksum

    @property
    def filename(self) -> str:
        return os.fspath(self.path)

    @cached_property
    def _stat(self) -> tuple[int, int, bool]:
        try:
            st = self.path.stat()
            mtime = int(st.st_mtime)
            is_dir = stat.S_ISDIR(st.st_mode)
            if is_dir:
                size = sum(1 for _ in self.path.iterdir())
            else:
                size = st.st_size
            return mtime, size, is_dir
        except OSError:
            return 0, -1, False

    @property
    def mtime(self) -> int:
        """The timestamp of the last modification."""
        return self._stat[0]

    @property
    def size(self) -> int:
        """The size of the file in bytes.  If the file is actually a
        dictionary then the size is actually the number of files in it.
        """
        return self._stat[1]

    @property
    def is_dir(self) -> bool:
        """Is this a directory?"""
        return self._stat[2]

    @property
    def exists(self) -> bool:
        return self.size >= 0

    @cached_property
    def checksum(self) -> str:
        """The checksum of the file or directory."""
        h = hashlib.sha1()
        with suppress(OSError):
            if self.path.is_dir():
                h.update(b"DIR\x00")
                for path in sorted(self.path.iterdir()):
                    if self.env.is_uninteresting_source_name(path.name):
                        continue
                    h.update(path.name.encode("utf-8"))
                    h.update(_describe_fs_path_for_checksum(path))
                    h.update(b"\x00")
            else:
                with self.path.open("rb") as f:
                    while chunk := f.read(16 * 1024):
                        h.update(chunk)

            return h.hexdigest()

        return "0" * 40

    @property
    @deprecated(version="3.4.0")
    def filename_and_checksum(self) -> str:
        """Like 'filename:checksum'."""
        return f"{self.path}:{self.checksum}"

    def unchanged(self, other: FileInfo) -> bool:
        """Given another file info checks if the are similar enough to
        not consider it changed.
        """
        if not isinstance(other, FileInfo):
            # XXX: should return NotImplemented?
            raise TypeError("'other' must be a FileInfo, not %r" % other)

        if self._stat != other._stat:
            return False
        # If mtime and size match, we skip the checksum comparison which
        # might require a file read which we do not want in those cases.
        # (Except if it's a directory, then we won't do that)
        if not self.is_dir:
            return True
        return self.checksum == other.checksum

    def is_changed(self, build_state: BuildState) -> bool:
        other = build_state.get_file_info(self.path)
        return not self.unchanged(other)


def _pack_virtual_source_path(path: str, alt: str | None) -> PackedVirtualSourcePath:
    """Pack VirtualSourceObject's path and alt into a single string.

    The full identity key for a VirtualSourceObject is its ``path`` along with its ``alt``.
    (Two VirtualSourceObjects with differing alts are not the same object.)

    This functions packs the (path, alt) pair into a single string for storage
    in the ``artifacts.path`` of the buildstate database.

    Note that if alternatives are not configured for the current site, there is
    only one alt, so we safely omit the alt from the packed path.
    """
    assert path.startswith("/") and "@" in path
    if alt is not None and alt != PRIMARY_ALT:
        path = f"{alt}@{path}"
    return PackedVirtualSourcePath(path)


def _is_packed_virtual_source_path(path: str) -> TypeGuard[PackedVirtualSourcePath]:
    return "@" in path


def _unpack_virtual_source_path(
    packed: PackedVirtualSourcePath,
) -> tuple[str, str | None]:
    """Unpack VirtualSourceObject's path and alt from packed path.

    This is the inverse of _pack_virtual_source_path.
    """
    alt, sep, path = packed.partition("@")
    if not sep:
        raise ValueError("A packed virtual source path must include at least one '@'")
    if "@" not in path:
        return packed, None
    return path, alt


@dataclass
class VirtualSourceInfo(_ArtifactSourceInfo):
    path: str
    alt: str | None
    mtime: int | None = None
    checksum: str | None = None

    def unchanged(self, other: VirtualSourceInfo) -> bool:
        if not isinstance(other, VirtualSourceInfo):
            raise TypeError("'other' must be a VirtualSourceInfo, not %r" % other)

        if (self.path, self.alt) != (other.path, other.alt):
            raise ValueError(
                "trying to compare mismatched virtual paths: "
                "%r.unchanged(%r)" % (self, other)
            )

        return (self.mtime, self.checksum) == (other.mtime, other.checksum)

    def is_changed(self, build_state: BuildState) -> bool:
        other = build_state.get_virtual_source_info(self.path, self.alt)
        return not self.unchanged(other)


artifacts_row = namedtuple(
    "artifacts_row",
    [
        "artifact",
        "source",
        "source_mtime",
        "source_size",
        "source_checksum",
        "is_dir",
        "is_primary_source",
    ],
)


ArtifactBuildFunc = Callable[["Artifact"], None]

_DBUpdateOp = Callable[[sqlite3.Connection], None]


class Artifact:
    """This class represents a build artifact."""

    def __init__(
        self,
        build_state: BuildState,
        artifact_id: ArtifactId,
        dst_filename: StrPath,
        sources: Collection[str],
        source_obj: SourceObject | None = None,  # XXX: is this ever legitimately None?
        extra: Any | None = None,  # XXX: appears unused?
        config_hash: str | None = None,
    ):
        self.build_state = build_state
        self.artifact_id = artifact_id
        self.dst_filename = dst_filename
        self.sources = sources
        self.in_update_block = False
        self.updated = False
        self.source_obj = source_obj
        self.extra = extra
        self.config_hash = config_hash

        self._new_artifact_file: StrPath | None = None
        self._pending_update_ops: list[_DBUpdateOp] = []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.dst_filename!r}>"

    @property
    @deprecated("renamed to artifact_id", version="3.4.0")
    def artifact_name(self) -> ArtifactId:
        return self.artifact_id

    @property
    def is_current(self) -> bool:
        """Checks if the artifact is current."""
        # If the artifact does not exist, we're not current.
        if not os.path.isfile(self.dst_filename):
            return False

        return self.build_state.check_artifact_is_current(
            self.artifact_id, self.sources, self.config_hash
        )

    @deprecated(version="3.4.0")
    def get_dependency_infos(self) -> list[BuildState._DependencyInfo]:
        return self.build_state.get_artifact_dependency_infos(
            self.artifact_id, self.sources
        )

    def ensure_dir(self) -> None:
        """Creates the directory if it does not exist yet."""
        with suppress(OSError):
            Path(self.dst_filename).parent.mkdir(parents=True, exist_ok=True)

    def open(
        self, mode: str = "rb", encoding: str | None = None, ensure_dir: bool = True
    ) -> IO[Any]:
        """Opens the artifact for reading or writing.  This is transaction
        safe by writing into a temporary file and by moving it over the
        actual source in commit.
        """
        if self._new_artifact_file is not None:
            return open(self._new_artifact_file, mode, encoding=encoding)

        if "r" in mode:
            return open(self.dst_filename, mode, encoding=encoding)

        if ensure_dir:
            self.ensure_dir()
        fd, self._new_artifact_file = tempfile.mkstemp(
            dir=os.path.dirname(self.dst_filename),
            prefix=".__trans",
        )
        return open(fd, mode, encoding=encoding)

    def replace_with_file(
        self, filename: StrPath, ensure_dir: bool = True, copy: bool = False
    ) -> None:
        """This is similar to open but it will move over a given named
        file.  The file will be deleted by a rollback or renamed by a
        commit.
        """
        if copy:
            with self.open("wb") as df:
                with open(filename, "rb") as sf:
                    shutil.copyfileobj(sf, df)
        else:
            if ensure_dir:
                self.ensure_dir()
            self._new_artifact_file = filename

    def render_template_into(
        self,
        template_name: str,
        this: SourceObject,
        *,
        values: TemplateValuesType | None = None,
        alt: str | None = None,
    ) -> None:
        """Renders a template into the artifact."""
        rv = self.build_state.env.render_template(
            template_name, self.build_state.pad, this=this, values=values, alt=alt
        )
        with self.open("wb") as f:
            f.write(rv.encode("utf-8") + b"\n")

    def _memorize_dependencies(
        self,
        dependencies: Collection[StrPath] | None = None,
        virtual_dependencies: Collection[VirtualSourceObject] | None = None,
        for_failure: bool = False,
    ) -> None:
        """This updates the dependencies recorded for the artifact based
        on the direct sources plus the provided dependencies.  This also
        stores the config hash.

        This normally defers the operation until commit but the `for_failure`
        more will immediately commit into a new connection.
        """

        def _iter_artifact_rows() -> Iterator[artifacts_row]:
            to_source_id = self.build_state.to_source_id
            primary_source_ids = set(map(to_source_id, self.sources))
            source_ids = primary_source_ids.union(map(to_source_id, dependencies or ()))
            for source_id in source_ids:
                info = self.build_state.get_file_info(source_id)
                yield artifacts_row(
                    artifact=self.artifact_id,
                    source=source_id,
                    source_mtime=info.mtime,
                    source_size=info.size,
                    source_checksum=info.checksum,
                    is_dir=info.is_dir,
                    is_primary_source=source_id in primary_source_ids,
                )

            for v_source in virtual_dependencies or ():
                checksum = v_source.get_checksum(self.build_state.path_cache)
                mtime = v_source.get_mtime(self.build_state.path_cache)
                yield artifacts_row(
                    artifact=self.artifact_id,
                    source=_pack_virtual_source_path(v_source.path, v_source.alt),
                    source_mtime=mtime,
                    source_size=None,
                    source_checksum=checksum,
                    is_dir=False,
                    is_primary_source=False,
                )

        def operation(con: sqlite3.Connection) -> None:
            rows = list(_iter_artifact_rows())
            reporter.report_dependencies(rows)

            if not for_failure:
                con.execute(
                    "DELETE FROM artifacts WHERE artifact = ?", [self.artifact_id]
                )
            if rows:
                con.executemany(
                    """
                    INSERT OR REPLACE INTO artifacts (
                        artifact, source, source_mtime, source_size,
                        source_checksum, is_dir, is_primary_source)
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            if self.config_hash is None:
                con.execute(
                    "DELETE FROM artifact_config_hashes WHERE artifact = ?",
                    [self.artifact_id],
                )
            else:
                con.execute(
                    """
                    INSERT OR REPLACE INTO artifact_config_hashes
                           (artifact, config_hash) VALUES (?, ?)
                    """,
                    [self.artifact_id, self.config_hash],
                )

        if for_failure:
            with self.build_state.build_db as con:
                operation(con)
        else:
            self._auto_deferred_update_operation(operation)

    def clear_dirty_flag(self) -> None:
        """Clears the dirty flag for all sources."""

        def operation(con: sqlite3.Connection) -> None:
            source_ids = [self.build_state.to_source_id(x) for x in self.sources]
            con.execute(
                f"DELETE FROM dirty_sources WHERE source in ({_placeholders(source_ids)})",
                source_ids,
            )
            reporter.report_dirty_flag(False)

        self._auto_deferred_update_operation(operation)

    def set_dirty_flag(self) -> None:
        """Set dirty flag for all sources.

        This will force the artifact to be rebuilt next time.
        """

        def operation(con: sqlite3.Connection) -> None:
            to_source_id = self.build_state.to_source_id
            if self.sources:
                con.executemany(
                    "INSERT OR REPLACE INTO dirty_sources (source) VALUES (?)",
                    ((to_source_id(source),) for source in self.sources),
                )
                reporter.report_dirty_flag(True)

        self._auto_deferred_update_operation(operation)

    def _auto_deferred_update_operation(self, operation: _DBUpdateOp) -> None:
        """Helper that defers an update operation when inside an update
        block to a later point.  Otherwise it's auto committed.
        """
        if self.in_update_block:
            self._pending_update_ops.append(operation)
        else:
            with self.build_state.build_db as con:
                operation(con)

    @contextmanager
    def update(self) -> Iterator[Context]:
        """Opens the artifact for modifications.  At the start the dirty
        flag is cleared out and if the commit goes through without errors it
        stays cleared.  The setting of the dirty flag has to be done by the
        caller however based on the `exc_info` on the context.
        """
        ctx = self.begin_update()
        try:
            yield ctx
        except BaseException as exc:
            exc_info = sys.exc_info()
            assert exc_info[1] is not None
            self.finish_update(ctx, exc_info)
            if not isinstance(exc, Exception):
                raise
        else:
            self.finish_update(ctx)

    def begin_update(self) -> Context:
        """Begins an update block."""
        if self.in_update_block:
            raise RuntimeError("Artifact is already open for updates.")
        self.updated = False
        ctx = Context(self)
        ctx.push()
        self.in_update_block = True
        self.clear_dirty_flag()
        return ctx

    def _commit(self) -> None:
        with self.build_state.build_db as con:
            for op in self._pending_update_ops:
                op(con)

            if self._new_artifact_file is not None:
                os.replace(self._new_artifact_file, self.dst_filename)
                self._new_artifact_file = None

        self.build_state.updated_artifacts.append(self)
        self.build_state.builder.failure_controller.clear_failure(self.artifact_id)

    def _rollback(self) -> None:
        if self._new_artifact_file is not None:
            try:
                os.remove(self._new_artifact_file)
            except OSError:
                pass
            self._new_artifact_file = None
        self._pending_update_ops = []

    def finish_update(self, ctx: Context, exc_info: ExcInfo | None = None) -> None:
        """Finalizes an update block."""
        if not self.in_update_block:
            raise RuntimeError("Artifact is not open for updates.")
        ctx.pop()
        self.in_update_block = False
        self.updated = True

        # If there was no error, we memoize the dependencies like normal
        # and then commit our transaction.
        if exc_info is None:
            self._memorize_dependencies(
                ctx.referenced_dependencies,
                ctx.referenced_virtual_dependencies,
            )
            self._commit()
            return

        # If an error happened we roll back all changes and record the
        # stacktrace in two locations: we record it on the context so
        # that a called can respond to our failure, and we also persist
        # it so that the dev server can render it out later.
        self._rollback()

        # This is a special form of dependency memorization where we do
        # not prune old dependencies and we just append new ones and we
        # use a new database connection that immediately commits.
        self._memorize_dependencies(
            ctx.referenced_dependencies,
            ctx.referenced_virtual_dependencies,
            for_failure=True,
        )

        ctx.exc_info = exc_info
        self.build_state.notify_failure(self, exc_info)


class PathCache:
    file_info_cache: dict[StrPath, FileInfo]
    source_id_cache: dict[StrPath, SourceId]

    def __init__(self, env: Environment):
        self.file_info_cache = {}
        self.source_id_cache = {}
        self.env = env
        self._root_path = Path(env.root_path).resolve()

    @deprecated("renamed to to_source_id", version="3.4.0")
    def to_source_filename(self, filename: StrPath) -> SourceId:
        return self.to_source_id(filename)

    def to_source_id(self, filename: StrPath) -> SourceId:
        """Given a path somewhere below the environment this will return the
        short source filename that is used internally.  Unlike the given
        path, this identifier is also platform independent.
        """
        key = filename
        source_id = self.source_id_cache.get(key)
        if source_id is None:
            source_id = self._to_source_id(filename)
            self.source_id_cache[key] = source_id
        return source_id

    def _to_source_id(self, filename: StrPath) -> SourceId:
        root_path = self._root_path
        path = root_path.joinpath(filename).resolve()
        try:
            return SourceId(path.relative_to(root_path).as_posix())
        except ValueError as exc:
            message = (
                f"The given value {filename!r} is not below the "
                f"source folder {self.env.root_path!r}"
            )
            raise ValueError(message) from exc

    def get_file_info(self, filename: StrPath) -> FileInfo:
        """Returns the file info for a given file.  This will be cached
        on the generator for the lifetime of it.  This means that further
        accesses to this file info will not cause more IO but it might not
        be safe to use the generator after modifications to the original
        files have been performed on the outside.

        Generally this function can be used to acquire the file info for
        any file on the file system but it should onl be used for source
        files or carefully for other things.

        The filename given can be a source filename.
        """
        path = self._root_path / filename
        file_info = self.file_info_cache.get(path)
        if file_info is None:
            self.file_info_cache[path] = file_info = FileInfo(self.env, path)
        return file_info


class Builder:
    def __init__(
        self,
        pad: Pad,
        destination_path: StrPath,
        buildstate_path: StrPath | None = None,
        extra_flags: dict[str, str] | Iterable[str] | None = None,
    ):
        self.extra_flags = process_extra_flags(extra_flags)
        self.pad = pad
        self.destination_path = os.path.abspath(
            os.path.join(pad.db.env.root_path, destination_path)
        )
        if buildstate_path:
            self.meta_path = buildstate_path
        else:
            self.meta_path = os.path.join(self.destination_path, ".lektor")
        self.failure_controller = FailureController(pad, self.destination_path)

        try:
            os.makedirs(self.meta_path)
            if os.listdir(self.destination_path) != [".lektor"]:
                if not click.confirm(
                    click.style(
                        "The build dir %s hasn't been used before, and other "
                        "files or folders already exist there. If you prune "
                        "(which normally follows the build step), "
                        "they will be deleted. Proceed with building?"
                        % self.destination_path,
                        fg="yellow",
                    )
                ):
                    os.rmdir(self.meta_path)
                    raise click.Abort()
        except OSError:
            pass

        self.build_db = SqliteConnectionPool(self.buildstate_database_filename)
        self.build_db.executescript(_BUILDSTATE_SCHEMA)  # initialize schema

    @property
    def env(self) -> Environment:
        """The environment backing this generator."""
        return self.pad.db.env

    @property
    def buildstate_database_filename(self) -> str:
        """The filename for the build state database."""
        return os.path.join(self.meta_path, "buildstate")

    def touch_site_config(self) -> None:
        """Touches the site config which typically will trigger a rebuild."""
        project_file = self.env.project.project_file
        if project_file:
            with suppress(OSError):
                os.utime(project_file)

    def find_files(
        self,
        query: str,
        alt: str = PRIMARY_ALT,
        lang: str | None = None,
        limit: int = 50,
        types: Collection[str] | None = None,
    ) -> list[FindFileResult]:
        """Returns a list of files that match the query.  This requires that
        the source info is up to date and is primarily used by the admin to
        show files that exist.
        """
        return find_files(self, query, alt, lang, limit, types)

    def new_build_state(self, path_cache: PathCache | None = None) -> BuildState:
        """Creates a new build state."""
        if path_cache is None:
            path_cache = PathCache(self.env)
        return BuildState(self, path_cache)

    def get_build_program(
        self, source: _SourceObj, build_state: BuildState
    ) -> BuildProgram[_SourceObj]:
        """Finds the right build function for the given source file."""
        for cls, builder in chain(
            reversed(self.env.build_programs), reversed(builtin_build_programs)
        ):
            if isinstance(source, cls):
                return cast(BuildProgram[_SourceObj], builder(source, build_state))

        raise RuntimeError("I do not know how to build %r" % source)

    def build_artifact(
        self, artifact: Artifact, build_func: ArtifactBuildFunc
    ) -> Context | None:
        """Various parts of the system once they have an artifact and a
        function to build it, will invoke this function.  This ultimately
        is what builds.

        The return value is the ctx that was used to build this thing
        if it was built, or `None` otherwise.
        """
        is_current = artifact.is_current
        with reporter.build_artifact(artifact, build_func, is_current):
            if not is_current:
                with artifact.update() as ctx:
                    # Upon builing anything we record a dependency to the
                    # project file.  This is not ideal but for the moment
                    # it will ensure that if the file changes we will
                    # rebuild.
                    project_file = self.env.project.project_file
                    if project_file:
                        ctx.record_dependency(os.fspath(project_file))
                    build_func(artifact)
                return ctx
        return None

    @staticmethod
    def update_source_info(
        prog: BuildProgram[SourceObject], build_state: BuildState
    ) -> None:
        """Updates a single source info based on a program.  This is done
        automatically as part of a build.
        """
        info = prog.describe_source_record()
        if info is not None:
            build_state.write_source_info(info)

    def prune(self, all: bool = False) -> None:
        """This cleans up data left in the build folder that does not
        correspond to known artifacts.
        """
        build_state = self.new_build_state()
        if all:
            activity = "clean"
            iter_prunable_artifacts = build_state.iter_existing_artifacts
        else:
            activity = "prune"
            iter_prunable_artifacts = build_state.iter_unreferenced_artifacts

        with reporter.build(activity, self):
            self.env.plugin_controller.emit("before-prune", builder=self, all=all)

            for aft in iter_prunable_artifacts():
                reporter.report_pruned_artifact(aft)
                filename = build_state.get_destination_filename(aft)
                prune_file_and_folder(filename, self.destination_path)
                build_state.remove_artifact(aft)

            build_state.prune_source_infos()
            if all:
                build_state.vacuum()
            self.env.plugin_controller.emit("after-prune", builder=self, all=all)

    def build(
        self, source: SourceObject, path_cache: PathCache | None = None
    ) -> tuple[BuildProgram[SourceObject], BuildState]:
        """Given a source object, builds it."""
        build_state = self.new_build_state(path_cache=path_cache)
        with reporter.process_source(source):
            prog = self.get_build_program(source, build_state)
            self.env.plugin_controller.emit(
                "before-build",
                builder=self,
                build_state=build_state,
                source=source,
                prog=prog,
            )
            prog.build()
            if build_state.updated_artifacts:
                self.update_source_info(prog, build_state)
            self.env.plugin_controller.emit(
                "after-build",
                builder=self,
                build_state=build_state,
                source=source,
                prog=prog,
            )
            return prog, build_state

    def get_initial_build_queue(self) -> deque[SourceObject]:
        """Returns the initial build queue as deque."""
        return deque(self.pad.get_all_roots())

    def extend_build_queue(
        self, queue: deque[SourceObject], prog: BuildProgram[SourceObject]
    ) -> None:
        queue.extend(prog.iter_child_sources())
        for func in self.env.custom_generators:
            queue.extend(func(prog.source) or ())

    def build_all(self) -> int:
        """Builds the entire tree.  Returns the number of failures."""
        failures = 0
        path_cache = PathCache(self.env)
        with reporter.build("build", self):
            self.env.plugin_controller.emit("before-build-all", builder=self)
            to_build = self.get_initial_build_queue()
            while to_build:
                source = to_build.popleft()
                prog, build_state = self.build(source, path_cache=path_cache)
                self.extend_build_queue(to_build, prog)
                failures += len(build_state.failed_artifacts)
            self.env.plugin_controller.emit("after-build-all", builder=self)
            if failures:
                reporter.report_build_all_failure(failures)
        return failures

    def update_all_source_infos(self) -> None:
        """Fast way to update all source infos without having to build
        everything.
        """
        build_state = self.new_build_state()
        with reporter.build("source info update", self):
            to_build = self.get_initial_build_queue()
            while to_build:
                source = to_build.popleft()
                with reporter.process_source(source):
                    prog = self.get_build_program(source, build_state)
                    self.update_source_info(prog, build_state)
                self.extend_build_queue(to_build, prog)
            build_state.prune_source_infos()
