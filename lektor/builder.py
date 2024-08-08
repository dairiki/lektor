from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from collections import deque
from collections import namedtuple
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import NewType
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import click

from lektor.build_programs import BuildProgram
from lektor.build_programs import builtin_build_programs
from lektor.build_programs import SourceInfo
from lektor.buildfailures import FailureController
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
from lektor.utils import deprecated
from lektor.typing import ExcInfo
from lektor.utils import process_extra_flags
from lektor.utils import prune_file_and_folder

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

if TYPE_CHECKING:
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


def create_tables(con: sqlite3.Connection) -> None:
    can_disable_rowid = (3, 8, 2) <= sqlite3.sqlite_version_info
    if can_disable_rowid:
        without_rowid = "without rowid"
    else:
        without_rowid = ""

    try:
        con.execute(
            f"""
            create table if not exists artifacts (
                artifact text,
                source text,
                source_mtime integer,
                source_size integer,
                source_checksum text,
                is_dir integer,
                is_primary_source integer,
                primary key (artifact, source)
            ) {without_rowid};
        """
        )
        con.execute(
            """
            create index if not exists artifacts_source on artifacts (
                source
            );
        """
        )
        con.execute(
            f"""
            create table if not exists artifact_config_hashes (
                artifact text,
                config_hash text,
                primary key (artifact)
            ) {without_rowid};
        """
        )
        con.execute(
            f"""
            create table if not exists dirty_sources (
                source text,
                primary key (source)
            ) {without_rowid};
        """
        )
        con.execute(
            f"""
            create table if not exists source_info (
                path text,
                alt text,
                lang text,
                type text,
                source text,
                title text,
                primary key (path, alt, lang)
            ) {without_rowid};
        """
        )
    finally:
        con.close()


class BuildState:
    def __init__(self, builder: Builder, path_cache: PathCache):
        self.builder = builder

        self.updated_artifacts = []
        self.failed_artifacts = []
        self.path_cache = path_cache

    updated_artifacts: list[Artifact]
    failed_artifacts: list[Artifact]

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

    def connect_to_database(self) -> sqlite3.Connection:
        """Returns a database connection for the build state db."""
        return self.builder.connect_to_database()

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

    def artifact_exists(self, artifact_id: ArtifactId) -> bool:
        """Given an artifact name this checks if it was already produced."""
        dst_filename = self.get_destination_filename(artifact_id)
        return os.path.exists(dst_filename)

    _DependencyInfo = Union[
        Tuple[SourceId, "FileInfo"],
        Tuple[PackedVirtualSourcePath, "VirtualSourceInfo"],
        Tuple["StrPath", None],
    ]

    # FIXME: rename parameter to artifact_id
    def get_artifact_dependency_infos(
        self, artifact_id: ArtifactId, sources: Iterable[StrPath]
    ) -> list[_DependencyInfo]:
        con = self.connect_to_database()
        try:
            cur = con.cursor()
            rv = list(self._iter_artifact_dependency_infos(cur, artifact_id, sources))
        finally:
            con.close()
        return rv

    def _iter_artifact_dependency_infos(
        self,
        cur: sqlite3.Cursor,
        artifact_id: ArtifactId,
        sources: Iterable[StrPath],
    ) -> Iterator[_DependencyInfo]:
        """This iterates over all dependencies as file info objects."""
        cur.execute(
            """
            select source, source_mtime, source_size,
                   source_checksum, is_dir
            from artifacts
            where artifact = ?
        """,
            [artifact_id],
        )
        rv = cur.fetchall()

        found = set()
        for path, mtime, size, checksum, is_dir in rv:
            if _is_packed_virtual_source_path(path):
                vpath, alt = _unpack_virtual_source_path(path)
                yield path, VirtualSourceInfo(vpath, alt, mtime, checksum)
            else:
                file_info = FileInfo(
                    self.env, path, mtime, size, checksum, bool(is_dir)
                )
                source_id = self.to_source_id(file_info.filename)
                found.add(source_id)
                yield source_id, file_info

        # In any case we also iterate over our direct sources, even if the
        # build state does not know about them yet.  This can be caused by
        # an initial build or a change in original configuration.
        for source in sources:
            source_id = self.to_source_id(source)
            if source_id not in found:
                yield source, None

    def write_source_info(self, info: SourceInfo) -> None:
        """Writes the source info into the database.  The source info is
        an instance of :class:`lektor.build_programs.SourceInfo`.
        """
        reporter.report_write_source_info(info)
        source_id = self.to_source_id(info.filename)
        con = self.connect_to_database()
        try:
            cur = con.cursor()
            for lang, title in info.title_i18n.items():
                cur.execute(
                    """
                    insert or replace into source_info
                        (path, alt, lang, type, source, title)
                        values (?, ?, ?, ?, ?, ?)
                """,
                    [info.path, info.alt, lang, info.type, source_id, title],
                )
            con.commit()
        finally:
            con.close()

    def prune_source_infos(self) -> None:
        """Remove all source infos of files that no longer exist."""
        MAX_VARS = 999  # Default SQLITE_MAX_VARIABLE_NUMBER.
        con = self.connect_to_database()
        to_clean = []
        try:
            cur = con.cursor()
            cur.execute(
                """
                select distinct source from source_info
            """
            )
            for (source,) in cur.fetchall():
                fs_path = os.path.join(self.env.root_path, source)
                if not os.path.exists(fs_path):
                    to_clean.append(source)

            if to_clean:
                for i in range(0, len(to_clean), MAX_VARS):
                    chunk = to_clean[i : i + MAX_VARS]
                    cur.execute(
                        """
                        delete from source_info
                         where source in (%s)
                    """
                        % ", ".join(["?"] * len(chunk)),
                        chunk,
                    )

                con.commit()
        finally:
            con.close()

        for source in to_clean:
            reporter.report_prune_source_info(source)

    def remove_artifact(self, artifact_id: ArtifactId) -> None:
        """Removes an artifact from the build state."""
        con = self.connect_to_database()
        try:
            cur = con.cursor()
            cur.execute(
                """
                delete from artifacts where artifact = ?
            """,
                [artifact_id],
            )
            con.commit()
        finally:
            con.close()

    def _any_sources_are_dirty(
        self, cur: sqlite3.Cursor, sources: Collection[str]
    ) -> bool:
        """Given a list of sources this checks if any of them are marked
        as dirty.
        """
        source_ids = [self.to_source_id(x) for x in sources]
        if not source_ids:
            return False

        cur.execute(
            """
            select source from dirty_sources where source in (%s) limit 1
        """
            % ", ".join(["?"] * len(source_ids)),
            source_ids,
        )
        return cur.fetchone() is not None

    @staticmethod
    def _get_artifact_config_hash(
        cur: sqlite3.Cursor, artifact_id: ArtifactId
    ) -> str | None:
        """Returns the artifact's config hash."""
        cur.execute(
            """
            select config_hash from artifact_config_hashes
             where artifact = ?
        """,
            [artifact_id],
        )
        rv = cur.fetchone()
        return rv[0] if rv else None

    # FIXME: rename parameter to artifact_id
    def check_artifact_is_current(
        self,
        artifact_id: ArtifactId,
        sources: Collection[str],
        config_hash: str | None,
    ) -> bool:
        con = self.connect_to_database()
        cur = con.cursor()
        try:
            # The artifact config changed
            if config_hash != self._get_artifact_config_hash(cur, artifact_id):
                return False

            # If one of our source files is explicitly marked as dirty in the
            # build state, we are not current.
            if self._any_sources_are_dirty(cur, sources):
                return False

            # If we do have an already existing artifact, we need to check if
            # any of the source files we depend on changed.
            for _, info in self._iter_artifact_dependency_infos(
                cur, artifact_id, sources
            ):
                # if we get a missing source info it means that we never
                # saw this before.  This means we need to build it.
                if info is None:
                    return False

                if info.is_changed(self):
                    return False

            return True
        finally:
            con.close()

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

    def iter_unreferenced_artifacts(self, all: bool = False) -> Iterator[ArtifactId]:
        """Finds all unreferenced artifacts in the build folder and yields
        them.
        """
        if all:
            yield from self.iter_existing_artifacts()

        con = self.connect_to_database()
        cur = con.cursor()

        def _is_unreferenced(artifact_id: ArtifactId) -> bool:
            # Check whether any of the primary sources for the artifact
            # exist and — if the source can be resolved to a record —
            # correspond to non-hidden records.
            cur.execute(
                """
                SELECT DISTINCT source, path, alt
                FROM artifacts LEFT JOIN source_info USING(source)
                WHERE artifact = ?
                    AND is_primary_source""",
                [artifact_id],
            )
            for source, path, alt in cur.fetchall():
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

        try:
            yield from filter(_is_unreferenced, self.iter_existing_artifacts())
        finally:
            con.close()

    def iter_artifacts(self) -> Iterator[tuple[ArtifactId, FileInfo]]:
        """Iterates over all artifact and their file infos.."""
        con = self.connect_to_database()
        try:
            cur = con.cursor()
            cur.execute(
                """
                select distinct artifact from artifacts order by artifact
            """
            )
            rows = cur.fetchall()
            con.close()
            for (artifact_id,) in rows:
                path = self.get_destination_filename(artifact_id)
                info = FileInfo(self.builder.env, path)
                if info.exists:
                    yield artifact_id, info
        finally:
            con.close()

    def vacuum(self) -> None:
        """Vacuums the build db."""
        con = self.connect_to_database()
        try:
            con.execute("vacuum")
        finally:
            con.close()


def _describe_fs_path_for_checksum(path: StrPath) -> bytes:
    """Given a file system path this returns a basic description of what
    this is.  This is used for checksum hashing on directories.
    """
    # This is not entirely correct as it does not detect changes for
    # contents from alternatives.  However for the moment it's good
    # enough.
    if os.path.isfile(path):
        return b"\x01"
    if os.path.isfile(os.path.join(path, "contents.lr")):
        return b"\x02"
    if os.path.isdir(path):
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
        filename: StrPath,
        mtime: int | None = None,
        size: int | None = None,
        checksum: str | None = None,
        is_dir: bool | None = None,
    ):
        self.env = env
        self.filename = filename
        if mtime is not None and size is not None and is_dir is not None:
            self._stat = (mtime, size, is_dir)
        else:
            self._stat = None
        self._checksum = checksum

    _stat: tuple[int, int, bool] | None

    def _get_stat(self) -> tuple[int, int, bool]:
        rv = self._stat
        if rv is not None:
            return rv

        try:
            st = os.stat(self.filename)
            mtime = int(st.st_mtime)
            if stat.S_ISDIR(st.st_mode):
                size = len(os.listdir(self.filename))
                is_dir = True
            else:
                size = int(st.st_size)
                is_dir = False
            rv = mtime, size, is_dir
        except OSError:
            rv = 0, -1, False
        self._stat = rv
        return rv

    @property
    def mtime(self) -> int:
        """The timestamp of the last modification."""
        return self._get_stat()[0]

    @property
    def size(self) -> int:
        """The size of the file in bytes.  If the file is actually a
        dictionary then the size is actually the number of files in it.
        """
        return self._get_stat()[1]

    @property
    def is_dir(self) -> bool:
        """Is this a directory?"""
        return self._get_stat()[2]

    @property
    def exists(self) -> bool:
        return self.size >= 0

    @property
    def checksum(self) -> str:
        """The checksum of the file or directory."""
        rv = self._checksum
        if rv is not None:
            return rv

        try:
            h = hashlib.sha1()
            if os.path.isdir(self.filename):
                h.update(b"DIR\x00")
                for filename in sorted(os.listdir(self.filename)):
                    if self.env.is_uninteresting_source_name(filename):
                        continue
                    h.update(filename.encode("utf-8"))
                    h.update(
                        _describe_fs_path_for_checksum(
                            os.path.join(self.filename, filename)
                        )
                    )
                    h.update(b"\x00")
            else:
                with open(self.filename, "rb") as f:
                    while 1:
                        chunk = f.read(16 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
            checksum = h.hexdigest()
        except OSError:
            checksum = "0" * 40
        self._checksum = checksum
        return checksum

    @property
    def filename_and_checksum(self) -> str:
        """Like 'filename:checksum'."""
        return f"{self.filename}:{self.checksum}"

    def unchanged(self, other: FileInfo) -> bool:
        """Given another file info checks if the are similar enough to
        not consider it changed.
        """
        if not isinstance(other, FileInfo):
            # XXX: should return NotImplemented?
            raise TypeError("'other' must be a FileInfo, not %r" % other)

        if self.mtime != other.mtime or self.size != other.size:
            return False
        # If mtime and size match, we skip the checksum comparison which
        # might require a file read which we do not want in those cases.
        # (Except if it's a directory, then we won't do that)
        if not self.is_dir:
            return True
        return self.checksum == other.checksum

    def is_changed(self, build_state: BuildState) -> bool:
        other = build_state.get_file_info(self.filename)
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

        def operation(con: sqlite3.Connection) -> None:
            primary_source_ids = {
                self.build_state.to_source_id(x) for x in self.sources
            }

            seen: set[SourceId] = set()
            rows = []
            for source in chain(self.sources, dependencies or ()):
                source_id = self.build_state.to_source_id(source)
                if source_id in seen:
                    continue
                info = self.build_state.get_file_info(source_id)
                rows.append(
                    artifacts_row(
                        artifact=self.artifact_id,
                        source=source_id,
                        source_mtime=info.mtime,
                        source_size=info.size,
                        source_checksum=info.checksum,
                        is_dir=info.is_dir,
                        is_primary_source=source_id in primary_source_ids,
                    )
                )

                seen.add(source_id)

            for v_source in virtual_dependencies or ():
                checksum = v_source.get_checksum(self.build_state.path_cache)
                mtime = v_source.get_mtime(self.build_state.path_cache)
                rows.append(
                    artifacts_row(
                        artifact=self.artifact_id,
                        source=_pack_virtual_source_path(v_source.path, v_source.alt),
                        source_mtime=mtime,
                        source_size=None,
                        source_checksum=checksum,
                        is_dir=False,
                        is_primary_source=False,
                    )
                )

            reporter.report_dependencies(rows)

            cur = con.cursor()
            if not for_failure:
                cur.execute(
                    "delete from artifacts where artifact = ?", [self.artifact_id]
                )
            if rows:
                cur.executemany(
                    """
                    insert or replace into artifacts (
                        artifact, source, source_mtime, source_size,
                        source_checksum, is_dir, is_primary_source)
                    values (?, ?, ?, ?, ?, ?, ?)
                """,
                    rows,
                )

            if self.config_hash is None:
                cur.execute(
                    """
                    delete from artifact_config_hashes
                     where artifact = ?
                """,
                    [self.artifact_id],
                )
            else:
                cur.execute(
                    """
                    insert or replace into artifact_config_hashes
                           (artifact, config_hash) values (?, ?)
                """,
                    [self.artifact_id, self.config_hash],
                )

            cur.close()

        if for_failure:
            con = self.build_state.connect_to_database()
            try:
                operation(con)
            except:  # noqa
                con.rollback()
                con.close()
                raise
            con.commit()
            con.close()
        else:
            self._auto_deferred_update_operation(operation)

    def clear_dirty_flag(self) -> None:
        """Clears the dirty flag for all sources."""

        def operation(con: sqlite3.Connection) -> None:
            source_ids = [self.build_state.to_source_id(x) for x in self.sources]
            cur = con.cursor()
            cur.execute(
                """
                delete from dirty_sources where source in (%s)
            """
                % ", ".join(["?"] * len(source_ids)),
                list(source_ids),
            )
            cur.close()
            reporter.report_dirty_flag(False)

        self._auto_deferred_update_operation(operation)

    def set_dirty_flag(self) -> None:
        """Given a list of artifacts this will mark all of their sources
        as dirty so that they will be rebuilt next time.
        """

        def operation(con: sqlite3.Connection) -> None:
            source_ids = set()
            for source in self.sources:
                source_ids.add(self.build_state.to_source_id(source))

            if not source_ids:
                return

            cur = con.cursor()
            cur.executemany(
                """
                insert or replace into dirty_sources (source) values (?)
            """,
                [(x,) for x in source_ids],
            )
            cur.close()

            reporter.report_dirty_flag(True)

        self._auto_deferred_update_operation(operation)

    def _auto_deferred_update_operation(self, f: _DBUpdateOp) -> None:
        """Helper that defers an update operation when inside an update
        block to a later point.  Otherwise it's auto committed.
        """
        if self.in_update_block:
            self._pending_update_ops.append(f)
            return
        con = self.build_state.connect_to_database()
        try:
            f(con)
        except:  # noqa
            con.rollback()
            raise
        con.commit()

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
        con = None
        try:
            for op in self._pending_update_ops:
                if con is None:
                    con = self.build_state.connect_to_database()
                op(con)

            if self._new_artifact_file is not None:
                os.replace(self._new_artifact_file, self.dst_filename)
                self._new_artifact_file = None

            if con is not None:
                con.commit()
                con.close()
                con = None

            self.build_state.updated_artifacts.append(self)
            self.build_state.builder.failure_controller.clear_failure(self.artifact_id)
        finally:
            if con is not None:
                con.rollback()
                con.close()

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

    def to_source_id(self, filename: StrPath) -> SourceId:
        """Given a path somewhere below the environment this will return the
        short source filename that is used internally.  Unlike the given
        path, this identifier is also platform independent.
        """
        key = filename
        rv = self.source_id_cache.get(key)
        if rv is not None:
            return rv
        folder = os.path.abspath(self.env.root_path)
        filename = os.path.normpath(os.path.join(folder, filename))
        if filename.startswith(folder):
            filename = filename[len(folder) :].lstrip(os.path.sep)
            if os.path.altsep:
                filename = filename.lstrip(os.path.altsep)
        else:
            raise ValueError(
                "The given value (%r) is not below the "
                "source folder (%r)" % (filename, self.env.root_path)
            )
        source_id = SourceId(filename.replace(os.path.sep, "/"))
        self.source_id_cache[key] = source_id
        return source_id

    @deprecated("renamed to to_source_id", version="3.4.0")
    def to_source_filename(self, filename: StrPath) -> SourceId:
        return self.to_source_id(filename)

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
        fn = os.path.join(self.env.root_path, filename)
        rv = self.file_info_cache.get(fn)
        if rv is None:
            self.file_info_cache[fn] = rv = FileInfo(self.env, fn)
        return rv


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

        con = self.connect_to_database()
        try:
            create_tables(con)
        finally:
            con.close()

    @property
    def env(self) -> Environment:
        """The environment backing this generator."""
        return self.pad.db.env

    @property
    def buildstate_database_filename(self) -> str:
        """The filename for the build state database."""
        return os.path.join(self.meta_path, "buildstate")

    def connect_to_database(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.buildstate_database_filename, timeout=10)
        cur = con.cursor()
        cur.execute("pragma journal_mode=WAL")
        cur.execute("pragma synchronous=NORMAL")
        con.commit()
        cur.close()
        return con

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
    ) -> list[dict[str, str | list[dict[str, str]]]]:
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
        path_cache = PathCache(self.env)
        build_state = self.new_build_state(path_cache=path_cache)
        with reporter.build(all and "clean" or "prune", self):
            self.env.plugin_controller.emit("before-prune", builder=self, all=all)

            for aft in build_state.iter_unreferenced_artifacts(all=all):
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
        # We keep a dummy connection here that does not do anything which
        # helps us with the WAL handling.  See #144
        con = self.connect_to_database()
        try:
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
        finally:
            con.close()

    def update_all_source_infos(self) -> None:
        """Fast way to update all source infos without having to build
        everything.
        """
        build_state = self.new_build_state()
        # We keep a dummy connection here that does not do anything which
        # helps us with the WAL handling.  See #144
        con = self.connect_to_database()
        try:
            with reporter.build("source info update", self):
                to_build = self.get_initial_build_queue()
                while to_build:
                    source = to_build.popleft()
                    with reporter.process_source(source):
                        prog = self.get_build_program(source, build_state)
                        self.update_source_info(prog, build_state)
                    self.extend_build_queue(to_build, prog)
                build_state.prune_source_infos()
        finally:
            con.close()
