from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from dataclasses import field
from itertools import chain
from typing import Any
from typing import Callable
from typing import Collection
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import TYPE_CHECKING
from typing import TypeVar

from lektor.assets import Asset
from lektor.assets import Directory
from lektor.assets import File
from lektor.constants import PRIMARY_ALT
from lektor.db import Attachment
from lektor.db import Page
from lektor.db import Record
from lektor.exception import LektorException
from lektor.i18n import I18nBlock
from lektor.sourceobj import SourceObject

if TYPE_CHECKING:
    from _typeshed import StrPath

    from lektor.builder import ArtifactTransaction
    from lektor.builder import Artifact
    from lektor.builder import BuildState
    from lektor.builder import SourceId


_SourceObj_co = TypeVar("_SourceObj_co", covariant=True, bound=SourceObject)


class BuildError(LektorException):
    pass


builtin_build_programs: list[
    tuple[type[SourceObject], type[BuildProgram[SourceObject]]]
] = []


def buildprogram(
    source_cls: type[_SourceObj_co],
) -> Callable[[type[BuildProgram[_SourceObj_co]]], type[BuildProgram[_SourceObj_co]]]:
    def decorator(
        builder_cls: type[BuildProgram[_SourceObj_co]],
    ) -> type[BuildProgram[_SourceObj_co]]:
        builtin_build_programs.append((source_cls, builder_cls))
        return builder_cls

    return decorator


@dataclass
class SourceInfo:
    """Holds some information about a source file for indexing into the
    build state.
    """

    path: str
    filename: SourceId | StrPath  # (see note below)
    alt: str = PRIMARY_ALT
    type: str = "unknown"
    title_i18n: I18nBlock = field(default_factory=I18nBlock)

    # XXX: currently the filename is always normalized (relative with forward slashes,
    # even on windows) when the SourceInfo is loaded from the build database. However
    # the filenames that come from Record.iter_source_filenames() are not normalized.
    # (This should probably be fixed at some point, so that all source paths are
    # normalized the same.)

    # XXX: currently the filename is always normalized (relative with forward slashes,
    # even on windows) when the SourceInfo is loaded from the build database. However
    # the filenames that come from Record.iter_source_filenames() are not normalized.
    # (This should probably be fixed at some point, so that all source paths are
    # normalized the same.)

    def __post_init__(self) -> None:
        title_i18n = self.title_i18n
        en_title = title_i18n.setdefault("en", self.path)
        for lang in list(title_i18n):
            if lang != "en" and title_i18n[lang] == en_title:
                del title_i18n[lang]

    @property
    def title_en(self) -> str:
        return self.title_i18n.get("en", self.path)


class BuildProgram(Generic[_SourceObj_co]):
    def __init__(self, source: _SourceObj_co, build_state: BuildState):
        self.source = source
        self.build_state = build_state
        self.artifacts: list[Artifact] = []
        self._built = False

    @property
    def primary_artifact(self) -> Artifact | None:
        """Returns the primary artifact for this build program.  By
        default this is the first artifact produced.  This needs to be the
        one that corresponds to the URL of the source if it has one.
        """
        return next(iter(self.get_artifacts()), None)

    def describe_source_record(self) -> SourceInfo | None:
        """Can be used to describe the source info by returning a
        :class:`SourceInfo` object.  This is indexed by the builder into
        the build state so that the UI can quickly find files without
        having to scan the file system.
        """

    def get_artifacts(self) -> Collection[Artifact]:
        """Get top-level artifacts for this build program.

        Normally, a program will produce at most one top-level artifact, however
        each top-level artifact, when built, may produce additional sub-artifacts which
        also are to be built as part of the build process for this record.

        """
        if len(self.artifacts) == 0:
            self.produce_artifacts()
        return self.artifacts

    def produce_artifacts(self) -> None:
        """This produces the artifacts for building.  Usually this only
        produces a single artifact.
        """

    def declare_artifact(
        self,
        artifact_name: str,
        sources: Collection[str] | None = None,
        extra: Any = None,  # FIXME: unused?
    ) -> None:
        """This declares an artifact to be built in this program."""
        self.artifacts.append(
            self.build_state.new_artifact(
                artifact_name=artifact_name,
                sources=sources,
                source_obj=self.source,
                extra=extra,
            )
        )

    def build_artifact(self, artifact_txn: ArtifactTransaction, /) -> None:
        """This is invoked for each artifact declared."""

    def iter_child_sources(self) -> Iterator[SourceObject]:
        """This allows a build program to produce children that also need
        building.  An individual build never recurses down to this, but
        a `build_all` will use this.
        """
        # pylint: disable=no-self-use
        return iter(())


@buildprogram(Page)
class PageBuildProgram(BuildProgram[Page]):
    def describe_source_record(self) -> SourceInfo | None:
        # When we describe the source record we need to consider that a
        # page has multiple source file names but only one will actually
        # be used.  The order of the source iter is in order the files are
        # attempted to be read.  So we go with the first that actually
        # exists and then return that.
        for filename in self.source.iter_source_filenames():
            if os.path.isfile(filename):
                return SourceInfo(
                    path=self.source.path,
                    alt=self.source["_source_alt"],
                    filename=filename,
                    type="page",
                    title_i18n=self.source.get_record_label_i18n(),
                )
        return None

    def produce_artifacts(self) -> None:
        pagination_enabled = self.source.datamodel.pagination_config.enabled

        if self.source.is_visible and (
            self.source.page_num is not None or not pagination_enabled
        ):
            artifact_name = self.source.url_path
            if artifact_name.endswith("/"):
                artifact_name += "index.html"

            self.declare_artifact(
                artifact_name, sources=list(self.source.iter_source_filenames())
            )

    def build_artifact(self, artifact_txn: ArtifactTransaction, /) -> None:
        # Record dependecies on all our sources and datamodel
        self.source.pad.db.track_record_dependency(self.source)

        try:
            self.source.url_path.encode("ascii")
        except UnicodeError as error:
            raise BuildError(
                "The URL for this record contains non ASCII "
                "characters.  This is currently not supported "
                "for portability reasons (%r)." % self.source.url_path
            ) from error

        artifact_txn.render_template_into(self.source["_template"], this=self.source)

    def _iter_paginated_children(self) -> Iterator[Page]:
        total = self.source.datamodel.pagination_config.count_pages(self.source)
        for page_num in range(1, total + 1):
            yield Page(self.source.pad, self.source._data, page_num=page_num)

    def iter_child_sources(self) -> Iterator[Record]:
        p_config = self.source.datamodel.pagination_config
        pagination_enabled = p_config.enabled
        child_sources: list[Iterable[Record]] = []

        # So this requires a bit of explanation:
        #
        # the basic logic is that if we have pagination enabled then we
        # need to consider two cases:
        #
        # 1. our build program has page_num = None which means that we
        #    are not yet pointing to a page.  In that case we want to
        #    iter over all children which will yield the pages.
        # 2. we are pointing to a page, then our child sources are the
        #    items that are shown on that page.
        #
        # In addition, attachments and pages excluded from pagination are
        # linked to the page with page_num = None.
        #
        # If pagination is disabled, all children and attachments are linked
        # to this page.
        all_children = self.source.children.include_undiscoverable(True)
        all_children = all_children.include_hidden(True)
        if pagination_enabled:
            if self.source.page_num is None:
                child_sources.append(self._iter_paginated_children())
                pq = p_config.get_pagination_query(self.source)
                child_sources.append(set(all_children) - set(pq))
                child_sources.append(self.source.attachments)
            else:
                child_sources.append(self.source.pagination.items)
        else:
            child_sources.append(all_children)
            child_sources.append(self.source.attachments)

        return chain(*child_sources)


@buildprogram(Attachment)
class AttachmentBuildProgram(BuildProgram[Attachment]):
    def describe_source_record(self) -> SourceInfo:
        return SourceInfo(
            path=self.source.path,
            alt=self.source.alt,
            filename=self.source.attachment_filename,
            type="attachment",
            title_i18n=I18nBlock(en=self.source["_id"]),
        )

    def produce_artifacts(self) -> None:
        primary_alt = self.build_state.config.primary_alternative or PRIMARY_ALT
        if self.source.is_visible and self.source.alt == primary_alt:
            self.declare_artifact(
                self.source.url_path, sources=list(self.source.iter_source_filenames())
            )

    def build_artifact(self, artifact_txn: ArtifactTransaction, /) -> None:
        with artifact_txn.open("wb") as df:
            with open(self.source.attachment_filename, "rb") as sf:
                shutil.copyfileobj(sf, df)


@buildprogram(File)
class FileAssetBuildProgram(BuildProgram[File]):
    def produce_artifacts(self) -> None:
        self.declare_artifact(
            self.source.artifact_name, sources=[self.source.source_filename]
        )

    def build_artifact(self, artifact_txn: ArtifactTransaction, /) -> None:
        with artifact_txn.open("wb") as df:
            with open(self.source.source_filename, "rb") as sf:
                shutil.copyfileobj(sf, df)


@buildprogram(Directory)
class DirectoryAssetBuildProgram(BuildProgram[Directory]):
    def iter_child_sources(self) -> Iterator[Asset]:
        return iter(self.source.children)
