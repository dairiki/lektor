from __future__ import annotations

import sys
from collections.abc import Hashable
from contextlib import contextmanager
from functools import cached_property
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import TYPE_CHECKING

from jinja2 import Undefined
from werkzeug.local import LocalProxy
from werkzeug.local import LocalStack

from lektor.reporter import reporter
from lektor.utils import deprecated

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from _typeshed import StrPath
    from _typeshed import Unused

    from lektor.assets import Asset
    from lektor.builder import ArtifactTransaction
    from lektor.builder import ArtifactBuildFunc
    from lektor.builder import SubArtifact
    from lektor.builder import BuildState
    from lektor.db import Pad
    from lektor.db import Record
    from lektor.environment.config import Config
    from lektor.environment import Environment
    from lektor.sourceobj import SourceObject
    from lektor.sourceobj import VirtualSourceObject
    from lektor.types.flow import FlowBlock
    from lektor.typing import SupportsUrlPath


_ctx_stack = LocalStack["Context"]()


def url_to(
    path: str | SourceObject | SupportsUrlPath,
    alt: str | None = None,
    absolute: bool | None = None,
    external: bool | None = None,
    resolve: bool | None = None,
    strict_resolve: bool | None = None,
) -> str:
    """Calculates a URL to another record."""
    ctx = get_ctx()
    if ctx is None:
        raise RuntimeError("No context found")
    return ctx.url_to(path, alt, absolute, external, resolve, strict_resolve)


def get_asset_url(asset_path: str, /) -> str | Undefined:
    """Calculates the asset URL relative to the current record."""
    ctx = get_ctx()
    if ctx is None:
        raise RuntimeError("No context found")
    asset = ctx.pad.get_asset(asset_path)
    if asset is None:
        return Undefined("Asset not found")
    return ctx.get_asset_url(asset)


@LocalProxy
def site_proxy() -> Pad | Undefined:
    """Returns the current pad."""
    ctx = get_ctx()
    if ctx is None:
        return Undefined(hint="Cannot access the site from here", name="site")
    return ctx.pad


@LocalProxy
def config_proxy() -> Config | None:  # FIXME: Is None right?
    """Returns the current config."""
    ctx = get_ctx()
    if ctx is not None:
        return ctx.pad.db.config
    return None


def get_ctx() -> Context | None:
    """Returns the current context."""
    return _ctx_stack.top


def get_locale(default: str = "en_US") -> str:
    """Returns the current locale."""
    ctx = get_ctx()
    if ctx is not None:
        rv = ctx.locale
        if rv is not None:
            return rv
        return ctx.pad.db.config.site_locale
    return default


DependencyCollector = Callable[["str | VirtualSourceObject"], None]


class Context:
    """The context is a thread local object that provides the system with
    general information about in which state it is.  The context is created
    whenever a source is processed and can be accessed by template engine and
    other things.

    It's considered read and write and also accumulates changes that happen
    during processing of the object.
    """

    def __init__(self, artifact_txn: ArtifactTransaction):
        self.artifact_txn = artifact_txn

        # Processing information

        self.sub_artifacts: list[SubArtifact] = []

        self.flow_block_render_stack: list[FlowBlock] = []

        self._forced_base_url: str | None = None
        self._resolving_url = False

        # General cache system where other things can put their temporary
        # stuff in.
        self.cache: dict[Hashable, Any] = {}

        self._dependency_collectors: list[DependencyCollector] = []

    @property
    @deprecated(
        "Context.artifact has been renamed to Context.artifact_txn.",
        version="3.4.0",
    )
    def artifact(self) -> ArtifactTransaction:
        return self.artifact_txn

    @property
    def referenced_dependencies(self) -> set[StrPath]:
        return self.artifact_txn._referenced_source_files

    @property
    def referenced_virtual_dependencies(self) -> set[VirtualSourceObject]:
        return self.artifact_txn._referenced_virtual_sources

    @property
    def source(self) -> SourceObject | None:  # XXX: can this really be None?
        return self.artifact_txn.source_obj

    @cached_property
    def build_state(self) -> BuildState:
        return self.artifact_txn.build_state

    @cached_property
    def pad(self) -> Pad:
        return self.build_state.builder.pad

    @cached_property
    def env(self) -> Environment:
        """The environment of the context."""
        return self.pad.db.env

    @property
    def record(self) -> Record | None:
        """If the source is a record it will be available here."""
        source = self.source
        if isinstance(source, Record):
            assert source.source_classification == "record"
            return source
        return None

    @property
    def locale(self) -> str | None:
        """Returns the current locale if it's available, otherwise `None`.
        This does not fall back to the site locale.
        """
        source = self.source
        if source is not None:
            alt_cfg = self.pad.db.config["ALTERNATIVES"].get(source.alt)
            if alt_cfg:
                return alt_cfg["locale"]
        return None

    def push(self) -> None:
        _ctx_stack.push(self)

    @staticmethod
    def pop() -> None:
        _ctx_stack.pop()

    def __enter__(self) -> Self:
        self.push()
        return self

    def __exit__(self, exc_type: Unused, exc_value: Unused, tb: Unused) -> None:
        self.pop()

    @property
    def base_url(self) -> str:
        """The URL path for the current context."""
        if self._forced_base_url:
            return self._forced_base_url
        if self.source is not None:
            return self.source.url_path
        return "/"

    def url_to(
        self,
        path: str | SourceObject | SupportsUrlPath,
        alt: str | None = None,
        absolute: bool | None = None,
        external: bool | None = None,
        resolve: bool | None = None,
        strict_resolve: bool | None = None,
    ) -> str:
        """Returns a URL to another path."""
        if self.source is None:
            raise RuntimeError(
                "Can only generate paths to other pages if "
                "the context has a source document set."
            )
        return self.source.url_to(
            path,
            alt=alt,
            base_url=self.base_url,
            absolute=absolute,
            external=external,
            resolve=resolve,
            strict_resolve=strict_resolve,
        )

    def get_asset_url(self, asset: Asset) -> str:
        """Calculates the asset URL relative to the current record."""
        if self.source is None:
            raise RuntimeError(
                "Can only generate paths to assets if "
                "the context has a source document set."
            )
        if self.build_state is None:
            raise ValueError(
                "Can only generate paths to assets if "
                "the context has a build state set."
            )
        asset_url = self.source.url_to(asset.url_path, resolve=False)
        info = self.build_state.get_file_info(asset.source_filename)
        self.record_dependency(asset.source_filename)
        return f"{asset_url}?h={info.checksum[:8]}"

    def sub_artifact(
        self,
        artifact_name: str,
        sources: Collection[str] | None = None,
        source_obj: SourceObject | None = None,
        config_hash: str | None = None,
    ) -> Callable[[ArtifactBuildFunc], ArtifactBuildFunc]:
        """Decorator version of :func:`add_sub_artifact`."""

        def decorator(build_func: ArtifactBuildFunc) -> ArtifactBuildFunc:
            self.add_sub_artifact(
                artifact_name, build_func, sources, source_obj, config_hash
            )
            return build_func

        return decorator

    def add_sub_artifact(
        self,
        artifact_name: str,
        build_func: ArtifactBuildFunc,  # | None = None,  # XXX: does None make sense?
        sources: Collection[str] | None = None,
        source_obj: SourceObject | None = None,
        config_hash: str | None = None,
    ) -> None:
        """Sometimes it can happen that while building an artifact another
        artifact needs building.  This function is generally used to record
        this request.
        """
        # pylint: disable=import-outside-toplevel
        from lektor.builder import SubArtifact  # FIXME: circdep

        artifact_txn = self.artifact_txn
        if artifact_txn is None:
            raise TypeError(
                "The context does not have a build state which "
                "means that artifact declaration is not possible."
            )
        artifact = artifact_txn.build_state.new_artifact(
            artifact_name=artifact_name,
            sources=sources,
            source_obj=source_obj,
            config_hash=config_hash,
            parent_artifact=artifact_txn.artifact,
        )
        self.sub_artifacts.append(SubArtifact(artifact, build_func))
        reporter.report_sub_artifact(artifact)

    def record_dependency(self, filename: str, affects_url: bool | None = None) -> None:
        """Records a dependency from processing.

        If ``affects_url`` is set to ``False`` the dependency will be ignored if
        we are in the process of resolving a URL.
        """
        if self._resolving_url and affects_url is False:
            return
        self.artifact_txn._referenced_source_files.add(filename)
        for coll in self._dependency_collectors:
            coll(filename)

    def record_virtual_dependency(self, virtual_source: VirtualSourceObject) -> None:
        """Records a dependency from processing."""
        self.artifact_txn._referenced_virtual_sources.add(virtual_source)
        for coll in self._dependency_collectors:
            coll(virtual_source)

    @contextmanager
    def gather_dependencies(self, func: DependencyCollector) -> Iterator[None]:
        """For the duration of the `with` block the provided function will be
        invoked for all dependencies encountered.
        """
        self._dependency_collectors.append(func)
        try:
            yield
        finally:
            self._dependency_collectors.pop()

    @contextmanager
    def changed_base_url(self, value: str) -> Iterator[None]:
        """Temporarily overrides the URL path of the context."""
        old = self._forced_base_url
        self._forced_base_url = value
        try:
            yield
        finally:
            self._forced_base_url = old


@contextmanager
def ignore_url_unaffecting_dependencies(value: bool = True) -> Iterator[None]:
    """Ignore dependencies which do not affect URL resolution within context."""
    ctx = get_ctx()
    if ctx is None:
        yield
        return

    old = ctx._resolving_url
    ctx._resolving_url = value
    try:
        yield
    finally:
        ctx._resolving_url = old


@contextmanager
def disable_dependency_recording() -> Iterator[None]:
    """Disable dependency recording within context

    We do this by pushing a new context onto the context stack.  The
    new context proxies all access — except for dependency reporting —
    back to the original context.

    """
    ctx = get_ctx()
    if ctx is None:
        yield
        return

    with DependencyIgnoringContextProxy(ctx):
        yield


class DependencyIgnoringContextProxy(Context):
    # pylint: disable=super-init-not-called

    __slots__ = ["_ctx"]

    def __init__(self, ctx: Context):
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    def record_dependency(self, filename: str, affects_url: bool | None = None) -> None:
        pass

    def record_virtual_dependency(self, virtual_source: VirtualSourceObject) -> None:
        pass
