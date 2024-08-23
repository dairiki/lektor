from __future__ import annotations

import builtins
import inspect
import os
import sys
import uuid
from fnmatch import fnmatch
from functools import update_wrapper
from itertools import chain
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import MutableMapping
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import babel.dates
import jinja2
from jinja2.loaders import split_template_path

from lektor.constants import PRIMARY_ALT
from lektor.context import config_proxy
from lektor.context import get_asset_url
from lektor.context import get_ctx
from lektor.context import get_locale
from lektor.context import site_proxy
from lektor.context import url_to
from lektor.environment.config import Config
from lektor.environment.config import DEFAULT_CONFIG  # noqa - reexport
from lektor.environment.config import ServerInfo  # noqa - reexport
from lektor.environment.config import update_config_from_ini  # noqa - reexport
from lektor.environment.expressions import Expression  # noqa - reexport
from lektor.environment.expressions import FormatExpression  # noqa - reexport
from lektor.markdown import Markdown
from lektor.packages import load_packages
from lektor.pluginsystem import initialize_plugins
from lektor.pluginsystem import Plugin
from lektor.pluginsystem import PluginController
from lektor.publisher import builtin_publishers
from lektor.utils import format_lat_long
from lektor.utils import tojson_filter

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

    from lektor.assets import Asset
    from lektor.build_programs import BuildProgram
    from lektor.databags import DatabagType
    from lektor.db import Pad
    from lektor.db import Record
    from lektor.project import Project
    from lektor.publisher import Publisher
    from lektor.sourceobj import SourceObject
    from lektor.sourceobj import VirtualSourceObject
    from lektor.types.base import Type


_P = ParamSpec("_P")
_T = TypeVar("_T")
_SourceObj_co = TypeVar("_SourceObj_co", bound="SourceObject", covariant=True)
_Asset_co = TypeVar("_Asset_co", bound="Asset", covariant=True)

UrlResolver = Callable[["Record", Sequence[str]], "Record | None"]
VirtualPathResolver = Callable[["Record", Sequence[str]], "VirtualSourceObject | None"]
SourceGenerator = Callable[["SourceObject"], Iterable["SourceObject"]]

TemplateValuesType = Union[
    "SupportsKeysAndGetItem[str, Any]", Iterable[Tuple[str, Any]]
]


def _prevent_inlining(wrapped: Callable[_P, _T]) -> Callable[_P, _T]:
    """Ensure wrapped jinja filter does not get inlined by the template compiler.

    The jinja compiler normally assumes that filters are pure functions (whose
    result depends only on their parameters) and will inline filter calls that
    are applied to compile-time constants.

    E.g.

        'say {{ "foo" | upper }}'

    will be compiled to

        "say Foo"

    Many of our filters depend on global state (e..g the Lektor build context).

    Applying this decorator to them will ensure they are not inlined.
    """

    # the use of @pass_context will prevent inlining
    @jinja2.pass_context
    def wrapper(
        _jinja_ctx: jinja2.runtime.Context, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        return wrapped(*args, **kwargs)

    return update_wrapper(wrapper, wrapped)  # type: ignore[return-value]


class _BabelFmtDate(Protocol):
    def __call__(self, arg: Any, /, format: str = ..., locale: str | None = ...) -> str:
        ...


class _BabelFmtTime(Protocol):
    # Also matches formt_datetime
    def __call__(
        self,
        arg: Any,
        /,
        format: str = ...,
        tzinfo: Any | None = ...,
        locale: str | None = ...,
    ) -> str:
        ...


_BabelFmt = TypeVar("_BabelFmt", _BabelFmtDate, _BabelFmtTime)


def _dates_filter(name: str, wrapped: _BabelFmt) -> _BabelFmt:
    """Wrap one of the babel.dates.format_* functions for use as a jinja filter.

    This will create a jinja filter that will:

    - Check for *undefined* date/time input (and, in that case, return an empty string).

    - Check that the ``format`` and ``locale`` parameters, if provided, have the correct
      types, otherwise raising ``TypeError``.

    - Raise ``TypeError`` with a somewhat informative message if the wrapped formatting
      function raises an unexpected exception.  Such an exception is most likely due to
      being passed an unsupported date/time time.  (The Babel formatting functions
      accept a fairly wide range of input types — and that range might potentially vary
      between releases — so we do not explicitly check the input type before passing it
      on to Babel.)

    If `locale` is not specified, we fill it in based on the current *alt*.

    """
    signature = inspect.signature(wrapped)

    @_prevent_inlining
    def wrapper(arg: Any, *args: Any, **kwargs: Any) -> str:
        bound = signature.bind(arg, *args, **kwargs)

        format_ = bound.arguments.setdefault("format", "medium")
        if not isinstance(format_, str):
            raise TypeError(
                f"The 'format' parameter to '{name}' should be a str, not {format_!r}"
            )

        if isinstance(arg, jinja2.Undefined):
            # This will typically return an empty string, though it depends on the
            # specific type of undefined instance.  E.g. if arg is a DebugUndefined, it
            # will return a more descriptive message, and if arg is a StrictUndefined,
            # an UndefinedError will be raised.
            return str(arg)

        locale = bound.arguments.get("locale")
        if locale is None:
            bound.arguments["locale"] = get_locale("en_US")

        try:
            return wrapped(*bound.args, **bound.kwargs)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise TypeError(
                f"While evaluating filter '{name}', an unexpected exception was raised. "
                "This is likely caused by an input or parameter of an unsupported type."
            ) from exc

    return update_wrapper(wrapper, wrapped)


@_prevent_inlining
def _markdown_filter(
    source: str,
    *,
    resolve_links: Literal["always", "never", "when-possible", None] = None,
    **kw: str,
) -> Markdown:
    """A jinja filter that converts markdown text to HTML."""
    ctx = get_ctx()
    source_obj = ctx.source if ctx is not None else None
    return Markdown(
        source, source_obj, field_options={**kw, "resolve_links": resolve_links}
    )


# Special files that should always be ignored.
IGNORED_FILES = ["thumbs.db", "desktop.ini", "Icon\r"]

# These files are important for artifacts and must not be ignored when
# they are built even though they start with dots.
SPECIAL_ARTIFACTS = [".htaccess", ".htpasswd"]

# Default glob pattern of ignored files.
EXCLUDED_ASSETS = ["_*", ".*"]

# Default glob pattern of included files (higher-priority than EXCLUDED_ASSETS).
INCLUDED_ASSETS: list[str] = []


class CustomJinjaEnvironment(jinja2.Environment):
    def _load_template(
        self, name: str, globals: MutableMapping[str, Any] | None
    ) -> jinja2.Template:
        ctx = get_ctx()

        try:
            rv = jinja2.Environment._load_template(self, name, globals)
            if ctx is not None:
                filename = rv.filename
                if filename is not None:
                    ctx.record_dependency(filename)
            return rv
        except jinja2.TemplateSyntaxError as e:
            if e.filename is not None and ctx is not None:
                ctx.record_dependency(e.filename)
            raise
        except jinja2.TemplateNotFound as e:
            if ctx is not None:
                # If we can't find the template we want to record at what
                # possible locations the template could exist.  This will help
                # out watcher to pick up templates that will appear in the
                # future.  This assumes the loader is a file system loader.
                for template_name in e.templates:
                    if not isinstance(template_name, (jinja2.Undefined, type(None))):
                        pieces = split_template_path(template_name)
                        assert isinstance(self.loader, jinja2.FileSystemLoader)
                        for base in self.loader.searchpath:
                            ctx.record_dependency(os.path.join(base, *pieces))
            raise


@jinja2.pass_context
def lookup_from_bag(
    jinja_ctx: jinja2.runtime.Context, *args: str
) -> DatabagType | str | None:
    pieces = ".".join(x for x in args if x)
    site: Pad = jinja_ctx.get("site", default=site_proxy)
    return site.databags.lookup(pieces)


class Environment:
    def __init__(
        self,
        project: Project,
        load_plugins: bool = True,
        extra_flags: Sequence[str] | dict[str, str] | None = None,
    ):
        self.project = project
        self.root_path = os.path.abspath(project.tree)

        self.theme_paths: list[str] = [
            os.path.join(self.root_path, "themes", theme)
            for theme in self.project.themes
        ]

        if not self.theme_paths:
            # load the directories in the themes directory as the themes
            try:
                for fname in os.listdir(os.path.join(self.root_path, "themes")):
                    f = os.path.join(self.root_path, "themes", fname)
                    if os.path.isdir(f):
                        self.theme_paths.append(f)
            except OSError:
                pass

        template_paths = [
            os.path.join(path, "templates")
            for path in [self.root_path] + self.theme_paths
        ]

        self.jinja_env = CustomJinjaEnvironment(
            autoescape=self.select_jinja_autoescape,
            extensions=["jinja2.ext.do"],
            loader=jinja2.FileSystemLoader(template_paths),
        )

        from lektor.db import F, get_alts  # pylint: disable=import-outside-toplevel

        def latlongformat(latlong: tuple[float, float], secs: bool = True) -> str:
            lat, lon = latlong
            return format_lat_long(lat=lat, long=lon, secs=secs)

        self.jinja_env.filters.update(
            tojson=tojson_filter,
            latformat=lambda x, secs=True: format_lat_long(lat=x, secs=secs),
            longformat=lambda x, secs=True: format_lat_long(long=x, secs=secs),
            latlongformat=latlongformat,
            url=_prevent_inlining(url_to),
            asseturl=_prevent_inlining(get_asset_url),
            markdown=_markdown_filter,
        )
        self.jinja_env.globals.update(
            F=F,
            url_to=url_to,
            site=site_proxy,
            config=config_proxy,
            bag=lookup_from_bag,
            get_alts=get_alts,
            get_random_id=lambda: uuid.uuid4().hex,
        )
        self.jinja_env.filters.update(
            dateformat=_dates_filter("dateformat", babel.dates.format_date),
            datetimeformat=_dates_filter("datetimeformat", babel.dates.format_datetime),
            timeformat=_dates_filter("timeformat", babel.dates.format_time),
        )

        # pylint: disable=import-outside-toplevel
        from lektor.types import builtin_types

        self.types = builtin_types.copy()

        self.publishers = builtin_publishers.copy()

        # The plugins that are loaded for this environment.  This is
        # modified by the plugin controller and registry methods on the
        # environment.
        self.plugin_controller = PluginController(self, extra_flags)
        self.plugins: dict[str, Plugin] = {}
        self.plugin_ids_by_class: dict[type[Plugin], str] = {}
        self.build_programs = []
        self.special_file_assets = {}
        self.special_file_suffixes = {}
        self.custom_url_resolvers: list[UrlResolver] = []
        self.custom_generators: list[SourceGenerator] = []
        self.virtual_sources: dict[str, VirtualPathResolver] = {}

        if load_plugins:
            self.load_plugins()
        # pylint: disable=import-outside-toplevel
        from lektor.db import siblings_resolver

        self.virtualpathresolver("siblings")(siblings_resolver)

    jinja_env: jinja2.Environment
    plugin_controller: PluginController
    root_path: str

    build_programs: list[tuple[type[SourceObject], type[BuildProgram[SourceObject]]]]
    special_file_assets: dict[str, type[Asset]]
    special_file_suffixes: dict[str, str]

    @property
    def asset_path(self) -> str:
        return os.path.join(self.root_path, "assets")

    @property
    def temp_path(self) -> str:
        return os.path.join(self.root_path, "temp")

    def load_plugins(self) -> None:
        """Loads the plugins."""
        load_packages(self)
        initialize_plugins(self)

    def load_config(self) -> Config:
        """Loads the current config."""
        return Config(self.project.project_file)

    def new_pad(self) -> Pad:
        """Convenience function to create a database and pad."""
        from lektor.db import Database  # pylint: disable=import-outside-toplevel

        return Database(self).new_pad()

    def is_uninteresting_source_name(self, filename: str) -> bool:
        """These files are ignored when sources are built into artifacts."""
        if filename.lower() in SPECIAL_ARTIFACTS:
            return False
        proj = self.project
        include_patterns = chain(INCLUDED_ASSETS, proj.included_assets)
        if any(fnmatch(filename, pat) for pat in include_patterns):
            return False
        exclude_patterns = chain(EXCLUDED_ASSETS, proj.excluded_assets)
        return any(fnmatch(filename, pat) for pat in exclude_patterns)

    @staticmethod
    def is_ignored_artifact(asset_name: str) -> bool:
        """This is used by the prune tool to figure out which files in the
        artifact folder should be ignored.
        """
        fn = asset_name.lower()
        if fn in SPECIAL_ARTIFACTS:
            return False
        return fn[:1] == "." or fn in IGNORED_FILES

    def render_template(
        self,
        name: str | Iterable[str],
        pad: Pad | None = None,
        this: object | None = None,
        values: TemplateValuesType | None = None,
        alt: str | None = None,
    ) -> str:
        if isinstance(name, str):
            template = self.jinja_env.get_template(name)
        else:
            assert isinstance(name, Iterable)  # Iterable[str]
            template = self.jinja_env.select_template(name)
        ctx = self.make_default_tmpl_values(
            pad, this, values, alt, template=template.name
        )
        return template.render(ctx)

    def make_default_tmpl_values(
        self,
        pad: Pad | None = None,
        this: object | None = None,
        values: TemplateValuesType | None = None,
        alt: str | None = None,
        template: str | None = None,
    ) -> dict[str, Any]:
        values = dict(values or ())

        # If not provided, pick the alt from the provided "this" object.
        # As there is no mandatory format for it, we make sure that we can
        # deal with a bad attribute there.
        if alt is None:
            if this is not None:
                alt = getattr(this, "alt", None)
                if not isinstance(alt, str):
                    alt = None
            if alt is None:
                alt = PRIMARY_ALT

        # This is already a global variable but we can inject it as a
        # local override if available.
        if pad is None:
            ctx = get_ctx()
            if ctx is not None:
                pad = ctx.pad
        if pad is not None:
            values["site"] = pad
        if this is not None:
            values["this"] = this
        if alt is not None:
            values["alt"] = alt
        self.plugin_controller.emit(
            "process-template-context", context=values, template=template
        )
        return values

    @staticmethod
    def select_jinja_autoescape(filename: str | None) -> bool:
        if filename is None:
            return False
        return filename.endswith((".html", ".htm", ".xml", ".xhtml"))

    def resolve_custom_url_path(
        self, obj: Record, url_path: Sequence[str]
    ) -> SourceObject | None:  # XXX: is annotation correct?
        for resolver in self.custom_url_resolvers:
            rv = resolver(obj, url_path)
            if rv is not None:
                return rv
        return None

    # -- methods for the plugin system

    def add_build_program(
        self, cls: type[_SourceObj_co], program: type[BuildProgram[_SourceObj_co]]
    ) -> None:
        self.build_programs.append((cls, program))

    def add_asset_type(
        self, asset_cls: type[_Asset_co], build_program: type[BuildProgram[_Asset_co]]
    ) -> None:
        self.build_programs.append((asset_cls, build_program))
        # XXX: this is identical to add_build_program, except in annotation
        # XXX: this is unused and broken.  Assets do not have a source_extension attribute
        # self.special_file_assets[asset_cls.source_extension] = asset_cls
        # if asset_cls.artifact_extension:
        #     cext = asset_cls.source_extension + asset_cls.artifact_extension
        #     self.special_file_suffixes[cext] = asset_cls.source_extension

    def add_publisher(self, scheme: str, publisher: type[Publisher]) -> None:
        if scheme in self.publishers:
            raise RuntimeError('Scheme "%s" is already registered.' % scheme)
        self.publishers[scheme] = publisher

    def add_type(self, type: builtins.type[Type]) -> None:
        name = type.name
        if name in self.types:
            raise RuntimeError('Type "%s" is already registered.' % name)
        self.types[name] = type

    def virtualpathresolver(
        self, prefix: str
    ) -> Callable[[VirtualPathResolver], VirtualPathResolver]:
        def decorator(func: VirtualPathResolver) -> VirtualPathResolver:
            if prefix in self.virtual_sources:
                raise RuntimeError('Prefix "%s" is already registered.' % prefix)
            self.virtual_sources[prefix] = func
            return func

        return decorator

    def urlresolver(self, func: UrlResolver) -> UrlResolver:
        self.custom_url_resolvers.append(func)
        return func

    def generator(self, func: SourceGenerator) -> SourceGenerator:
        self.custom_generators.append(func)
        return func
