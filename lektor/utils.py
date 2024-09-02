from __future__ import annotations

import codecs
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import threading
import unicodedata
import urllib.parse
import uuid
import warnings
from collections import defaultdict
from collections.abc import Hashable
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from functools import wraps
from pathlib import PurePosixPath
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from jinja2 import is_undefined
from markupsafe import Markup
from slugify import slugify as _slugify
from werkzeug.http import http_date
from werkzeug.urls import iri_to_uri
from werkzeug.urls import uri_to_iri

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from _typeshed import StrPath
    from _typeshed import Unused


_F = TypeVar("_F", bound=Callable[..., Any])
_H = TypeVar("_H", bound=Hashable)
_T = TypeVar("_T")
_U = TypeVar("_U")
_T_co = TypeVar("_T_co", covariant=True)


class _MappingFactory(Protocol, Generic[_T, _U]):
    def __call__(self, data: Iterable[tuple[_T, _U]], /) -> Mapping[_T, _U]:
        ...


is_windows = os.name == "nt"

_slash_escape = "\\/" not in json.dumps("/")

_last_num_re = re.compile(r"^(.*)(\d+)(.*?)$")
_list_marker = object()
_value_marker = object()

# Figure out our fs encoding, if it's ascii we upgrade to utf-8
fs_enc = sys.getfilesystemencoding()
try:
    if codecs.lookup(fs_enc).name == "ascii":
        fs_enc = "utf-8"
except LookupError:
    pass


def split_virtual_path(path):
    if "@" in path:
        return path.split("@", 1)
    return path, None


def _norm_join(a, b):
    return posixpath.normpath(posixpath.join(a, b))


def join_path(a, b):
    """Join two DB-paths.

    It is assumed that both paths are already normalized in that
    neither contains an extra "." or ".." components, double-slashes,
    etc.
    """
    # NB: This function is really only during URL resolution.  The only
    # place that references it is lektor.source.SourceObject._resolve_url.

    if posixpath.isabs(b):
        return b

    a_p, a_v = split_virtual_path(a)
    b_p, b_v = split_virtual_path(b)

    # Special case: paginations are considered special virtual paths
    # where the parent is the actual parent of the page.  This however
    # is explicitly not done if the path we join with refers to the
    # current path (empty string or dot).
    if b_p not in ("", ".") and a_v and a_v.isdigit():
        a_v = None

    # New path has a virtual path, add that to it.
    if b_v:
        rv = _norm_join(a_p, b_p) + "@" + b_v
    elif a_v:
        rv = a_p + "@" + _norm_join(a_v, b_p)
    else:
        rv = _norm_join(a_p, b_p)
    if rv[-2:] == "@.":
        rv = rv[:-2]
    return rv


def cleanup_path(path):
    # NB: POSIX allows for two leading slashes in a pathname, so we have to
    # deal with the possiblity of leading double-slash ourself.
    return posixpath.normpath("/" + path.lstrip("/"))


def cleanup_url_path(url_path):
    """Clean up a URL path.

    This strips any query, and/or fragment that may be present in the
    input path.

    Raises ValueError if the path contains a _scheme_
    which is neither ``http`` nor ``https``, or a _netloc_.
    """
    scheme, netloc, path, _, _ = urllib.parse.urlsplit(url_path, scheme="http")
    if scheme not in ("http", "https"):
        raise ValueError(f"Invalid scheme: {url_path!r}")
    if netloc:
        raise ValueError(f"Invalid netloc: {url_path!r}")

    # NB: POSIX allows for two leading slashes in a pathname, so we have to
    # deal with the possiblity of leading double-slash ourself.
    return posixpath.normpath("/" + path.lstrip("/"))


def parse_path(path):
    x = cleanup_path(path).strip("/").split("/")
    if x == [""]:
        return []
    return x


def is_path_child_of(a, b, strict=True):
    a_p, a_v = split_virtual_path(a)
    b_p, b_v = split_virtual_path(b)
    a_p = parse_path(a_p)
    b_p = parse_path(b_p)
    a_v = parse_path(a_v or "")
    b_v = parse_path(b_v or "")

    if not strict and a_p == b_p and a_v == b_v:
        return True
    if not a_v and b_v:
        return False
    if a_p == b_p and a_v[: len(b_v)] == b_v and len(a_v) > len(b_v):
        return True
    return a_p[: len(b_p)] == b_p and len(a_p) > len(b_p)


def untrusted_to_os_path(path: str) -> str:
    # FIXME: This is used in Database.to_fs_path, to convert DB paths to relative paths.
    # Hence it strips any leading slashes.
    #
    # BUT it is also used by Project.from_file, where it is used to sanitize filesystem
    # paths?  I'm not sure that we want to strip leading slashes in that case.  In any
    # case for filesystem paths should just use os.path.* or pathlib.Path operations?
    assert isinstance(path, str)
    clean_path = cleanup_path(path)
    assert clean_path.startswith("/")
    return clean_path[1:].replace("/", os.path.sep)


def is_path(path):
    return os.path.sep in path or (os.path.altsep and os.path.altsep in path)


def magic_split_ext(filename, ext_check=True):
    """Splits a filename into base and extension.  If ext check is enabled
    (which is the default) then it verifies the extension is at least
    reasonable.
    """

    def bad_ext(ext):
        if not ext_check:
            return False
        if not ext or ext.split() != [ext] or ext.strip() != ext:
            return True
        return False

    parts = filename.rsplit(".", 2)
    if len(parts) == 1:
        return parts[0], ""
    if len(parts) == 2 and not parts[0]:
        return "." + parts[1], ""
    if len(parts) == 3 and len(parts[1]) < 5:
        ext = ".".join(parts[1:])
        if not bad_ext(ext):
            return parts[0], ext
    ext = parts[-1]
    if bad_ext(ext):
        return filename, ""
    basename = ".".join(parts[:-1])
    return basename, ext


def iter_dotted_path_prefixes(dotted_path: str) -> Iterator[tuple[str, str | None]]:
    pieces = dotted_path.split(".")
    if len(pieces) == 1:
        yield dotted_path, None
    else:
        for x in range(1, len(pieces)):
            yield ".".join(pieces[:x]), ".".join(pieces[x:])


def resolve_dotted_value(obj: object | None, dotted_path: str) -> Any:
    node = obj
    for key in dotted_path.split("."):
        if isinstance(node, dict):
            new_node = node.get(key)
            if new_node is None and key.isdigit():
                new_node = node.get(int(key))
        elif isinstance(node, list):
            try:
                new_node = node[int(key)]
            except (ValueError, TypeError, IndexError):
                new_node = None
        else:
            new_node = None
        node = new_node
        if node is None:
            break
    return node


DecodedFlatDataType = Union[
    # The key type should probably just be `str` rather than `str | int`,
    # but here we match current behavior.
    #
    # Currently:
    #     decode_flat_data([("1", "x"), ("a", "y")]) == {1: "x", "a": "y"}
    # While:
    #     decode_flat_data([("a", "x"), ("1", "y")])
    # raises a TypeError
    #
    # They should probably both return {"1": "x", "a": "y"}
    Mapping[Union[str, int], Union[_T, "DecodedFlatDataType[_T]"]],
    Sequence[Union[_T, "DecodedFlatDataType[_T]"]],
]


def decode_flat_data(
    itemiter: Iterable[tuple[str, _T_co]],
    dict_cls: _MappingFactory[str | int, _T_co | DecodedFlatDataType[_T_co]] = dict,
) -> DecodedFlatDataType[_T_co]:
    class _Node:
        value: _T_co
        subnodes: dict[str | int, _Node]  # pylint: disable=undefined-variable

        def __init__(self):
            self.subnodes = defaultdict(_Node)

    top = _Node()

    for key, value in itemiter:
        node = top
        for part in key.split("."):
            node = node.subnodes[part if not part.isdigit() else int(part)]
        node.value = value

    assert not hasattr(top, "value")

    def _result(node: _Node) -> DecodedFlatDataType[_T_co] | _T_co:
        has_value = hasattr(node, "value")
        subnodes = node.subnodes

        if has_value and not subnodes:
            # NB: value ignored if subnodes
            return node.value

        last_key = next(reversed(subnodes.keys()), None)
        force_list = isinstance(last_key, int)

        # XXX: here we duplicate original behavior which ignores force_list
        # if self has a value. Perhaps we should change this?
        if force_list and not has_value:
            # NB: note that this raises TypeError if subnodes has keys
            # of mixed str and int types. This is what the original did.
            return [_result(subnodes[key]) for key in sorted(subnodes)]

        return dict_cls((key, _result(node_)) for key, node_ in subnodes.items())

    rv = _result(top)
    assert isinstance(rv, (Mapping, Sequence))
    return rv


@overload
def merge(a: _T, b: None) -> _T:
    ...


@overload
def merge(a: None, b: _T) -> _T:
    ...


@overload
def merge(a: list[_T], b: list[_U]) -> list[_T | _U]:
    ...


@overload
def merge(a: dict[str, _T], b: dict[str, _U]) -> dict[str, _T | _U]:
    ...


@overload
def merge(a: _T, b: _U) -> _T:
    ...


def merge(a: Any, b: Any) -> Any:
    """Merges two values together."""
    if b is None and a is not None:
        return a
    if a is None:
        return b
    if isinstance(a, list) and isinstance(b, list):
        for idx, (item_1, item_2) in enumerate(zip(a, b)):
            a[idx] = merge(item_1, item_2)
    if isinstance(a, dict) and isinstance(b, dict):
        for key, value in b.items():
            a[key] = merge(a.get(key), value)
        return a
    return a


def slugify(text: str) -> str:
    """
    A wrapper around python-slugify which preserves file extensions
    and forward slashes.
    """

    parts = text.split("/")
    parts[-1], ext = magic_split_ext(parts[-1])

    out = "/".join(_slugify(part) for part in parts)

    if ext:
        return out + "." + ext
    return out


def secure_filename(filename, fallback_name="file"):
    base = filename.replace("/", " ").replace("\\", " ")
    basename, ext = magic_split_ext(base)
    rv = slugify(basename).lstrip(".")
    if not rv:
        rv = fallback_name
    if ext:
        return rv + "." + ext
    return rv


def increment_filename(filename):
    directory, filename = os.path.split(filename)
    basename, ext = magic_split_ext(filename, ext_check=False)

    match = _last_num_re.match(basename)
    if match is not None:
        rv = match.group(1) + str(int(match.group(2)) + 1) + match.group(3)
    else:
        rv = basename + "2"

    if ext:
        rv += "." + ext
    if directory:
        return os.path.join(directory, rv)
    return rv


@lru_cache(maxsize=None)
def locate_executable(exe_file, cwd=None, include_bundle_path=True):
    """Locates an executable in the search path."""
    choices = [exe_file]
    resolve = True

    # If it's already a path, we don't resolve.
    if os.path.sep in exe_file or (os.path.altsep and os.path.altsep in exe_file):
        resolve = False

    extensions = os.environ.get("PATHEXT", "").split(";")
    _, ext = os.path.splitext(exe_file)
    if (
        os.name != "nt"
        and "" not in extensions
        or any(ext.lower() == extension.lower() for extension in extensions)
    ):
        extensions.insert(0, "")

    if resolve:
        paths = os.environ.get("PATH", "").split(os.pathsep)
        choices = [os.path.join(path, exe_file) for path in paths]

    if os.name == "nt":
        choices.append(os.path.join((cwd or os.getcwd()), exe_file))

    try:
        for path in choices:
            for ext in extensions:
                if os.access(path + ext, os.X_OK):
                    return path + ext
        return None
    except OSError:
        return None


class JSONEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if is_undefined(o):
            return None
        if isinstance(o, datetime):
            return http_date(o)
        if isinstance(o, uuid.UUID):
            return str(o)
        if hasattr(o, "__html__"):
            return str(o.__html__())
        return json.JSONEncoder.default(self, o)


def htmlsafe_json_dump(obj, **kwargs):
    kwargs.setdefault("cls", JSONEncoder)
    rv = (
        json.dumps(obj, **kwargs)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )
    if not _slash_escape:
        rv = rv.replace("\\/", "/")
    return rv


def tojson_filter(obj, **kwargs):
    return Markup(htmlsafe_json_dump(obj, **kwargs))


class Url(urllib.parse.SplitResult):
    """Make various parts of a URL accessible.

    This is the type of the values exposed by Lektor record fields of type "url".

    Since Lektor 3.4.0, this is essentially a `urllib.parse.SplitResult` as obtained by
    calling `urlsplit` on the URL normalized to an IRI.

    Generally, attributes such as ``netloc``, ``host``, ``path``, ``query``, and
    ``fragment`` return the IRI (internationalied) versions of those components.

    The URI (ASCII-encoded) version of the URL is available from the `ascii_url`
    attribute.

    NB: Changed in 3.4.0: The ``query`` attribute used to return the URI
    (ASCII-encoded) version of the query — I'm not sure why. Now it returns
    the IRI (internationalized) version of the query.

    """

    url: str

    def __new__(cls: type[Self], value: str) -> Self:
        # XXX: deprecate use of constructor so that eventually we can make its signature
        # match that of the SplitResult base class.
        warnings.warn(
            DeprecatedWarning(
                "Url",
                reason=(
                    "Direct construction of a Url instance is deprecated. "
                    "Use the Url.from_string classmethod instead."
                ),
                version="3.4.0",
            ),
            stacklevel=2,
        )
        return cls.from_string(value)

    @classmethod
    def from_string(cls: type[Self], value: str) -> Self:
        """Construct instance from URL string.

        The input URL can be a URI (all ASCII) or an IRI (internationalized).
        """
        # The iri_to_uri operation is nominally idempotent — it can be passed either an
        # IRI or a URI (or something inbetween) and will return a URI.  So to fully
        # normalize input which can be either an IRI or a URI, first convert to URI,
        # then to IRI.
        iri = uri_to_iri(iri_to_uri(value))
        obj = cls._make(urllib.parse.urlsplit(iri))
        obj.url = value
        return obj

    def __str__(self) -> str:
        """The original un-normalized URL string."""
        return self.url

    @property
    def ascii_url(self) -> str:
        """The URL encoded to an all-ASCII URI."""
        return iri_to_uri(self.geturl())

    @property
    def ascii_host(self) -> str | None:
        """The hostname part of the URL IDNA-encoded to ASCII."""
        return urllib.parse.urlsplit(self.ascii_url).hostname

    @property
    def host(self) -> str | None:
        """The IRI (internationalized) version of the hostname.

        This attribute is provided for backwards-compatibility.  New code should use the
        ``hostname`` attribute instead.
        """
        return self.hostname

    @property
    def anchor(self) -> str:
        """The IRI (internationalized) version of the "anchor" part of the URL.

        This attribute is provided for backwards-compatibility.  New code should use the
        ``fragment`` attribute instead.
        """
        return self.fragment


def is_unsafe_to_delete(path: StrPath, base: StrPath) -> bool:
    a = os.path.abspath(path)
    b = os.path.abspath(base)
    diff = os.path.relpath(a, b)
    first = diff.split(os.path.sep, maxsplit=1)[0]
    return first in (os.path.curdir, os.path.pardir)


def prune_file_and_folder(name: StrPath, base: StrPath) -> bool:
    if is_unsafe_to_delete(name, base):
        return False
    try:
        os.remove(name)
    except OSError:
        try:
            os.rmdir(name)
        except OSError:
            return False
    head, tail = os.path.split(name)
    if not tail:
        head, tail = os.path.split(head)
    while head and tail:
        try:
            if is_unsafe_to_delete(head, base):
                return False
            os.rmdir(head)
        except OSError:
            break
        head, tail = os.path.split(head)
    return True


def sort_normalize_string(s):
    return unicodedata.normalize("NFD", str(s).lower().strip())


def get_dependent_url(url_path: str, suffix: str, ext: str | None = None) -> str:
    path = PurePosixPath(url_path)
    if ext is not None:
        path = path.with_name(f"{path.stem}@{suffix}{ext}")
    else:
        path = path.with_stem(f"{path.stem}@{suffix}")
    return str(path)


@contextmanager
def atomic_open(filename, mode="r", encoding=None):
    if "r" not in mode:
        fd, tmp_filename = tempfile.mkstemp(
            dir=os.path.dirname(filename), prefix=".__atomic-write"
        )
        os.chmod(tmp_filename, 0o644)
        f = os.fdopen(fd, mode)
    else:
        f = open(filename, mode=mode, encoding=encoding)
        tmp_filename = None
    try:
        with f:
            yield f
    except Exception:
        if tmp_filename is not None:
            with suppress(OSError):
                os.remove(tmp_filename)
        raise

    if tmp_filename is not None:
        os.replace(tmp_filename, filename)


def portable_popen(cmd, *args, **kwargs):
    """A portable version of subprocess.Popen that automatically locates
    executables before invoking them.  This also looks for executables
    in the bundle bin.
    """
    if cmd[0] is None:
        raise RuntimeError("No executable specified")
    exe = locate_executable(cmd[0], kwargs.get("cwd"))
    if exe is None:
        raise RuntimeError('Could not locate executable "%s"' % cmd[0])

    if isinstance(exe, str) and sys.platform != "win32":
        exe = exe.encode(sys.getfilesystemencoding())
    cmd[0] = exe
    return subprocess.Popen(cmd, *args, **kwargs)


def is_valid_id(value):
    if value == "":
        return True
    return (
        "/" not in value
        and value.strip() == value
        and value.split() == [value]
        and not value.startswith(".")
    )


def secure_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.password is not None:
        _, _, host_port = parts.netloc.rpartition("@")
        parts = parts._replace(netloc=f"{parts.username}@{host_port}")
    return parts.geturl()


@overload
def bool_from_string(val: object, default: None = None) -> bool | None:
    ...


@overload
def bool_from_string(val: object, default: _T) -> bool | _T:
    ...


def bool_from_string(val: object, default: Any = None) -> Any:
    if val in (True, False, 1, 0):
        return bool(val)
    if isinstance(val, str):
        val = val.lower()
        if val in ("true", "yes", "1"):
            return True
        if val in ("false", "no", "0"):
            return False
    return default


def make_relative_url(source, target):
    """
    Returns the relative path (url) needed to navigate
    from `source` to `target`.
    """

    # WARNING: this logic makes some unwarranted assumptions about
    # what is a directory and what isn't. Ideally, this function
    # would be aware of the actual filesystem.
    s_is_dir = source.endswith("/")
    t_is_dir = target.endswith("/")

    source = PurePosixPath(posixpath.normpath(source))
    target = PurePosixPath(posixpath.normpath(target))

    if not s_is_dir:
        source = source.parent

    relpath = str(get_relative_path(source, target))
    if t_is_dir:
        relpath += "/"

    return relpath


def get_relative_path(source, target):
    """
    Returns the relative path needed to navigate from `source` to `target`.

    get_relative_path(source: PurePosixPath,
                      target: PurePosixPath) -> PurePosixPath
    """

    if not source.is_absolute() and target.is_absolute():
        raise ValueError("Cannot navigate from a relative path to an absolute one")

    if source.is_absolute() and not target.is_absolute():
        # nothing to do
        return target

    if source.is_absolute() and target.is_absolute():
        # convert them to relative paths to simplify the logic
        source = source.relative_to("/")
        target = target.relative_to("/")

    # is the source an ancestor of the target?
    try:
        return target.relative_to(source)
    except ValueError:
        pass

    # even if it isn't, one of the source's ancestors might be
    # (and if not, the root will be the common ancestor)
    distance = PurePosixPath(".")
    for ancestor in source.parents:
        distance /= ".."

        try:
            relpath = target.relative_to(ancestor)
        except ValueError:
            continue
        else:
            # prepend the distance to the common ancestor
            return distance / relpath
    # We should never get here.  (The last ancestor in source.parents will
    # be '.' — target.relative_to('.') will always succeed.)
    raise AssertionError("This should not happen")


def deg_to_dms(deg: float) -> tuple[int, int, float]:
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return (d, m, sd)


def format_lat_long(
    lat: float | None = None, long: float | None = None, secs: bool = True
) -> str:
    def _format(value: float, sign: Sequence[str]) -> str:
        d, m, sd = deg_to_dms(value)
        return "%d° %d′ %s%s" % (
            abs(d),
            abs(m),
            secs and ("%d″ " % abs(sd)) or "",
            sign[d < 0],
        )

    rv = []
    if lat is not None:
        rv.append(_format(lat, "NS"))
    if long is not None:
        rv.append(_format(long, "EW"))
    return ", ".join(rv)


def get_cache_dir() -> str:
    if is_windows:
        folder = os.environ.get("LOCALAPPDATA")
        if folder is None:
            folder = os.environ.get("APPDATA")
            if folder is None:
                folder = os.path.expanduser("~")
        return os.path.join(folder, "Lektor", "Cache")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~/Library/Caches/Lektor"))
    return os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "lektor"
    )


class URLBuilder:
    def __init__(self):
        self.items = []

    def append(self, item):
        if item is None:
            return
        item = str(item).strip("/")
        if item:
            self.items.append(item)

    def get_url(self, trailing_slash=None):
        url = "/" + "/".join(self.items)
        if trailing_slash is not None and not trailing_slash:
            return url
        if url == "/":
            return url
        if trailing_slash is None:
            _, last = url.split("/", 1)
            if "." in last:
                return url
        return url + "/"


def build_url(iterable, trailing_slash=None):
    # NB: While this function is not used by Lektor itself, it is used
    # by a number of plugins including: lektor-atom,
    # lektor-gemini-capsule, lektor-index-pages, and lektor-tags.
    builder = URLBuilder()
    for item in iterable:
        builder.append(item)
    return builder.get_url(trailing_slash=trailing_slash)


def comma_delimited(s: str) -> Iterator[str]:
    """Split a comma-delimited string."""
    for part in s.split(","):
        stripped = part.strip()
        if stripped:
            yield stripped


def process_extra_flags(flags: dict[str, str] | Iterable[str] | None) -> dict[str, str]:
    if isinstance(flags, dict):
        return flags
    rv: dict[str, str] = {}
    for flag in flags or ():
        key, sep, value = flag.partition(":")
        rv[key] = value if sep else key
    return rv


def unique_everseen(seq: Iterable[_H]) -> Iterable[_H]:
    """Filter out duplicates from iterable."""
    # This is a less general version of more_itertools.unique_everseen.
    # Should we need more general functionality, consider using that instead.
    seen = set()
    for val in seq:
        if val not in seen:
            seen.add(val)
            yield val


class RecursionCheck(threading.local):
    """A context manager that retains a count of how many times it's been entered.

    Example:

        >>> recursion_check = RecursionCheck()

        >>> with recursion_check:
        ...     assert recursion_check.level == 1
        ...     with recursion_check as recursion_level:
        ...         assert recursion_check.level == 2
        ...         print("depth", recursion_level)
        ...     assert recursion_check.level == 1
        ... assert recursion_check.level == 0
        depth 2
    """

    level = 0

    def __enter__(self) -> int:
        self.level += 1
        return self.level

    def __exit__(self, _t: Unused, _v: Unused, _tb: Unused) -> None:
        self.level -= 1


class DeprecatedWarning(DeprecationWarning):
    """Warning category issued by our ``deprecated`` decorator."""

    def __init__(
        self,
        name: str,
        reason: str | None = None,
        version: str | None = None,
    ):
        self.name = name
        self.reason = reason
        self.version = version

    def __str__(self) -> str:
        message = f"{self.name!r} is deprecated"
        if self.reason:
            message += f" ({self.reason})"
        if self.version:
            message += f" since version {self.version}"
        return message


@dataclass
class _Deprecate:
    """A decorator to mark callables as deprecated."""

    name: str | None = None
    reason: str | None = None
    version: str | None = None
    stacklevel: int = 1

    _recursion_check: ClassVar = RecursionCheck()

    def __call__(self, wrapped: _F) -> _F:
        if not callable(wrapped):
            raise TypeError("do not know how to deprecate {wrapped!r}")

        name = self.name or wrapped.__name__
        message = DeprecatedWarning(name, self.reason, self.version)

        @wraps(wrapped)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self._recursion_check as recursion_level:
                if recursion_level == 1:
                    warnings.warn(message, stacklevel=self.stacklevel + 1)
                return wrapped(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


@overload
def deprecated(
    __wrapped: Callable[..., Any],
    *,
    name: str | None = ...,
    reason: str | None = ...,
    version: str | None = ...,
    stacklevel: int = ...,
) -> Callable[..., Any]:
    ...


@overload
def deprecated(
    __reason: str,
    *,
    name: str | None = ...,
    version: str | None = ...,
    stacklevel: int = ...,
) -> _Deprecate:
    ...


@overload
def deprecated(
    *,
    name: str | None = ...,
    reason: str | None = ...,
    version: str | None = ...,
    stacklevel: int = ...,
) -> _Deprecate:
    ...


def deprecated(*args: Any, **kwargs: Any) -> _F | _Deprecate:
    """A decorator to mark callables or descriptors as deprecated.

    This can be used to decorate functions, methods, classes, and descriptors.
    In particular, this decorator can be applied to instances of ``property``,
    ``functools.cached_property`` and ``werkzeug.utils.cached_property``.

    When the decorated object is called (or — in the case of a descriptor — accessed), a
    ``DeprecationWarning`` is issued.

    The warning message will include the name of the decorated object, and may include
    further information if provided from the ``reason`` and ``version`` arguments.

    The ``name`` argument may be used to specify an alternative name to use when
    generating the warning message. By default, the ``__name__`` attribute of the
    decorated object is used.

    The ``stacklevel`` argument controls which call in the call stack the warning
    is attributed to. The default value, ``stacklevel=1`` means the warning is
    reported for the immediate caller of the decorated object.  Higher values
    attribute the warning callers further back in the stack.

    """
    if len(args) > 1:
        raise TypeError("deprecated accepts a maximum of one positional parameter")

    wrapped: _F | None = None
    if args:
        if isinstance(args[0], str):
            kwargs.setdefault("reason", args[0])
        else:
            wrapped = args[0]

    deprecate = _Deprecate(**kwargs)
    if wrapped is not None:
        return deprecate(wrapped)
    return deprecate
