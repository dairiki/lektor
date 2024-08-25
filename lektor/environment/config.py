from __future__ import annotations

import copy
import dataclasses
import os
from collections import OrderedDict
from functools import cached_property
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import overload
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypedDict
from urllib.parse import urlsplit

from inifile import IniFile

from lektor.constants import PRIMARY_ALT
from lektor.i18n import get_i18n_block
from lektor.i18n import I18nBlock
from lektor.utils import bool_from_string
from lektor.utils import secure_url

if TYPE_CHECKING:
    from _typeshed import StrPath


class ProjectConfig(TypedDict):
    name: str | None
    locale: str
    url: str | None
    url_style: Literal["relative", "absolute", "external"]


class AltConfig(TypedDict):
    name: I18nBlock
    url_prefix: str | None
    url_suffix: str | None
    primary: bool
    locale: str


class ConfigValues(TypedDict):
    EPHEMERAL_RECORD_CACHE_SIZE: int
    ATTACHMENT_TYPES: dict[str, str]
    PROJECT: ProjectConfig
    THEME_SETTINGS: dict[str, str]
    PACKAGES: dict[str, str]
    ALTERNATIVES: dict[str, AltConfig]
    PRIMARY_ALTERNATIVE: str | None
    SERVERS: dict[str, dict[str, str]]


DEFAULT_CONFIG = {
    "EPHEMERAL_RECORD_CACHE_SIZE": 500,
    "ATTACHMENT_TYPES": {
        # Only enable image formats here that we can handle in imagetools.
        # Right now this is limited to jpg, png and gif.
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".gif": "image",
        ".svg": "image",
        ".avi": "video",
        ".mpg": "video",
        ".mpeg": "video",
        ".wmv": "video",
        ".ogv": "video",
        ".mp4": "video",
        ".mp3": "audio",
        ".wav": "audio",
        ".ogg": "audio",
        ".pdf": "document",
        ".doc": "document",
        ".docx": "document",
        ".htm": "document",
        ".html": "document",
        ".txt": "text",
        ".log": "text",
    },
    "PROJECT": {
        "name": None,
        "locale": "en_US",
        "url": None,
        "url_style": "relative",
    },
    "THEME_SETTINGS": {},
    "PACKAGES": {},
    "ALTERNATIVES": OrderedDict(),
    "PRIMARY_ALTERNATIVE": None,
    "SERVERS": {},
}


def update_config_from_ini(config: dict[str, Any], inifile: IniFile) -> None:
    for section_name in ("ATTACHMENT_TYPES", "PROJECT", "PACKAGES", "THEME_SETTINGS"):
        config[section_name].update(inifile.section_as_dict(section_name.lower()))

    for sect in inifile.sections():
        if sect.startswith("servers."):
            server_id = sect.split(".")[1]
            config["SERVERS"][server_id] = inifile.section_as_dict(sect)
        elif sect.startswith("alternatives."):
            alt = sect.split(".")[1]
            config["ALTERNATIVES"][alt] = {
                "name": get_i18n_block(inifile, "alternatives.%s.name" % alt),
                "url_prefix": inifile.get("alternatives.%s.url_prefix" % alt),
                "url_suffix": inifile.get("alternatives.%s.url_suffix" % alt),
                "primary": inifile.get_bool("alternatives.%s.primary" % alt),
                "locale": inifile.get("alternatives.%s.locale" % alt, "en_US"),
            }

    for alt, alt_data in config["ALTERNATIVES"].items():
        if alt_data["primary"]:
            config["PRIMARY_ALTERNATIVE"] = alt
            break
    else:
        if config["ALTERNATIVES"]:
            raise RuntimeError("Alternatives defined but no primary set.")


@dataclasses.dataclass
class ServerInfo:
    id: str
    name_i18n: I18nBlock
    target: str
    enabled: bool = True
    default: bool = False
    extra: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.name_i18n.get("en") or self.id

    @property
    def short_target(self) -> str:
        url = urlsplit(self.target)
        if url.scheme and url.netloc:
            return f"{url.netloc} via {url.scheme}"
        return self.target

    def to_json(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "name": self.name,
            "short_target": self.short_target,
        }


class Config:
    def __init__(self, filename: StrPath | None = None):
        self.filename = filename
        self.values = copy.deepcopy(DEFAULT_CONFIG)

        if filename is not None and os.path.isfile(filename):
            inifile = IniFile(filename)
            update_config_from_ini(self.values, inifile)

    @overload
    def __getitem__(self, name: Literal["PROJECT"]) -> ProjectConfig:
        ...

    @overload
    def __getitem__(self, name: Literal["ALTERNATIVES"]) -> Mapping[str, AltConfig]:
        ...

    @overload
    def __getitem__(self, name: Literal["SERVERS"]) -> Mapping[str, Mapping[str, str]]:
        ...

    @overload
    def __getitem__(self, name: Literal["PRIMARY_ALTERNATIVE"]) -> str | None:
        ...

    @overload
    def __getitem__(self, name: Literal["EPHEMERAL_RECORD_CACHE_SIZE"]) -> int:
        ...

    @overload
    def __getitem__(
        self, name: Literal["ATTACHMENT_TYPES", "THEME_SETTINGS", "PACKAGES"]
    ) -> Mapping[str, str]:
        ...

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    @property
    def site_locale(self) -> str:
        """The locale of this project."""
        return self["PROJECT"]["locale"]

    def get_servers(self, public: bool = False) -> dict[str, ServerInfo]:
        """Returns a list of servers."""
        return {
            server: server_info
            for server in self["SERVERS"]
            if (server_info := self.get_server(server, public=public)) is not None
        }

    def get_default_server(self, public: bool = False) -> ServerInfo | None:
        """Returns the default server."""
        server_infos = list(self.get_servers().values())
        for server_info in server_infos:
            if server_info.default:
                return server_info
        if len(server_infos) == 1:
            return server_infos[0]
        return None

    def get_server(self, name: str, public: bool = False) -> ServerInfo | None:
        """Looks up a server info by name."""
        data = self["SERVERS"].get(name)
        if data is None:
            return None
        info = dict(data)
        if (target := info.pop("target", None)) is None:
            return None
        if public:
            target = secure_url(target)
        return ServerInfo(
            id=name,
            name_i18n=get_i18n_block(info, "name", pop=True),
            target=target,
            enabled=bool_from_string(info.pop("enabled", None), True),
            default=bool_from_string(info.pop("default", None), False),
            extra=info,
        )

    def is_valid_alternative(self, alt: str) -> bool:
        """Checks if an alternative ID is known."""
        if alt == PRIMARY_ALT:
            return True
        return alt in self["ALTERNATIVES"]

    def list_alternatives(self) -> Sequence[str]:
        """Returns a sorted list of alternative IDs."""
        return sorted(self["ALTERNATIVES"])

    def iter_alternatives(self) -> Iterator[str]:
        """Iterates over all alternatives.  If the system is disabled this
        yields '_primary'.
        """
        found = False
        for alt in self["ALTERNATIVES"]:
            if alt != PRIMARY_ALT:
                yield alt
                found = True
        if not found:
            yield PRIMARY_ALT

    def get_alternative(self, alt: str) -> AltConfig | None:
        """Returns the config setting of the given alt."""
        if alt == PRIMARY_ALT:
            if self.primary_alternative is None:
                return None
            alt = self.primary_alternative
        return self["ALTERNATIVES"].get(alt)

    def get_alternative_url_prefixes(self) -> Sequence[tuple[str, str]]:
        """Returns a list of alternative url prefixes by length."""

        def sort_key(item: tuple[str, str]) -> int:
            url_prefix, _alt = item
            return len(url_prefix)

        return sorted(
            (
                (url_prefix.lstrip("/"), alt)
                for alt, alt_cfg in self["ALTERNATIVES"].items()
                if (url_prefix := alt_cfg["url_prefix"])
            ),
            key=sort_key,
            reverse=True,
        )

    def get_alternative_url_suffixes(self) -> Sequence[tuple[str, str]]:
        """Returns a list of alternative url suffixes by length."""

        def sort_key(item: tuple[str, str]) -> int:
            url_suffix, _alt = item
            return len(url_suffix)

        return sorted(
            (
                (url_suffix.rstrip("/"), alt)
                for alt, alt_cfg in self["ALTERNATIVES"].items()
                if (url_suffix := alt_cfg["url_suffix"])
            ),
            key=sort_key,
            reverse=True,
        )

    def get_alternative_url_span(self, alt: str = PRIMARY_ALT) -> tuple[str, str]:
        """Returns the URL span (prefix, suffix) for an alt."""
        if alt == PRIMARY_ALT:
            if self.primary_alternative is None:
                return "", ""
            alt = self.primary_alternative

        alt_cfg = self["ALTERNATIVES"].get(alt)
        if alt_cfg is not None:
            return alt_cfg["url_prefix"] or "", alt_cfg["url_suffix"] or ""
        return "", ""

    @cached_property
    def primary_alternative_is_rooted(self) -> bool:
        """`True` if the primary alternative is sitting at the root of
        the URL handler.
        """
        primary = self.primary_alternative
        if primary is None:
            return True

        alt_cfg = self["ALTERNATIVES"].get(primary)
        if alt_cfg is None:
            return True

        url_prefix = (alt_cfg["url_prefix"] or "").lstrip("/")
        url_suffix = (alt_cfg["url_suffix"] or "").rstrip("/")
        return url_prefix == "" and url_suffix == ""

    @property
    def primary_alternative(self) -> str | None:
        """The identifier that acts as primary alternative."""
        return self["PRIMARY_ALTERNATIVE"]

    @cached_property
    def base_url(self) -> str | None:
        """The external base URL."""
        url = self["PROJECT"].get("url")
        if url and urlsplit(url).scheme:
            return url.rstrip("/") + "/"
        return None

    @cached_property
    def base_path(self) -> str:
        """The base path of the URL."""
        url = self["PROJECT"].get("url")
        if url:
            return urlsplit(url).path.rstrip("/") + "/"
        return "/"

    @cached_property
    def url_style(self) -> Literal["relative", "absolute", "external"]:
        """The intended URL style."""
        style = self["PROJECT"].get("url_style")
        if style in ("relative", "absolute", "external"):
            return style
        return "relative"
