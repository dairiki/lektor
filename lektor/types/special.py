from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Undefined

from lektor.types.primitives import SingleInputType
from lektor.utils import slugify
from lektor.utils import Url

if TYPE_CHECKING:
    from lektor.types.base import RawValue


class SortKeyType(SingleInputType):
    widget = "integer"

    def value_from_raw(self, raw: RawValue) -> int | Undefined:
        if raw.value is None:
            return raw.missing_value("Missing sort key")
        try:
            return int(raw.value.strip())
        except ValueError:
            return raw.bad_value("Bad sort key value")


class SlugType(SingleInputType):
    widget = "slug"

    def value_from_raw(self, raw: RawValue) -> str | Undefined:
        if raw.value is None:
            return raw.missing_value("Missing slug")
        return slugify(raw.value)


class UrlType(SingleInputType):
    widget = "url"

    def value_from_raw(self, raw: RawValue) -> Url | Undefined:
        if raw.value is None:
            return raw.missing_value("Missing URL")
        return Url.from_string(raw.value)
