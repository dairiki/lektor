from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from jinja2 import Undefined

from lektor.constants import PRIMARY_ALT

if TYPE_CHECKING:
    from _typeshed import Unused

    from lektor.datamodel import Field
    from lektor.db import Pad
    from lektor.db import Record
    from lektor.environment import Environment


class BadValue(Undefined):
    __slots__ = ()


# XXX: appears unused
def get_undefined_info(undefined: object) -> str:
    if isinstance(undefined, Undefined):
        try:
            undefined._fail_with_undefined_error()
        except Exception as e:
            return str(e)
    return "defined value"


class RawValue:
    __slots__ = ("name", "value", "field", "pad")

    def __init__(
        self,
        name: str,
        value: str | None = None,
        field: Field | None = None,
        pad: Pad | None = None,
    ):
        self.name = name
        self.value = value
        self.field = field
        self.pad = pad

    def _get_hint(self, prefix: str, reason: str) -> str:
        if self.field is not None:
            return f"{prefix} in field '{self.field.name}': {reason}"
        return f"{prefix}: {reason}"

    def bad_value(self, reason: str) -> BadValue:
        return BadValue(hint=self._get_hint("Bad value", reason), obj=self.value)

    def missing_value(self, reason: str) -> Undefined:
        return Undefined(hint=self._get_hint("Missing value", reason), obj=self.value)


class _NameDescriptor:
    def __get__(self, obj: Unused, type_: type[Type]) -> str:
        rv = type_.__name__
        if rv.endswith("Type"):
            rv = rv[:-4]
        return rv.lower()


class Type:
    widget = "multiline-text"

    def __init__(self, env: Environment, options: dict[str, str]):
        self.env = env
        self.options = options

    @property
    def size(self) -> str:
        size = self.options.get("size") or "normal"
        if size not in ("normal", "small", "large"):
            size = "normal"
        return size

    @property
    def width(self) -> str:
        return self.options.get("width") or "1/1"

    name = _NameDescriptor()

    def to_json(
        self, pad: Pad, record: Record | None = None, alt: str = PRIMARY_ALT
    ) -> dict[str, Any]:
        return {
            "name": self.name,
            "widget": self.widget,
            "size": self.size,
            "width": self.width,
        }

    def value_from_raw(self, raw: RawValue) -> Any | Undefined:
        # pylint: disable=no-self-use
        return raw  # XXX: should return raw.value?

    def value_from_raw_with_default(self, raw: RawValue) -> Any | Undefined:
        value = self.value_from_raw(raw)
        if (
            isinstance(value, Undefined)
            and raw.field is not None
            and raw.field.default is not None
        ):
            return self.value_from_raw(
                RawValue(raw.name, raw.field.default, field=raw.field, pad=raw.pad)
            )
        return value

    def __repr__(self) -> str:
        return "%s()" % self.__class__.__name__
