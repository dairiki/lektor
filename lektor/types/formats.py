from __future__ import annotations

import sys
from typing import Any
from typing import Collection
from typing import TYPE_CHECKING
from warnings import warn

from lektor.markdown import Markdown
from lektor.types.base import Type

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from _typeshed import Unused

    from lektor.environment import Environment
    from lektor.sourceobj import SourceObject
    from lektor.types.base import RawValue


class MarkdownDescriptor:
    def __init__(self, source: str, options: dict[str, str]):
        self.source = source
        self.options = options

    def __get__(self, obj: SourceObject, type_: Unused = None) -> Self | Markdown:
        if obj is None:
            return self
        return Markdown(self.source, record=obj, field_options=self.options)


class MarkdownType(Type):
    widget = "multiline-text"

    def __init__(self, env: Environment, options: dict[str, str]):
        super().__init__(env, options)
        _check_option(options, "resolve_links", ("always", "never", "when-possible"))

    def value_from_raw(self, raw: RawValue) -> MarkdownDescriptor:
        return MarkdownDescriptor(raw.value or "", self.options)


def _check_option(
    options: dict[str, Any], name: str, choices: Collection[object]
) -> None:
    value = options.get(name)
    if value is not None and value not in choices:
        warn(
            f"Unrecognized value {value!r} for the {name!r} markdown field option. "
            f"Valid values are: {', '.join(repr(_) for _ in choices)}."
        )
