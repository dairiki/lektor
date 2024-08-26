from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from lektor.constants import PRIMARY_ALT
from lektor.i18n import get_i18n_block
from lektor.types.base import Type

if TYPE_CHECKING:
    from lektor.db import Pad
    from lektor.db import Record
    from lektor.types.base import RawValue


class FakeType(Type):
    def value_from_raw(self, raw: RawValue) -> None:
        return None

    def to_json(
        self, pad: Pad, record: Record | None = None, alt: str = PRIMARY_ALT
    ) -> dict[str, Any]:
        return {
            **super().to_json(pad, record, alt),
            "is_fake_type": True,
        }


class LineType(FakeType):
    widget = "f-line"


class SpacingType(FakeType):
    widget = "f-spacing"


class InfoType(FakeType):
    widget = "f-info"


class HeadingType(FakeType):
    widget = "f-heading"

    def to_json(
        self, pad: Pad, record: Record | None = None, alt: str = PRIMARY_ALT
    ) -> dict[str, Any]:
        return {
            **super().to_json(pad, record, alt),
            "heading_i18n": get_i18n_block(self.options, "heading"),
        }
