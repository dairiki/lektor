from __future__ import annotations

import traceback
from itertools import zip_longest
from typing import Iterable
from typing import Iterator
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from jinja2 import Undefined

from lektor.constants import PRIMARY_ALT
from lektor.environment.expressions import Expression
from lektor.environment.expressions import FormatExpression
from lektor.i18n import get_i18n_block
from lektor.i18n import I18nBlock
from lektor.types.base import Type

if TYPE_CHECKING:
    from lektor.db import Pad
    from lektor.db import Record
    from lektor.environment import Environment
    from lektor.types.base import RawValue


def _reflow_and_split_labels(labels_i18n: I18nBlock) -> list[I18nBlock]:
    """Split a "packed" i18n block where the values are comma-separated choice labels.

    Returns a list of i18n blocks, each block specifying the label for one choice.
    """

    by_lang = [
        [(lang, label) for label in packed_labels.split(",")]
        for lang, packed_labels in labels_i18n.items()
    ]
    return [
        I18nBlock(lang_label for lang_label in by_index if lang_label is not None)
        for by_index in zip_longest(*by_lang)
    ]


_Choice = Tuple[Union[str, int], I18nBlock]


def _parse_choices(options: dict[str, str]) -> list[_Choice] | None:
    packed_choices = options.get("choices")
    if not packed_choices:
        return None

    choices: list[str | int] = []
    implied_labels: list[str] = []
    for item in packed_choices.split(","):
        if "=" in item:
            choice, value = item.split("=", 1)
            choice = choice.strip()
            choices.append(int(choice) if choice.isdigit() else choice)
            implied_labels.append(value.strip())
        else:
            choices.append(item.strip())
            implied_labels.append(item.strip())

    user_labels = get_i18n_block(options, "choice_labels")
    if user_labels:
        return list(zip(choices, _reflow_and_split_labels(user_labels)))
    return [
        (key, I18nBlock(en=label)) for key, label in zip(choices, implied_labels)
    ]


class ChoiceSource:
    source: Expression | None
    choices: list[_Choice] | None
    item_key: FormatExpression
    item_label: FormatExpression | None

    def __init__(self, env: Environment, options: dict[str, str]):
        source = options.get("source")
        if source is not None:
            self.source = Expression(env, source)
            self.choices = None
            item_key = options.get("item_key") or "{{ this._id }}"
            item_label = options.get("item_label")
        else:
            self.source = None
            self.choices = _parse_choices(options)
            item_key = options.get("item_key") or "{{ this.0 }}"
            item_label = options.get("item_label")
        self.item_key = FormatExpression(env, item_key)
        if item_label is not None:
            self.item_label = FormatExpression(env, item_label)
        else:
            self.item_label = None

    @property
    def has_choices(self) -> bool:
        return self.source is not None or self.choices is not None

    def iter_choices(
        self, pad: Pad, record: Record | None = None, alt: str = PRIMARY_ALT
    ) -> Iterator[tuple[str, I18nBlock]]:
        values = {}
        if record is not None:
            values["record"] = record

        iterable: Iterable[_Choice | Record]
        if self.choices is not None:
            iterable = self.choices
        else:
            assert self.source is not None
            try:
                iterable = self.source.evaluate(pad, alt=alt, values=values)
            except Exception:
                traceback.print_exc()
                iterable = ()

        for item in iterable or ():
            key = self.item_key.evaluate(pad, this=item, alt=alt, values=values)

            # If there is a label expression, use it.  Since in that case
            # we only have one language to fill in, we fill it in for the
            # default language
            if self.item_label is not None:
                label_i18n = I18nBlock(
                    en=self.item_label.evaluate(pad, this=item, alt=alt, values=values)
                )

            # Otherwise we create a proper internationalized key out of
            # our target label
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                label_i18n = item[1]
            elif hasattr(item, "get_record_label_i18n"):
                label_i18n = item.get_record_label_i18n()
            else:
                try:
                    label = item["_id"]
                except Exception:
                    label = repr(item)
                label_i18n = I18nBlock(en=label)

            yield key, label_i18n


class MultiType(Type):
    def __init__(self, env: Environment, options: dict[str, str]):
        super().__init__(env, options)
        self.source = ChoiceSource(env, options)

    def get_labels(
        self, pad: Pad, record: Record | None = None, alt: str = PRIMARY_ALT
    ) -> dict[str, I18nBlock]:
        return dict(self.source.iter_choices(pad, record, alt))

    def to_json(
        self, pad: Pad, record: Record | None = None, alt: str = PRIMARY_ALT
    ) -> dict[str, str | list[tuple[str, I18nBlock]]]:
        rv = super().to_json(pad, record, alt)
        if self.source.has_choices:
            rv["choices"] = list(self.source.iter_choices(pad, record, alt))
        return rv


class SelectType(MultiType):
    widget = "select"

    def value_from_raw(self, raw: RawValue) -> str | Undefined:
        if raw.value is None:
            return raw.missing_value("Missing select value")
        return raw.value


class CheckboxesType(MultiType):
    widget = "checkboxes"

    def value_from_raw(self, raw: RawValue) -> list[str]:
        rv = [x.strip() for x in (raw.value or "").split(",")]
        if rv == [""]:
            rv = []
        return rv
