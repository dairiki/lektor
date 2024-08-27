from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lektor.db import Pad
    from lektor.environment import Environment
    from lektor.environment import TemplateValuesType


class Expression:
    def __init__(self, env: Environment, expr: str):
        self.env = env
        self.tmpl = env.jinja_env.from_string("{{ __result__(%s) }}" % expr)

    def evaluate(
        self,
        pad: Pad | None = None,
        this: object | None = None,
        values: TemplateValuesType | None = None,
        alt: str | None = None,
    ) -> Any:
        result = []

        def result_func(value: Any) -> str:
            result.append(value)
            return ""

        values = self.env.make_default_tmpl_values(pad, this, values, alt)
        values["__result__"] = result_func
        self.tmpl.render(values)
        return result[0]


class FormatExpression:
    def __init__(self, env: Environment, expr: str):
        self.env = env
        self.tmpl = env.jinja_env.from_string(expr)

    def evaluate(
        self,
        pad: Pad | None = None,
        this: object | None = None,
        values: TemplateValuesType | None = None,
        alt: str | None = None,
    ) -> str:
        values = self.env.make_default_tmpl_values(pad, this, values, alt)
        return self.tmpl.render(values)
