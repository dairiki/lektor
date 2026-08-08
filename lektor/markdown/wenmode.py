"""MarkdownController implementation for mistune 2.x"""

from __future__ import annotations

import html
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Final
from typing import TYPE_CHECKING

import wenmode.ast
import wenmode.nodes
import wenmode.rules
from wenmode import HTMLRenderer
from wenmode import Wenmode
from wenmode.presets import commonmark
from wenmode.presets import create_preset
from wenmode.rules import Rule
from wenmode.rules.transforms import RootTransform

from lektor.markdown.controller import MarkdownController
from lektor.markdown.controller import RendererHelper


if TYPE_CHECKING:
    from _typeshed import Incomplete
    from types import ModuleType

    from wenmode.nodes import Root
    from wenmode.parser import Parser
    from wenmode.plugins.types import PluginModule
    from wenmode.renderers import DirectiveHtmlRenderer


def escape(text: str) -> str:
    # This is only here to provide the implementation for the
    # deprecated lektor.markdown.escape method.
    #
    # (We don't use it below and it can be deleted once access
    # to lektor.markdown.escape is removed.)
    return html.escape(text, quote=True)


class TransformLektorURLs(RootTransform):
    name = "transform-lektor-urls"

    __helper: Final = RendererHelper()

    def transform(self, parser: Parser, root: Root, state: Incomplete) -> None:
        for node in wenmode.ast.walk(root):
            if isinstance(node, (wenmode.nodes.Link, wenmode.nodes.Image)):
                node.url = self.__helper.resolve_url(node.url)


class ResolveLektorURLs(Rule):
    name = "resolve-lektor-urls"
    root_transforms = [TransformLektorURLs()]


DEFAULT_RULES: Final = tuple(
    create_preset(
        commonmark,
        prepend=[
            wenmode.rules.Table,  # FIXME: Table(require_body_pipe=False)?
        ],
        append=[
            wenmode.rules.Footnote,
            wenmode.rules.Strikethrough,
            wenmode.rules.ExtendedAutolink,
            ResolveLektorURLs,
        ],
    )
)


@dataclass
class MarkdownConfig:
    rules: Iterable[type[Rule] | Rule] = DEFAULT_RULES
    directives: Iterable[DirectiveHtmlRenderer] = field(default_factory=list)
    plugins: Iterable[PluginModule | ModuleType] = field(default_factory=list)


class MarkdownControllerWenmode(MarkdownController):
    def make_parser(self) -> Callable[[str | Iterable[str]], str]:
        env = self.env
        cfg = MarkdownConfig()
        # FIXME: call different hooks here for wenmode?
        env.plugin_controller.emit("markdown-config", config=cfg)
        renderer = HTMLRenderer(escape=False, sanitize_urls=False, sanitize_attrs=False)
        env.plugin_controller.emit(
            "markdown-lexer-config", config=cfg, renderer=renderer
        )
        wenmode = Wenmode(
            renderer=renderer,
            rules=cfg.rules,
            directives=cfg.directives,
            plugins=cfg.plugins,
        )
        return wenmode.render
