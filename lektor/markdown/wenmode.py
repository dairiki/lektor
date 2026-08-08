"""MarkdownController implementation for mistune 2.x"""

from __future__ import annotations

import dataclasses
import html
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Final
from typing import TYPE_CHECKING

import wenmode
import wenmode.ast
import wenmode.nodes
import wenmode.renderers.html
import wenmode.rules
from wenmode import HTMLRenderer
from wenmode import Wenmode
from wenmode.nodes import Image
from wenmode.nodes import Link
from wenmode.presets import commonmark
from wenmode.presets import create_preset

from lektor.markdown.controller import MarkdownController
from lektor.markdown.controller import RendererHelper


if TYPE_CHECKING:
    from types import ModuleType

    from wenmode.plugins.types import PluginModule
    from wenmode.renderers import DirectiveHtmlRenderer
    from wenmode.renderers.html import HTMLRenderContext
    from wenmode.rules import Rule


def escape(text: str) -> str:
    # This is only here to provide the implementation for the
    # deprecated lektor.markdown.escape method.
    #
    # (We don't use it below and it can be deleted once access
    # to lektor.markdown.escape is removed.)
    return html.escape(text, quote=True)


DEFAULT_RULES: Final = tuple(
    # Here, for b/c we attempt to duplicate the behavior of mistune 0.*
    create_preset(
        commonmark,
        prepend=[
            wenmode.rules.Table,  # FIXME: Table(require_body_pipe=False)?
        ],
        append=[
            wenmode.rules.Footnote,
            wenmode.rules.Strikethrough,
            wenmode.rules.ExtendedAutolink,
        ],
    )
)


_RENDERER_HELPER: Final = RendererHelper()


def lektor_render_link(
    renderer: HTMLRenderer, node: Link, context: HTMLRenderContext
) -> str:
    resolved_url = _RENDERER_HELPER.resolve_url(node.url)
    return wenmode.renderers.html.render_link(
        renderer, dataclasses.replace(node, url=resolved_url), context
    )


def lektor_render_image(
    renderer: HTMLRenderer, node: Image, context: HTMLRenderContext
) -> str:
    resolved_url = _RENDERER_HELPER.resolve_url(node.url)
    return wenmode.renderers.html.render_image(
        renderer, dataclasses.replace(node, url=resolved_url), context
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
        renderer.register_handler(Link.type, lektor_render_link)
        renderer.register_handler(Image.type, lektor_render_image)

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
