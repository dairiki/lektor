from __future__ import annotations

import datetime
import re
import sys
from html import unescape
from pathlib import Path
from unittest import mock

import pytest

import lektor.environment
from lektor.db import Pad
from lektor.environment import Environment


@pytest.fixture
def scratch_project_data(scratch_project_data):
    # Add a sub-page to the scratch project
    data = {"_model": "page", "title": "Subpage", "body": "Subpage body"}
    subpage_lr = scratch_project_data / "content/sub-page/contents.lr"
    subpage_lr.parent.mkdir()
    subpage_lr.write_text("".join(lektor.metaformat.serialize(data.items())))

    testbag_ini = scratch_project_data / "databags/testbag.ini"
    testbag_ini.parent.mkdir()
    testbag_ini.write_text("foo = bar")

    return scratch_project_data


@pytest.fixture
def compile_template(scratch_env):
    def compile_template(source, name="tmpl.html"):
        Path(scratch_env.root_path, "templates", name).write_text(
            source, encoding="utf-8"
        )
        return scratch_env.jinja_env.get_template(name)

    return compile_template


@pytest.fixture
def source_path():
    return "/"


@pytest.fixture
def dummy_ctx(dummy_ctx, source_path):
    if source_path is None:
        source = None
    else:
        source = dummy_ctx.pad.get(source_path)

    artifact = dummy_ctx.artifact_txn.artifact
    artifact.source_obj = source
    if source is not None:
        artifact.sources = tuple(source.iter_source_filenames())
    return dummy_ctx


def test_jinja2_feature_autoescape(compile_template):
    tmpl = compile_template("{{ value }}", "tmpl.html")
    rendered = tmpl.render(value="<tag>")
    assert unescape(rendered) == "<tag>"
    assert "<" not in rendered


def test_jinja2_feature_with(compile_template):
    tmpl = compile_template("{% with x = 'good' %}{{ x }}{% endwith %}")
    assert tmpl.render() == "good"


def test_jinja2_feature_do(compile_template):
    tmpl = compile_template(
        "{% set x = ['a'] %}{% do x.append('b') %}{{ x|join('-') }}"
    )
    assert tmpl.render() == "a-b"


@pytest.mark.parametrize("source_path", [None, "/"])
@pytest.mark.usefixtures("dummy_ctx")
def test_jinja2_markdown_filter(compile_template):
    tmpl = compile_template("{{ '**word**' | markdown }}")
    assert "<strong>word</strong>" in tmpl.render()


@pytest.mark.usefixtures("dummy_ctx")
def test_jinja2_markdown_filter_resolve_links(compile_template):
    tmpl = compile_template(
        "{{ '[subpage](sub-page)' | markdown(resolve_links='always') }}"
    )
    assert re.search(r"<a.*\bhref=(['\"])sub-page/\1.*>subpage</a>", tmpl.render())


@pytest.mark.parametrize(
    "source_path, resolve_links",
    [
        (None, "if-possible"),
        (None, "never"),
        ("/", "never"),
    ],
)
@pytest.mark.usefixtures("dummy_ctx")
def test_jinja2_markdown_filter_noresolve_links(compile_template, resolve_links):
    tmpl = compile_template(
        f"{{{{ '[subpage](sub-page)' | markdown(resolve_links={resolve_links!r}) }}}}"
    )
    assert re.search(r"<a.*\bhref=(['\"])sub-page\1.*>subpage</a>", tmpl.render())


@pytest.mark.parametrize("source_path", [None])
@pytest.mark.usefixtures("dummy_ctx")
def test_jinja2_markdown_filter_resolve_raises_if_no_source_obj(compile_template):
    tmpl = compile_template(
        "{{ '[subpage](sub-page)' | markdown(resolve_links='always') }}"
    )
    with pytest.raises(RuntimeError) as exc_info:
        tmpl.render()
    assert re.search(r"\bsource object\b.*\brequired\b", str(exc_info.value))


def test_no_reference_cycle_in_environment(project):
    env = project.make_env(load_plugins=False)
    # reference count should be two: one from our `env` variable, and
    # another from the argument to sys.getrefcount
    assert sys.getrefcount(env) == 2


@pytest.fixture
def render_string(env):
    def render_string(s, **kwargs):
        template = env.jinja_env.from_string(s)
        return template.render(**kwargs)

    return render_string


def test_dateformat_filter(render_string):
    tmpl = "{{ dt | dateformat('yyyy-MM-dd') }}"
    dt = datetime.date(2001, 2, 3)
    assert render_string(tmpl, dt=dt) == "2001-02-03"


@pytest.mark.usefixtures("dummy_ctx")
def test_datetimeformat_filter_not_inlined(pad):
    template = pad.env.jinja_env.from_string("{{ 1678749806 | datetimeformat }}")
    en_date = template.render()
    with mock.patch.object(lektor.environment, "get_locale", return_value="de_DE"):
        de_date = template.render()
    assert en_date != de_date


def test_datetimeformat_filter(render_string):
    tmpl = "{{ dt | datetimeformat('yyyy-MM-ddTHH:mm') }}"
    dt = datetime.datetime(2001, 2, 3, 4, 5, 6)
    assert render_string(tmpl, dt=dt) == "2001-02-03T04:05"


def test_timeformat_filter(render_string):
    tmpl = "{{ dt | datetimeformat('HH:mm') }}"
    dt = datetime.time(1, 2, 3)
    assert render_string(tmpl, dt=dt) == "01:02"


@pytest.fixture(params=["dateformat", "datetimeformat", "timeformat"])
def dates_filter(request: pytest.FixtureRequest) -> str:
    return request.param


def test_dates_format_filter_handles_undefined(
    env: Environment, dates_filter: str
) -> None:
    template = env.jinja_env.from_string("{{ undefined | %s }}" % dates_filter)
    assert template.render() == ""


def test_dates_format_filter_raises_type_error_on_bad_arg(
    env: Environment, dates_filter: str
) -> None:
    template = env.jinja_env.from_string("{{ obj | %s }}" % dates_filter)
    with pytest.raises(TypeError, match="unexpected exception"):
        template.render(obj=object())


def test_dates_format_filter_raises_type_error_on_bad_format(
    env: Environment, dates_filter: str
) -> None:
    template = env.jinja_env.from_string("{{ now | %s(42) }}" % dates_filter)
    with pytest.raises(TypeError, match="should be a str"):
        template.render(now=datetime.datetime.now())


@pytest.mark.parametrize("arg", ["locale", "tzinfo"])
def test_dates_format_filter_raises_type_error_on_bad_kwarg(
    env: Environment, dates_filter: str, arg: str
) -> None:
    template = env.jinja_env.from_string(f"{{{{ now | {dates_filter}({arg}=42) }}}}")
    with pytest.raises(TypeError):
        template.render(now=datetime.datetime.now())


def test_bag_gets_site_from_jinja_context(
    scratch_env: Environment, scratch_pad: Pad
) -> None:
    template = scratch_env.jinja_env.from_string("{{ bag('testbag.foo') }}")
    assert template.render(site=scratch_pad) == "bar"


@pytest.mark.usefixtures("dummy_ctx")
def test_bag_gets_site_from_lektor_context(scratch_env: Environment) -> None:
    template = scratch_env.jinja_env.from_string("{{ bag('testbag.foo') }}")
    assert template.render() == "bar"
