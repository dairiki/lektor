"""Test that all modules/packages in the lektor tree are importable in any order

Here we import each module by itself, one at a time, each in a new
python interpreter.

"""

import pkgutil
import sys
from subprocess import run

import pytest

import lektor
from lektor.markdown import controller_class


ignored = set()

# Do not check importability of unused markdown implementations
match controller_class.implementation:
    case "mistune0":
        ignored.add("lektor.markdown.mistune2")
        ignored.add("lektor.markdown.wenmode")
    case "mistune2":
        ignored.add("lektor.markdown.mistune0")
        ignored.add("lektor.markdown.wenmode")
    case "wenmode":
        ignored.add("lektor.markdown.mistune0")
        ignored.add("lektor.markdown.mistune2")


def iter_lektor_modules():
    for module in pkgutil.walk_packages(lektor.__path__, f"{lektor.__name__}."):
        if module.name not in ignored:
            yield module.name


@pytest.fixture(params=iter_lektor_modules())
def module(request):
    return request.param


@pytest.mark.slowtest
def test_import(module):
    python = sys.executable
    assert run([python, "-c", f"import {module}"], check=False).returncode == 0
