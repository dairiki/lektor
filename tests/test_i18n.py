from __future__ import annotations

import importlib

import pytest

import lektor.i18n
from lektor.i18n import get_i18n_block


def test_loading_i18n_triggers_no_warnings(recwarn):
    importlib.reload(lektor.i18n)
    for warning in recwarn.list:
        print(warning)  # debugging: display warnings on stdout
    assert len(recwarn) == 0


I18N_BLOCK_TEST_DATA = {
    "name": "Mount",
    "name[de]": "Spitze",
    "name[bad": "foo",
    "notname": "bar",
}


@pytest.mark.parametrize(
    "key, expect",
    [
        ("name", {"en": "Mount", "de": "Spitze"}),
    ],
)
def test_get_18n_block(key, expect):
    assert get_i18n_block(I18N_BLOCK_TEST_DATA, key) == expect


def test_get_18n_block_pop():
    data = I18N_BLOCK_TEST_DATA.copy()
    i18n_block = get_i18n_block(data, "name", pop=True)
    assert i18n_block == {"en": "Mount", "de": "Spitze"}
    assert set(data) == set(I18N_BLOCK_TEST_DATA).difference(["name", "name[de]"])
