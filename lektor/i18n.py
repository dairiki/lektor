from __future__ import annotations

import json
import os
import re
from typing import Dict
from typing import Literal
from typing import Mapping
from typing import MutableMapping
from typing import overload


class I18nBlock(Dict[str, str]):
    """An dict mapping lang to internationalized translations for a particular label.

    Note that lang "en" is treated specially in that it is the fallback
    if no translation for the desired lang is found.

    I18nBlocks *should* always have a value for "en".

    """


translations_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "translations"
)
KNOWN_LANGUAGES = list(
    x[:-5] for x in os.listdir(translations_path) if x.endswith(".json")
)


translations = {}
for _lang in KNOWN_LANGUAGES:
    with open(os.path.join(translations_path, _lang + ".json"), "rb") as f:
        translations[_lang] = json.load(f)


def get_translations(language):
    """Looks up the translations for a given language."""
    return translations.get(language)


def is_valid_language(lang):
    """Verifies a language is known and valid."""
    return lang in KNOWN_LANGUAGES


def get_default_lang():
    """Returns the default language the system should use."""
    for key in "LANGUAGE", "LC_ALL", "LC_CTYPE", "LANG":
        value = os.environ.get(key)
        if not value:
            continue
        lang = value.split("_")[0].lower()
        if is_valid_language(lang):
            return lang
    return "en"


def load_i18n_block(key):
    """Looks up an entire i18n block from a known translation."""
    rv = {}
    for lang in KNOWN_LANGUAGES:
        val = translations.get(lang, {}).get(key)
        if val is not None:
            rv[lang] = val
    return rv


@overload
def get_i18n_block(
    inifile_or_dict: MutableMapping[str, str], key: str, pop: Literal[True]
) -> I18nBlock:
    ...


@overload
def get_i18n_block(
    inifile_or_dict: Mapping[str, str], key: str, pop: Literal[False] = False
) -> I18nBlock:
    ...


def get_i18n_block(
    inifile_or_dict: Mapping[str, str], key: str, pop: bool = False
) -> I18nBlock:
    """Extracts an i18n block from an ini file or dictionary for a given
    key. If "pop", delete keys from "inifile_or_dict".
    """
    data = inifile_or_dict
    if pop and not isinstance(data, MutableMapping):
        raise TypeError("A mutable mapping if pop is set")

    # English is the internal default language with preferred
    # treatment.
    key_re = re.escape(key) + r"(?x: \[ (?P<lang> \S+) \] )?"
    match_key = re.compile(key_re).fullmatch

    key_map = [(key_, m.group(1) or "en") for key_ in data if (m := match_key(key_))]
    if pop:
        assert isinstance(data, MutableMapping)
        return I18nBlock((lang, data.pop(key_)) for key_, lang in key_map)

    return I18nBlock((lang, data[key_]) for key_, lang in key_map)


def generate_i18n_kvs(**opts):
    """Generates key-value pairs based on the kwargs passed into this function.
    For every key ending in "_i18n", its corresponding value will be translated
    and returned once for every language that has a known translation.
    """
    for key, value in opts.items():
        if key.endswith("_i18n"):
            base_key = key[:-5]
            for lang, trans in load_i18n_block(value).items():
                lang_key = f"{base_key}[{lang}]"
                yield lang_key, trans
        else:
            yield key, value
