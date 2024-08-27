from __future__ import annotations

import sqlite3
from collections.abc import Sized
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import TYPE_CHECKING

from lektor.constants import PRIMARY_ALT

if TYPE_CHECKING:
    from _typeshed import Unused

    from lektor.builder import Builder


def _iter_parents(path: str) -> Iterator[str]:
    path = path.strip("/")
    if path:
        pieces = path.split("/")
        for x in range(len(pieces)):
            yield "/" + "/".join(pieces[:x])


_Info = Dict[str, str]
_Breadcrumb = Dict[str, str]


def _find_info(infos: Iterable[_Info], alt: str, lang: str) -> _Info | None:
    for info in infos:
        if info["alt"] == alt and info["lang"] == lang:
            return info
    return None


def _id_from_path(path: str) -> str:
    try:
        return path.strip("/").split("/")[-1]
    except IndexError:
        return ""


def _mapping_from_cursor(cur: sqlite3.Cursor) -> dict[str, list[_Info]]:
    rv: dict[str, list[dict[str, str]]] = {}
    for path, alt, lang, type, title in cur.fetchall():
        rv.setdefault(path, []).append(
            {
                "id": _id_from_path(path),
                "path": path,
                "alt": alt,
                "type": type,
                "lang": lang,
                "title": title,
            }
        )
    return rv


def _find_best_info(infos: Collection[_Info], alt: str, lang: str) -> _Info | None:
    for _alt, _lang in [
        (alt, lang),
        (PRIMARY_ALT, lang),
        (alt, "en"),
        (PRIMARY_ALT, "en"),
    ]:
        rv = _find_info(infos, _alt, _lang)
        if rv is not None:
            return rv
    return None


def _build_parent_path(
    path: str, mapping: Mapping[str, Collection[_Info]], alt: str, lang: str
) -> list[_Breadcrumb]:
    rv = []
    for parent in _iter_parents(path):
        info = _find_best_info(mapping.get(parent) or [], alt, lang)
        id = _id_from_path(parent)
        if info is None or (title := info.get("title")) is None:
            title = id or "(Index)"
        rv.append({"id": id, "path": parent, "title": title})
    return rv


def _process_search_results(
    builder: Unused, cur: sqlite3.Cursor, alt: str, lang: str, limit: int
) -> list[dict[str, str | list[_Breadcrumb]]]:
    matches = []
    mapping = _mapping_from_cursor(cur)

    files_needed = set()

    for path, infos in mapping.items():
        info = _find_best_info(infos, alt, lang)
        if info is None:
            continue

        for parent in _iter_parents(path):
            if parent not in mapping:
                files_needed.add(parent)

        matches.append(info)
        if len(matches) == limit:
            break

    if files_needed:
        cur.execute(
            """
            select path, alt, lang, type, title
              from source_info
             where path in (%s)
        """
            % ", ".join(["?"] * len(files_needed)),
            list(files_needed),
        )
        mapping.update(_mapping_from_cursor(cur))

    return [
        {
            **info,
            "parents": _build_parent_path(info["path"], mapping, alt, lang),
        }
        for info in matches
    ]


def _placeholders(values: Sized) -> str:
    return ", ".join(["?"] * len(values))


def find_files(
    builder: Builder,
    query: str,
    alt: str = PRIMARY_ALT,
    lang: str | None = None,
    limit: int = 50,
    types: Iterable[str] | None = None,
) -> list[dict[str, str | list[_Breadcrumb]]]:
    if types is None:
        types_ = {"page"}
    else:
        types_ = set(types)

    if lang is None:
        lang = "en"
    languages = {"en", lang}

    alts = {PRIMARY_ALT, alt}

    query = query.strip()
    title_like = "%" + query + "%"
    path_like = "/%" + query.rstrip("/") + "%"

    con = sqlite3.connect(builder.buildstate_database_filename, timeout=10)
    try:
        cur = con.cursor()
        cur.execute(
            f"""
            select path, alt, lang, type, title
              from source_info
             where (title like ? or path like ?)
               and lang in ({_placeholders(languages)})
               and alt in ({_placeholders(alts)})
               and type in ({_placeholders(types_)})
            order by title
            collate nocase
             limit ?
            """,
            [title_like, path_like, *languages, *alts, *types_, limit * 2],
        )
        return _process_search_results(builder, cur, alt, lang, limit)
    finally:
        con.close()
