"""Query the source_info database for matching records.

This is the implementation for the ``Builder.find_files`` method.

It is used by the Admin API to support the find-page functionality in the Admin UI.

"""
from __future__ import annotations

from collections.abc import Sized
from itertools import chain
from itertools import islice
from operator import itemgetter
from typing import Collection
from typing import Iterator
from typing import TYPE_CHECKING
from typing import TypedDict

from lektor.constants import PRIMARY_ALT
from lektor.utils import unique_everseen

if TYPE_CHECKING:
    from lektor.builder import Builder


class Breadcrumb(TypedDict):
    path: str
    title: str
    id: str                     # XXX: unused


class FindFileResult(TypedDict):
    # FIXME: more specific types
    path: str
    alt: str
    title: str
    type: str
    lang: str
    id: str  # XXX: unused
    parents: list[Breadcrumb]


def _iter_parents(path: str) -> Iterator[str]:  # XXX: more specific type? (DB path)
    path = path.strip("/")
    if path:
        pieces = path.split("/")
        for x in range(len(pieces)):
            yield "/" + "/".join(pieces[:x])


def _id_from_path(path: str) -> str:
    _, _, id_ = path.strip("/").rpartition("/")
    return id_


def _get_breadcrumb_data(
    builder: Builder, paths: Collection[str], *, alt: str, lang: str
) -> dict[str, list[Breadcrumb]]:
    """Fetch information about parents."""

    parent_paths = set(chain.from_iterable(_iter_parents(path) for path in paths))

    cur = builder.build_db.execute(
        f"""
        SELECT path, title
        FROM source_info
        WHERE path in ({_placeholders(parent_paths)})
        ORDER BY
            -- Sort by order of preference. We will keep only the first result
            -- for each path.
            CASE
                WHEN lang = ? THEN 1  -- prefer requested lang
                WHEN lang = ? THEN 2  -- fallback to default lang ("en")
                ELSE 3
            END,
            CASE
                WHEN alt = ? THEN 1  -- prefer requested alt
                WHEN alt = ? THEN 2  -- fallback to PRIMARY_ALT
                ELSE 3
            END
        """,
        [*parent_paths, lang, "en", alt, PRIMARY_ALT],
    )
    rows = unique_everseen(cur, key=itemgetter(0))  # take first match for each path
    titles_by_path = {str(path): str(title) for path, title in rows}

    def title_for_path(path: str) -> str:
        try:
            return titles_by_path[path]
        except KeyError:
            return _id_from_path(path) or "(Index)"

    return {
        path: [
            {"path": ppath, "title": title_for_path(ppath), "id": _id_from_path(ppath)}
            for ppath in _iter_parents(path)
        ]
        for path in paths
    }


def _placeholders(values: Sized) -> str:
    return ", ".join(["?"] * len(values))


def find_files(
    builder: Builder,
    query: str,
    alt: str = PRIMARY_ALT,
    lang: str | None = None,
    limit: int = 50,
    types: Collection[str] | None = None,
) -> list[FindFileResult]:
    """Query the source_info database for records whose title or path contain the query
    string.

    Records for the specified ``alt`` as well as the default (``PRIMARY_ALT``) alt are
    searched.  Titles are searched in the language specified by ``lang``, as well as the
    default language (which is always ``"en"``).

    For each db path, if multiple records match, only the "best match" is kept.

    Results will be returned as a sequence of dicts containing the following keys:

      - ``path`` - the db path to the matched record
      - ``type`` - the SourceInfo.type (e.g. "page")
      - ``alt`` - the alt of the record
      - ``lang`` - the lang of the title
      - ``title`` - the title of the source
      - ``id`` - the id, or last component, of the matched record's path (*deprecated*)
      - ``parents`` - breadcrumbs: a list of information about the ancestors of the record.
        This is a list, ordered from root to leaves, containing a dict for each ancestor.
        Each dict has the keys:
          - ``path`` - db path of the ancestor
          - ``title`` - display title of the ancestor
          - ``id`` - the id, or last component, of the ancestor's path (*deprecated*)

    """
    if types is None:
        types = {"page"}
    else:
        types = set(types)
    if lang is None:
        lang = "en"

    query = query.strip()
    title_query = "%" + query + "%"
    path_query = "/%" + query.rstrip("/") + "%"

    cur = builder.build_db.execute(
        f"""
        SELECT path, alt, lang, type, title
        FROM source_info
        WHERE (title like ? or path like ?)
            AND (alt = ? OR alt = ?)
            AND (lang = ? OR lang = ?)
            AND type IN ({_placeholders(types)})
        ORDER BY
            -- Sort by order of preference. We will keep only the first result
            -- for each path.
            CASE
                WHEN lang = ? THEN 1  -- prefer requested lang
                ELSE 2
            END,
            CASE
                WHEN alt = ? THEN 1  -- prefer requested alt
                ELSE 2
            END
        """,
        [title_query, path_query, alt, PRIMARY_ALT, lang, "en", *types, lang, alt],
    )
    rows = list(
        islice(
            unique_everseen(cur, key=itemgetter(0)),  # take first match for each path
            limit,
        )
    )
    paths = [path for path, *_ in rows]
    breadcrumbs = _get_breadcrumb_data(builder, paths, alt=alt, lang=lang)

    return [
        {
            "path": path,
            "alt": _alt,
            "title": title,
            "type": type_,
            "lang": _lang,
            "id": _id_from_path(path),
            "parents": breadcrumbs[path],
        }
        for path, _alt, _lang, type_, title in rows
    ]
