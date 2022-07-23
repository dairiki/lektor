from itertools import chain
from itertools import groupby
from itertools import islice
from operator import itemgetter

from lektor.constants import PRIMARY_ALT


def _iter_parents(path):
    path = path.strip("/")
    if path:
        pieces = path.split("/")
        for x in range(len(pieces)):
            yield "/" + "/".join(pieces[:x])


def _id_from_path(path):
    try:
        return path.strip("/").split("/")[-1]
    except IndexError:
        return ""


def _build_parent_path(path, infos):
    for parent in _iter_parents(path):
        id = _id_from_path(parent)
        info = infos.get(parent)
        if info is None:
            title = id or "(Index)"
        else:
            title = info.get("title")
        yield {"id": id, "path": parent, "title": title}


def _qmarks(values):
    """Return SQL placeholders for values."""
    return ",".join(["?"] * len(values))


def _best_info(infos, alt, lang):
    alt_lang_prefs = [
        (alt, lang),
        (PRIMARY_ALT, lang),
        (alt, "en"),
        (PRIMARY_ALT, "en"),
    ]

    def by_alt_lang(info):
        key = info["alt"], info["lang"]
        for n, alt_lang in enumerate(alt_lang_prefs):
            if key == alt_lang:
                return n
        return len(alt_lang_prefs)

    return min(infos, default=None, key=by_alt_lang)


def _process_rows(cur, alt, lang):
    column_names = [desc[0] for desc in cur.description]
    by_path = itemgetter("path")

    infos = (dict(zip(column_names, row)) for row in cur)

    for _, infos_ in groupby(sorted(infos, key=by_path), key=by_path):
        info = _best_info(infos_, alt, lang)
        if info is not None:
            yield info


def find_files(builder, query, alt=PRIMARY_ALT, lang=None, limit=50, types=None):
    if types is None:
        types = ["page"]
    else:
        types = list(types)
    languages = ["en"]
    if lang not in ("en", None):
        languages.append(lang)
    else:
        lang = "en"
    alts = [PRIMARY_ALT]
    if alt != PRIMARY_ALT:
        alts.append(alt)

    query = query.strip()
    title_like = "%" + query + "%"
    path_like = "/%" + query.rstrip("/") + "%"

    con = builder.connect_to_database()

    cur = con.execute(
        f"""
        SELECT path, alt, lang, type, title
          FROM source_info
         WHERE (title LIKE ? or PATH like ?)
           AND lang in ({_qmarks(languages)})
           AND alt in ({_qmarks(alts)})
           AND type in ({_qmarks(types)})
       COLLATE nocase
         LIMIT ?
        """,
        [title_like, path_like] + languages + alts + types + [limit * 2],
    )

    infos = {
        info["path"]: info for info in islice(_process_rows(cur, alt, lang), limit)
    }
    hits = sorted(infos.values(), key=itemgetter("title"))

    parent_paths = set(
        chain.from_iterable(_iter_parents(info["path"]) for info in hits)
    )
    paths_needed = parent_paths - set(infos.keys())

    if paths_needed:
        cur = con.execute(
            f"""
            SELECT path, alt, lang, type, title
              FROM source_info
             WHERE path in ({_qmarks(paths_needed)})
            """,
            list(paths_needed),
        )
        infos.update({info["path"]: info for info in _process_rows(cur, alt, lang)})

    for info in hits:
        info["id"] = _id_from_path(info["path"])
        info["parents"] = list(_build_parent_path(info["path"], infos))

    return hits
