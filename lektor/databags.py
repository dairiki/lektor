from __future__ import annotations

import errno
import json
import os
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import TYPE_CHECKING
from typing import Union

from inifile import IniFile

from lektor.context import get_ctx
from lektor.utils import decode_flat_data
from lektor.utils import DecodedFlatDataType
from lektor.utils import iter_dotted_path_prefixes
from lektor.utils import merge
from lektor.utils import resolve_dotted_value

# XXX: should probably just move all these from lektor.utils to here.
# They are not used elsewhere and are pretty idiosyncratic.

if TYPE_CHECKING:
    from _typeshed import StrPath

    from lektor.environment import Environment


JSONValue = Union[
    None, bool, str, float, int, List["JSONValue"], Dict[str, "JSONValue"]
]
IniFileValue = DecodedFlatDataType[str]

DatabagType = Union[JSONValue, IniFileValue]


def load_databag(filename: StrPath) -> DatabagType | None:
    path = Path(filename)
    try:
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as f:
                return json.load(f, object_pairs_hook=OrderedDict)
        elif path.suffix == ".ini":
            return decode_flat_data(IniFile(path).items(), dict_cls=OrderedDict)
        else:
            return None
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        return None


class Databags:
    def __init__(self, env: Environment):
        self.env = env
        self.root_path = Path(self.env.root_path, "databags")
        self._known_bags: dict[str, list[Path]] = {}
        self._bags: dict[str, tuple[DatabagType, list[Path]]] = {}
        with suppress(OSError):
            for bag_path in self.root_path.iterdir():
                if bag_path.suffix in (".ini", ".json") and bag_path.is_file():
                    self._known_bags.setdefault(bag_path.stem, []).append(bag_path)

    def get_bag(self, name: str) -> DatabagType | None:
        sources = self._known_bags.get(name)
        if not sources:
            return None

        if name not in self._bags:
            bag_data: DatabagType = OrderedDict()
            for source in sources:
                bag_data = cast(DatabagType, merge(bag_data, load_databag(source)))
            self._bags[name] = bag_data, list(sources)
        else:
            bag_data, sources = self._bags[name]

        ctx = get_ctx()
        if ctx is not None:
            for source in sources:
                ctx.record_dependency(os.fspath(source))

        return bag_data

    def lookup(self, key: str) -> DatabagType | str | None:
        for prefix, local_key in iter_dotted_path_prefixes(key):
            bag = self.get_bag(prefix)
            if bag is not None:
                if local_key is None:
                    return bag
                return resolve_dotted_value(bag, local_key)
        return None
