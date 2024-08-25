from __future__ import annotations

from types import TracebackType
from typing import Protocol
from typing import Tuple
from typing import Type


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


class SupportsUrlPath(Protocol):
    """Objects that have a ``url_path`` attribute.

    Various of our objects: Records, Assets, Thumbnails have such a `url_path` attributes
    that contains the URL path.
    """

    @property
    def url_path(self) -> str:
        ...
