from __future__ import annotations

from typing import Protocol


class SupportsUrlPath(Protocol):
    """Objects that have a ``url_path`` attribute.

    Various of our objects: Records, Assets, Thumbnails have such a `url_path` attributes
    that contains the URL path.
    """

    @property
    def url_path(self) -> str:
        ...
