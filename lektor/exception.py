from __future__ import annotations

from typing import cast


class LektorException(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message)

    @property
    def message(self) -> str:
        return cast(str, self.args[0])

    def to_json(self) -> dict[str, str]:
        return {
            "type": self.__class__.__name__,
            "message": self.message,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"
