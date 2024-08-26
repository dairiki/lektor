from __future__ import annotations

from typing import cast
from typing import Container
from typing import Iterable
from typing import Iterator
from typing import overload


def _line_is_dashes(line: str) -> int:
    line = line.strip()
    return line == "-" * len(line) and len(line) >= 3


def _process_buf(buf: Iterable[str]) -> list[str]:
    # remove leading dashes in dashed lines
    unescaped = [line[1:] if _line_is_dashes(line) else line for line in buf]
    if unescaped:
        last_line = unescaped[-1]
        if last_line.endswith("\n"):
            unescaped[-1] = last_line[:-1]

    return unescaped


@overload
def tokenize(
    iterable: Iterable[str], interesting_keys: Container[str], encoding: None = None
) -> Iterator[tuple[str, list[str] | None]]:
    ...


@overload
def tokenize(
    iterable: Iterable[bytes],
    interesting_keys: Container[str],
    encoding: str,
) -> Iterator[tuple[str, list[str] | None]]:
    ...


@overload
def tokenize(
    iterable: Iterable[str], interesting_keys: None = None, encoding: None = None
) -> Iterator[tuple[str, list[str]]]:
    ...


@overload
def tokenize(
    iterable: Iterable[bytes],
    interesting_keys: None,
    encoding: str,
) -> Iterator[tuple[str, list[str]]]:
    ...


@overload
def tokenize(
    iterable: Iterable[bytes],
    *,
    interesting_keys: None = None,
    encoding: str,
) -> Iterator[tuple[str, list[str]]]:
    ...


def tokenize(
    iterable: Iterable[str] | Iterable[bytes],
    interesting_keys: Container[str] | None = None,
    encoding: str | None = None,
) -> Iterator[tuple[str, list[str] | None]]:
    """This tokenizes an iterable of newlines as bytes into key value
    pairs out of the lektor bulk format.  By default it will process all
    fields, but optionally it can skip values of uninteresting keys and
    will instead yield `None`.  The values are left as list of decoded
    lines with their endings preserved.

    This will not perform any other processing on the data other than
    decoding and basic tokenizing.
    """
    key: list[str] = []
    buf: list[str] = []
    want_newline = False
    is_interesting = True

    if encoding is None:
        lines = cast(Iterable[str], iterable)
    else:
        lines = (x.decode(encoding, "replace") for x in cast(Iterable[bytes], iterable))

    def _flush_item() -> tuple[str, list[str] | None]:
        the_key = key[0]
        if not is_interesting:
            value = None
        else:
            value = _process_buf(buf)
        del key[:], buf[:]
        return the_key, value

    for line in lines:
        line = line.rstrip("\r\n") + "\n"

        if line.rstrip() == "---":
            want_newline = False
            if key:
                yield _flush_item()
        elif key:
            if want_newline:
                want_newline = False
                if not line.strip():
                    continue
            if is_interesting:
                buf.append(line)
        else:
            bits = line.split(":", 1)
            if len(bits) == 2:
                key = [bits[0].strip()]
                if interesting_keys is None:
                    is_interesting = True
                else:
                    is_interesting = key[0] in interesting_keys
                if is_interesting:
                    first_bit = bits[1].strip("\t ")
                    if first_bit.strip():
                        buf = [first_bit]
                    else:
                        buf = []
                        want_newline = True

    if key:
        yield _flush_item()


@overload
def serialize(
    iterable: Iterable[tuple[str, str]], encoding: None = None
) -> Iterator[str]:
    ...


@overload
def serialize(iterable: Iterable[tuple[str, str]], encoding: str) -> Iterator[bytes]:
    ...


def serialize(
    iterable: Iterable[tuple[str, str]], encoding: str | None = None
) -> Iterator[str] | Iterator[bytes]:
    """Serializes an iterable of key value pairs into a stream of
    string chunks.  If an encoding is provided, it will be encoded into that.

    This is primarily used by the editor to write back data to a source file.
    """

    if encoding is not None:
        for line in serialize(iterable):
            yield line.encode(encoding)
        return

    def _escape(line: str) -> str:
        if _line_is_dashes(line):
            return "-" + line
        return line

    for idx, (key, value) in enumerate(iterable):
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        if idx > 0:
            yield "---\n"
        if "\n" in value or value.strip("\t ") != value:
            yield key + ":\n"
            yield "\n"
            for line in value.splitlines(keepends=True):
                yield _escape(line)
            yield "\n"
        else:
            yield f"{key}: {value}\n"
