from __future__ import annotations

import sys
from typing import Any
from typing import Generic
from typing import Iterator
from typing import Protocol
from typing import TYPE_CHECKING
from typing import TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from lektor.datamodel import PaginationConfig
    from lektor.db import Pad
    from lektor.db import Query


class Paginatable(Protocol):
    """This describes the parts of Page that Pagination requires."""

    def __init__(self, pad: Pad, data: dict[str, Any], page_num: int | None = None):
        ...

    @property
    def pad(self) -> Pad:
        ...

    @property
    def path(self) -> str:
        ...

    @property
    def alt(self) -> str:
        ...

    @property
    def _data(self) -> dict[str, Any]:
        ...

    @property
    def page_num(self) -> int | None:
        ...

    @property
    def children(self) -> Query:
        ...

    @property
    def pagination(self) -> Pagination[Self]:
        ...


_Page = TypeVar("_Page", bound=Paginatable)


class Pagination(Generic[_Page]):
    def __init__(self, record: _Page, pagination_config: PaginationConfig):
        #: the pagination config
        self.config = pagination_config
        #: the current page's record
        self.current = record
        #: the current page number (1 indexed)
        self.page = record.page_num
        #: the number of items to be displayed on a page.
        self.per_page = pagination_config.per_page
        #: the total number of items matching the query
        self.total = pagination_config.count_total_items(record)

    @property
    def items(self) -> Query:
        """The children for this page."""
        return self.config.slice_query_for_page(self.current, self.page)

    @property
    def pages(self) -> int:
        """The total number of pages."""
        pages = (self.total + self.per_page - 1) // self.per_page
        # Even when there are no children, we want at least one page
        return max(pages, 1)

    @property
    def prev_num(self) -> int | None:
        """The page number of the previous page."""
        if self.page is not None:
            if self.page > 1:
                return self.page - 1
        return None

    @property
    def has_prev(self) -> bool:
        """True if a previous page exists."""
        if self.page is not None:
            return self.page > 1
        return False

    @property
    def prev(self) -> _Page | None:
        """The record for the previous page."""
        if self.page is not None and self.has_prev:
            return self.config.get_record_for_page(self.current, self.page - 1)
        return None

    @property
    def has_next(self) -> bool:
        """True if a following page exists."""
        if self.page is not None:
            return self.page < self.pages
        return False

    @property
    def next_num(self) -> int | None:
        """The page number of the following page."""
        if self.page is not None:
            if self.page < self.pages:
                return self.page + 1
        return None

    @property
    def next(self) -> _Page | None:
        """The record for the following page."""
        if self.page is not None and self.has_next:
            return self.config.get_record_for_page(self.current, self.page + 1)
        return None

    def for_page(self, page: int) -> _Page | None:
        """Returns the record for a specific page."""
        if 1 <= page <= self.pages:
            return self.config.get_record_for_page(self.current, page)
        return None

    def iter_pages(
        self,
        left_edge: int = 2,
        left_current: int = 2,
        right_current: int = 5,
        right_edge: int = 2,
    ) -> Iterator[int | None]:
        """Iterate over the page numbers in the pagination, with elision.

        In the general case, this returns the concatenation of three ranges:

            1. A range (always starting at page one) at the beginning
               of the page number sequence.  The length of the this
               range is specified by the ``left_edge`` argument (which
               may be zero).

            2. A range around the current page.  This range will
               include ``left_current`` pages before, and
               ``right_current`` pages after the current page.  This
               range always includes the current page.

            3. Finally, a range (always ending at the last page) at
               the end of the page sequence.  The length of this range
               is specified by the ``right_edge`` argument.

        If any of these ranges overlap, they will be merged.  A
        ``None`` will be inserted between non-overlapping ranges to
        signify that pages have been elided.

        This is how you could render such a pagination in the templates:
        .. sourcecode:: html+jinja
            {% macro render_pagination(pagination, endpoint) %}
              <div class=pagination>
              {%- for page in pagination.iter_pages() %}
                {% if page %}
                  {% if page != pagination.page %}
                    <a href="{{ url_for(endpoint, page=page) }}">{{ page }}</a>
                  {% else %}
                    <strong>{{ page }}</strong>
                  {% endif %}
                {% else %}
                  <span class=ellipsis>...</span>
                {% endif %}
              {%- endfor %}
              </div>
            {% endmacro %}

        """
        if self.page is None:
            return

        last = 0
        for num in range(1, self.pages + 1):
            # pylint: disable=chained-comparison
            if (
                num <= left_edge
                or (
                    num >= self.page - left_current and num <= self.page + right_current
                )
                or num > self.pages - right_edge
            ):
                if last + 1 != num:
                    yield None
                yield num
                last = num
        if last != self.pages:
            yield None
