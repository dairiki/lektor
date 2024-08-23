from __future__ import annotations

import os
import sysconfig
import types
from typing import Any
from typing import Generator
from typing import TYPE_CHECKING

from lektor.utils import get_cache_dir

# Importing watchfiles currently segfault under free-threading python
# https://github.com/samuelcolvin/watchfiles/issues/299
WATCHFILES_IS_BORKED = sysconfig.get_config_var("Py_GIL_DISABLED")

if TYPE_CHECKING:
    from _typeshed import StrPath
    import watchfiles
    from watchfiles.main import FileChange
    from lektor.environment import Environment
elif WATCHFILES_IS_BORKED:
    watchfiles = types.SimpleNamespace(DefaultFilter=object)
else:
    import watchfiles


def watch_project(
    env: Environment, output_path: StrPath, **kwargs: Any
) -> Generator[set[FileChange], None, None]:
    """Watch project source files for changes.

    Returns an generator that yields sets of changes as they are noticed.

    Changes to files within ``output_path`` are ignored, along with other files
    deemed not to be Lektor source files.

    """
    if WATCHFILES_IS_BORKED:
        return iter([])  # type: ignore[return-value]

    watch_paths = [
        os.fspath(path)
        for path in (env.root_path, env.project.project_file, *env.theme_paths)
        if path is not None
    ]
    ignore_paths = [os.path.abspath(p) for p in (get_cache_dir(), output_path)]
    watch_filter = WatchFilter(env, ignore_paths=ignore_paths)

    return watchfiles.watch(*watch_paths, watch_filter=watch_filter, **kwargs)


class WatchFilter(watchfiles.DefaultFilter):
    def __init__(self, env: Environment, **kwargs: Any):
        super().__init__(**kwargs)
        self.env = env

    def __call__(self, change: watchfiles.Change, path: str) -> bool:
        if self.env.is_uninteresting_source_name(os.path.basename(path)):
            return False
        return super().__call__(change, path)
