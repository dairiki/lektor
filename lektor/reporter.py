from __future__ import annotations

import copy
import sys
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from traceback import TracebackException
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import TypedDict

import click
from click import style
from werkzeug.local import LocalProxy
from werkzeug.local import LocalStack

from lektor.utils import DeprecatedWarning

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import Unpack
else:
    from typing_extensions import Unpack

if TYPE_CHECKING:
    from _typeshed import StrPath
    from _typeshed import Unused

    from lektor.build_programs import SourceInfo
    from lektor.builder import Artifact
    from lektor.builder import ArtifactBuildFunc
    from lektor.builder import ArtifactId
    from lektor.builder import ArtifactsRow
    from lektor.builder import SourceId
    from lektor.builder import VsourceArtifactsRow
    from lektor.builder import Builder
    from lektor.environment import Environment
    from lektor.sourceobj import SourceObject
    from lektor.typing import ExcInfo


class ArtifactBuildFuncExceptionWarning(UserWarning):
    """Warning issued when the build_func for an artifact raise an exception."""

    def __init__(
        self,
        *args: Any,
        tb_exc: TracebackException,
        artifact: Artifact | None = None,
    ):
        super().__init__(*args)
        self.tb_exc = tb_exc
        self.artifact = artifact


BuildChangeCallback = Callable[["Artifact"], None]

_reporter_stack: LocalStack[Reporter] = LocalStack()


def describe_build_func(func: ArtifactBuildFunc) -> str:
    if hasattr(func, "func"):
        func = func.func  # unwrap functools.partial
    try:
        qualname = func.__qualname__
        class_name, _, method = qualname.rpartition(".")
        if class_name and method == "build_artifact":
            # Strip method name from methods of BuildProgram instances
            qualname = class_name
        return f"{func.__module__}.{qualname}"
    except AttributeError:
        return repr(func)


class Reporter:
    _change_callbacks: set[BuildChangeCallback] = set()

    def __init__(self, env: Environment | None = None, verbosity: int = 0):
        if env is not None:
            reason = (
                f"passing an `env` to {self.__class__.__name__}.__init__ is deprecated."
            )
            warnings.warn(
                DeprecatedWarning("env", reason=reason, version="3.4.0"),
                stacklevel=2,
            )

        self.verbosity = verbosity

        self.builder_stack: list[Builder] = []
        self.artifact_stack: list[Artifact] = []
        self.source_stack: list[SourceObject] = []

    def copy(self) -> Self:
        clone = copy.copy(self)
        clone.builder_stack = list(self.builder_stack)
        clone.artifact_stack = list(self.artifact_stack)
        clone.source_stack = list(self.source_stack)
        return clone

    def push(self) -> None:
        _reporter_stack.push(self)

    @staticmethod
    def pop() -> None:
        _reporter_stack.pop()

    def __enter__(self) -> Self:
        self.push()
        return self

    def __exit__(self, exc_type: Unused, exc_value: Unused, tb: Unused) -> None:
        self.pop()

    @property
    def builder(self) -> Builder | None:
        if self.builder_stack:
            return self.builder_stack[-1]
        return None

    @property
    def current_artifact(self) -> Artifact | None:
        if self.artifact_stack:
            return self.artifact_stack[-1]
        return None

    @property
    def current_source(self) -> SourceObject | None:
        if self.source_stack:
            return self.source_stack[-1]
        return None

    @property
    def show_build_info(self) -> bool:
        return self.verbosity >= 1

    @property
    def show_tracebacks(self) -> bool:
        return self.verbosity >= 1

    @property
    def show_current_artifacts(self) -> bool:
        return self.verbosity >= 2

    @property
    def show_artifact_internals(self) -> bool:
        return self.verbosity >= 3

    @property
    def show_source_internals(self) -> bool:
        return self.verbosity >= 3

    @property
    def show_debug_info(self) -> bool:
        return self.verbosity >= 4

    @contextmanager
    def build(self, activity: str, builder: Builder) -> Iterator[None]:
        now = time.time()
        self.builder_stack.append(builder)
        self.start_build(activity)
        try:
            yield
        finally:
            self.builder_stack.pop()
            self.finish_build(activity, now)

    @contextmanager
    def on_build_change(self, callback: BuildChangeCallback) -> Iterator[None]:
        self._change_callbacks.add(callback)
        try:
            yield
        finally:
            self._change_callbacks.discard(callback)

    def start_build(self, activity: str) -> None:
        pass

    def finish_build(self, activity: str, start_time: float) -> None:
        pass

    @contextmanager
    def build_artifact(
        self, artifact: Artifact, build_func: ArtifactBuildFunc, is_current: bool
    ) -> Iterator[None]:
        now = time.time()
        self.artifact_stack.append(artifact)
        self.start_artifact_build(is_current)
        self.report_build_func(build_func)
        try:
            yield
        finally:
            self.report_artifact_built(artifact, is_current)
            self.finish_artifact_build(now)
            self.artifact_stack.pop()

    def report_artifact_built(self, artifact: Artifact, is_current: bool) -> None:
        if is_current:
            return
        for callback in self._change_callbacks:
            callback(artifact)

    def start_artifact_build(self, is_current: bool) -> None:
        pass

    def finish_artifact_build(self, start_time: float) -> None:
        pass

    def report_failure(self, artifact: Artifact, exc_info: ExcInfo) -> None:
        # In general, we always want to report exceptions.  Otherwise, if
        # an exception is raised by an artifact build_func in a unit test,
        # we get no indication.
        tb_exc = TracebackException(*exc_info, limit=-6, compact=True)

        message = "".join(tb_exc.format_exception_only())
        loc = []
        if self.current_artifact:
            loc.append(f"while building {self.current_artifact.artifact_id!r}")
        if self.current_source:
            loc.append(f"for {self.current_source!r}")
        if loc:
            message = f"{message.rstrip()}, {' '.join(loc)}\n"

        lines = (message, *tb_exc.format(chain=True))
        full_message = "| ".join(lines).rstrip()

        warnings.warn(
            ArtifactBuildFuncExceptionWarning(
                full_message, tb_exc=tb_exc, artifact=self.current_artifact
            ),
            stacklevel=2,
        )

    def report_build_all_failure(self, failures: int) -> None:
        pass

    def report_dependencies(
        self, dependencies: Iterable[ArtifactsRow | VsourceArtifactsRow]
    ) -> None:
        for dep in dependencies:
            self.report_debug_info("dependency", dep.source)

    def report_dirty_flag(self, value: bool) -> None:
        pass

    def report_write_source_info(self, info: SourceInfo) -> None:
        pass

    def report_prune_source_info(self, source: SourceId) -> None:
        pass

    def report_sub_artifact(self, artifact: Artifact) -> None:
        pass

    def report_build_func(self, build_func: ArtifactBuildFunc) -> None:
        pass

    def report_debug_info(self, key: str, value: object) -> None:
        pass

    def report_generic(self, message: str) -> None:
        pass

    def report_pruned_artifact(self, artifact_id: ArtifactId) -> None:
        pass

    @contextmanager
    def process_source(self, source: SourceObject) -> Iterator[None]:
        now = time.time()
        self.source_stack.append(source)
        self.enter_source()
        try:
            yield
        finally:
            self.leave_source(now)
            self.source_stack.pop()

    def enter_source(self) -> None:
        pass

    def leave_source(self, start_time: float) -> None:
        pass


class NullReporter(Reporter):
    pass


class _ReportData(TypedDict, total=False):
    activity: str
    artifact: Artifact | None
    artifact_id: ArtifactId
    exc_info: ExcInfo
    failures: int
    func: str
    info: SourceInfo
    is_current: bool
    key: str
    message: str
    source: SourceId | SourceObject | None
    value: bool | object


class _Report(NamedTuple):
    event: str
    data: _ReportData


class BufferReporter(Reporter):
    def __init__(self, env: Environment | None = None, verbosity: int = 0):
        super().__init__(env, verbosity)
        self.buffer: list[_Report] = []

    def clear(self) -> None:
        self.buffer.clear()

    def get_recorded_dependencies(self) -> Collection[SourceId | StrPath]:
        deps = {
            data["value"]
            for event, data in self.buffer
            if event == "debug-info" and data["key"] == "dependency"
        }
        return deps  # type: ignore[return-value]

    def get_major_events(self) -> list[_Report]:
        return [
            report
            for report in self.buffer
            if report.event not in ("debug-info", "dirty-flag", "write-source-info")
        ]

    def get_failures(self) -> list[_ReportData]:
        rv = []
        for event, data in self.buffer:
            if event == "failure":
                rv.append(data)
        return rv

    def _emit(self, _event: str, **extra: Unpack[_ReportData]) -> None:
        self.buffer.append(_Report(_event, extra))

    def start_build(self, activity: str) -> None:
        self._emit("start-build", activity=activity)

    def finish_build(self, activity: str, start_time: float) -> None:
        self._emit("finish-build", activity=activity)

    def start_artifact_build(self, is_current: bool) -> None:
        self._emit(
            "start-artifact-build",
            artifact=self.current_artifact,
            is_current=is_current,
        )

    def finish_artifact_build(self, start_time: float) -> None:
        self._emit("finish-artifact-build", artifact=self.current_artifact)

    def report_build_all_failure(self, failures: int) -> None:
        self._emit("build-all-failure", failures=failures)

    def report_failure(self, artifact: Artifact, exc_info: ExcInfo) -> None:
        self._emit("failure", artifact=artifact, exc_info=exc_info)

    def report_dirty_flag(self, value: bool) -> None:
        self._emit("dirty-flag", artifact=self.current_artifact, value=value)

    def report_write_source_info(self, info: SourceInfo) -> None:
        self._emit("write-source-info", info=info, artifact=self.current_artifact)

    def report_prune_source_info(self, source: SourceId) -> None:
        self._emit("prune-source-info", source=source)

    def report_build_func(self, build_func: ArtifactBuildFunc) -> None:
        self._emit("build-func", func=describe_build_func(build_func))

    def report_sub_artifact(self, artifact: Artifact) -> None:
        self._emit("sub-artifact", artifact=artifact)

    def report_debug_info(self, key: str, value: object) -> None:
        self._emit("debug-info", key=key, value=value)

    def report_generic(self, message: str) -> None:
        self._emit("generic", message=message)

    def enter_source(self) -> None:
        self._emit("enter-source", source=self.current_source)

    def leave_source(self, start_time: float) -> None:
        self._emit("leave-source", source=self.current_source)

    def report_pruned_artifact(self, artifact_id: ArtifactId) -> None:
        self._emit("pruned-artifact", artifact_id=artifact_id)


class CliReporter(Reporter):
    def __init__(self, env: Environment, verbosity: int = 0):
        super().__init__(verbosity=verbosity)
        self.env = env
        self.indentation = 0

    def indent(self) -> None:
        self.indentation += 1

    def outdent(self) -> None:
        self.indentation -= 1

    def _write_line(self, text: str) -> None:
        line = f"{'  ' * self.indentation} {text}"
        current_thread = threading.current_thread()
        if current_thread is not threading.main_thread():
            line += style(f" {self.current_source}", fg="cyan")

        click.echo(line)

    def _write_kv_info(self, key: str, value: object) -> None:
        self._write_line(f"{key}: {style(str(value), fg='yellow')}")

    def start_build(self, activity: str) -> None:
        self._write_line(style("Started %s" % activity, fg="cyan"))
        if not self.show_build_info:
            return
        builder = self.builder
        if builder is None:
            return
        self._write_line(style(f"  Tree: {self.env.root_path}", fg="cyan"))
        self._write_line(style(f"  Output path: {builder.destination_path}", fg="cyan"))

    def finish_build(self, activity: str, start_time: float) -> None:
        self._write_line(
            style(
                f"Finished {activity} in {time.time() - start_time:.2f} sec",
                fg="cyan",
            )
        )

    def start_artifact_build(self, is_current: bool) -> None:
        if self.current_artifact:
            artifact_id = str(self.current_artifact.artifact_id)
        else:
            artifact_id = click.style("<artifact unkown>", fg="red")

        if is_current:
            if not self.show_current_artifacts:
                return
            sign = click.style("X", fg="cyan")
        else:
            sign = click.style("U", fg="green")
        self._write_line(f"{sign} {artifact_id}")

        self.indent()

    def finish_artifact_build(self, start_time: float) -> None:
        self.outdent()

    def report_build_all_failure(self, failures: int) -> None:
        unit = "failure" if failures == 1 else "failures"
        self._write_line(
            click.style(f"Error: Build failed with {failures} {unit}.", fg="red")
        )

    def report_failure(self, artifact: Artifact, exc_info: ExcInfo) -> None:
        sign = click.style("E", fg="red")
        err = " ".join(
            "".join(traceback.format_exception_only(*exc_info[:2])).splitlines()
        ).strip()
        self._write_line(f"{sign} {artifact.artifact_id} ({err})")

        if not self.show_tracebacks:
            return

        tb = traceback.format_exception(*exc_info)
        for line in "".join(tb).splitlines():
            if line.startswith("Traceback "):
                line = click.style(line, fg="red")
            elif line.startswith("  File "):
                line = click.style(line, fg="yellow")
            elif not line.startswith("    "):
                line = click.style(line, fg="red")
            self._write_line("  " + line)

    def report_dirty_flag(self, value: bool) -> None:
        if self.show_artifact_internals and (value or self.show_debug_info):
            self._write_kv_info("forcing sources dirty", value)

    def report_write_source_info(self, info: SourceInfo) -> None:
        if self.show_artifact_internals and self.show_debug_info:
            self._write_kv_info("writing source info", f"{info.title_en} [{info.type}]")

    def report_prune_source_info(self, source: SourceId) -> None:
        if self.show_artifact_internals and self.show_debug_info:
            self._write_kv_info("pruning source info", source)

    def report_build_func(self, build_func: ArtifactBuildFunc) -> None:
        if self.show_artifact_internals:
            self._write_kv_info("build program", describe_build_func(build_func))

    def report_sub_artifact(self, artifact: Artifact) -> None:
        if self.show_artifact_internals:
            self._write_kv_info("sub artifact", artifact.artifact_id)

    def report_debug_info(self, key: str, value: object) -> None:
        if self.show_debug_info:
            self._write_kv_info(key, value)

    def report_generic(self, message: str) -> None:
        self._write_line(style(str(message), fg="cyan"))

    def enter_source(self) -> None:
        if not self.show_source_internals:
            return
        source_repr = style(repr(self.current_source), fg="magenta")
        self._write_line(f"Source {source_repr}")
        self.indent()

    def leave_source(self, start_time: float) -> None:
        if self.show_source_internals:
            self.outdent()

    def report_pruned_artifact(self, artifact_id: ArtifactId) -> None:
        sign = style("D", fg="red")
        self._write_line(f"{sign} {artifact_id}")


null_reporter = NullReporter()


reporter: Reporter  # lie about the type


@LocalProxy  # type: ignore[no-redef]
def reporter() -> Reporter:
    rv = _reporter_stack.top
    if rv is None:
        rv = null_reporter
    return rv
