import pytest

from lektor.context import Context
from lektor.context import DependencyIgnoringContextProxy
from lektor.context import disable_dependency_recording
from lektor.context import get_ctx


@pytest.fixture
def dummy_ctx(pad):
    with Context(pad=pad) as ctx:
        yield ctx


class Test_disable_dependency_recording:
    # pylint: disable=no-self-use

    @pytest.mark.usefixtures("dummy_ctx")
    def test(self):
        def ctx() -> Context:
            ctx = get_ctx()
            assert ctx is not None
            return ctx

        ctx().record_dependency("a")
        with disable_dependency_recording():
            ctx().record_dependency("b")
        ctx().record_dependency("c")
        assert ctx().referenced_dependencies == {"a", "c"}

    def test_no_context(self):
        assert get_ctx() is None
        with disable_dependency_recording():
            assert get_ctx() is None
        assert get_ctx() is None


class TestDependencyIgnoringContextProxy:
    # pylint: disable=no-self-use

    @pytest.fixture
    def proxy(self, dummy_ctx):
        return DependencyIgnoringContextProxy(dummy_ctx)

    def test_sets_context(self, proxy, dummy_ctx):
        assert get_ctx() is dummy_ctx
        with proxy:
            assert get_ctx() is proxy
        assert get_ctx() is dummy_ctx

    def test_isinstance(self, proxy):
        assert isinstance(proxy, DependencyIgnoringContextProxy)
        assert isinstance(proxy, Context)

    def test_cache(self, proxy, dummy_ctx):
        proxy.cache["test"] = "value"
        assert dummy_ctx.cache["test"] == "value"

    def test_record_dependency(self, proxy, dummy_ctx):
        dummy_ctx.record_dependency("a")
        proxy.record_dependency("b")
        assert proxy.referenced_dependencies == set("a")

    def test_record_dependency_accepts_affects_url(self, proxy):
        proxy.record_dependency("b", affects_url=True)
        assert proxy.referenced_dependencies == set()

    def test_record_virtual_dependency(self, proxy, dummy_ctx, pad):
        proxy.record_virtual_dependency(pad.get("/projects@1"))
        assert not proxy.referenced_virtual_dependencies
        dummy_ctx.record_virtual_dependency(pad.get("/projects@2"))
        assert proxy.referenced_virtual_dependencies
