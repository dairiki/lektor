from __future__ import annotations

import os.path
import weakref
from pathlib import Path
from unittest import mock

import pytest

from lektor.builder import _is_packed_virtual_source_path
from lektor.builder import _pack_virtual_source_path
from lektor.builder import _unpack_virtual_source_path
from lektor.builder import FileInfo
from lektor.builder import PathCache
from lektor.project import Project
from lektor.reporter import NullReporter


def get_child_sources(prog):
    return sorted(list(prog.iter_child_sources()), key=lambda x: x["_id"])


def test_basic_build(pad, builder):
    root = pad.root

    prog, build_state = builder.build(root)
    assert prog.source is root
    assert build_state.failed_artifacts == []

    (artifact,) = prog.artifacts
    # Root and its thumbnail image were updated.
    assert artifact in build_state.updated_artifacts
    assert artifact.artifact_id == "index.html"
    assert set(artifact.sources) == set(root.iter_source_filenames())
    assert artifact.extra is None
    assert artifact.config_hash is None


def test_child_sources_basic(pad, builder):
    extra = pad.get("/extra")

    prog, _ = builder.build(extra)
    child_sources = get_child_sources(prog)

    assert [x["_id"] for x in child_sources] == [
        "a",
        "b",
        "container",  # hidden children should be built, too
        "file.ext",
        "hello.txt",
        "paginated",
        "slash-slug",
    ]


def test_child_sources_pagination(pad, builder):
    projects = pad.get("/projects")

    def ids(sources):
        return [x["_id"] for x in sources]

    prog, _ = builder.build(projects)

    child_sources = get_child_sources(prog)
    assert ids(child_sources) == [
        "attachment.txt",
        "filtered",
        "projects",
        "projects",
        "secret",
    ]

    page1 = child_sources[2]
    assert page1["_id"] == "projects"
    assert page1.page_num == 1
    page2 = child_sources[3]
    assert page2["_id"] == "projects"
    assert page2.page_num == 2

    prog, _ = builder.build(page1)
    child_sources_p1 = get_child_sources(prog)

    assert ids(child_sources_p1) == [
        "bagpipe",
        "coffee",
        "master",
        "oven",
    ]

    prog, _ = builder.build(page2)
    child_sources_p2 = get_child_sources(prog)

    assert ids(child_sources_p2) == [
        "postage",
        "slave",
        "wolf",
    ]


def test_basic_artifact_current_test(pad, builder, reporter):
    post1 = pad.get("blog/post1")

    build_result = builder.build(post1)
    artifact = build_result.primary_artifact

    assert reporter.get_major_events() == [
        (
            "enter-source",
            {
                "source": post1,
            },
        ),
        (
            "start-artifact-build",
            {
                "artifact": artifact,
                "is_current": False,
            },
        ),
        (
            "build-func",
            {
                "func": "lektor.build_programs.PageBuildProgram",
            },
        ),
        (
            "finish-artifact-build",
            {
                "artifact": artifact,
            },
        ),
        (
            "leave-source",
            {
                "source": post1,
            },
        ),
    ]

    assert set(reporter.get_recorded_dependencies()) == {
        "Website.lektorproject",
        "content/blog/post1/contents+en.lr",
        "content/blog/post1/contents.lr",
        "content/blog/post1/hello.txt+en.lr",
        "content/blog/post1/hello.txt.lr",
        "templates/blog-post.html",
        "templates/layout.html",
        "models/blog-post.ini",
    }

    reporter.clear()
    build_result2 = builder.build(post1)
    artifact2 = build_result2.primary_artifact

    assert reporter.get_major_events() == [
        (
            "enter-source",
            {
                "source": post1,
            },
        ),
        (
            "start-artifact-build",
            {
                "artifact": artifact2,
                "is_current": True,
            },
        ),
        (
            "build-func",
            {
                "func": "lektor.build_programs.PageBuildProgram",
            },
        ),
        (
            "finish-artifact-build",
            {
                "artifact": artifact2,
            },
        ),
        (
            "leave-source",
            {
                "source": post1,
            },
        ),
    ]


def test_basic_template_rendering(pad, builder):
    prog, _ = builder.build(pad.root)
    artifact = prog.artifacts[0]

    assert artifact.artifact_id == "index.html"
    content = Path(artifact.dst_filename).read_text("utf-8")

    assert "<title>My Website</title>" in content
    assert "<h1>Welcome</h1>" in content
    assert '<link href="static/style.css" rel="stylesheet">' in content
    assert "<p>Welcome to this pretty nifty website.</p>" in content


def test_attachment_copying(pad, builder):
    root = pad.root
    text_file = root.attachments.get("hello.txt")

    prog, _ = builder.build(text_file)
    artifact = prog.artifacts[0]

    assert artifact.artifact_id == "hello.txt"
    content = Path(artifact.dst_filename).read_text("utf-8")
    assert content.strip() == "Hello I am an Attachment"


def test_asset_processing(pad, builder):
    static = pad.asset_root.get_child("static")

    prog, _ = builder.build(static)
    assets = list(prog.iter_child_sources())
    assert len(assets) == 1
    assert assets[0].name == "demo.css"

    prog, _ = builder.build(assets[0])
    content = Path(prog.primary_artifact.dst_filename).read_text("utf-8")
    assert "color: red" in content


def test_included_assets(pad, builder):
    # In demo-project/Website.lektorproject, included_assets = "_*".
    root = pad.asset_root

    prog, _ = builder.build(root)
    assets = list(prog.iter_child_sources())
    assert "_include_me_despite_underscore" in [a.name for a in assets]


def test_excluded_assets(pad, builder):
    # In demo-project/Website.lektorproject, excluded_assets = "foo*".
    root = pad.asset_root

    prog, _ = builder.build(root)
    assets = list(prog.iter_child_sources())
    assert "foo-prefix-makes-me-excluded" not in [a.name for a in assets]


def test_iter_child_pages(child_sources_test_project_builder):
    # Test that child sources are built even if they're filtered out by a
    # pagination query like "this.children.filter(F._model == 'doesnt-exist')".
    builder = child_sources_test_project_builder
    pad = builder.pad
    prog, _ = builder.build(pad.root)
    assert builder.pad.get("filtered-page") in prog.iter_child_sources()


def test_iter_child_attachments(child_sources_test_project_builder):
    # Test that attachments are built, even if a pagination has no items.
    builder = child_sources_test_project_builder
    pad = builder.pad
    prog, _ = builder.build(pad.root)
    assert builder.pad.get("attachment.txt") in prog.iter_child_sources()


@pytest.mark.parametrize(
    "parent_path, child_name",
    [
        ("/extra", "container"),  # a hidden page
        ("/extra/container", "a"),  # a page whose parent is hidden
        ("/extra/container", "hello.txt"),  # an attachment whose parent is hidden
    ],
)
def test_iter_children_of_hidden_pages(builder, pad, parent_path, child_name):
    # Test that child sources are built even if they're parent is hidden
    parent = pad.get(parent_path)
    child = pad.get(f"{parent_path}/{child_name}")
    # sanity checks
    assert parent is not None and child is not None
    assert parent.is_hidden or child.is_hidden

    prog, _ = builder.build(parent)
    assert child in prog.iter_child_sources()


def test_record_is_file(pad, builder):
    record = pad.get("/extra/file.ext")

    prog, _ = builder.build(record)
    (artifact,) = prog.artifacts
    assert artifact.artifact_id == "extra/file.ext"


def test_slug_contains_slash(pad, builder):
    record = pad.get("/extra/slash-slug")

    prog, _ = builder.build(record)
    (artifact,) = prog.artifacts
    assert artifact.artifact_id == "extra/long/path/index.html"


def test_asseturl_dependency_tracking_integration(
    scratch_project_data, scratch_pad, scratch_builder
):
    scratch_project_data.joinpath("templates/page.html").write_text(
        "{{ '/asset.txt'|asseturl }}", "utf-8"
    )
    asset_txt = scratch_project_data / "assets/asset.txt"
    asset_txt.parent.mkdir(exist_ok=True)
    asset_txt.write_text("an asset\n", "utf-8")

    prog, build_state = scratch_builder.build(scratch_pad.root)
    assert len(build_state.updated_artifacts) == 1
    output = Path(prog.artifacts[0].dst_filename)
    asset_url = output.read_text(encoding="utf-8").rstrip()

    prog, build_state = scratch_builder.build(scratch_pad.root)
    assert len(build_state.updated_artifacts) == 0

    asset_txt.write_text("updated asset\n", "utf-8")
    prog, build_state = scratch_builder.build(scratch_pad.root)
    updated_asset_url = output.read_text(encoding="utf-8").rstrip()
    assert updated_asset_url != asset_url
    assert len(build_state.updated_artifacts) == 1


def test_prune_remove_artifacts_of_hidden_pages(scratch_project_data, scratch_builder):
    pad = scratch_builder.pad
    # Build root page
    prog, _ = scratch_builder.build(pad.root)
    (artifact,) = prog.artifacts
    assert Path(artifact.dst_filename).is_file()

    # Do a prune.  Output artifact should survive
    pad.cache.flush()
    scratch_builder.prune()
    assert Path(artifact.dst_filename).is_file()

    # Mark page as hidden, prune should then clean the artifact
    contents_lr = scratch_project_data.joinpath("content/contents.lr")
    contents_lr.write_text(contents_lr.read_text() + "\n---\n_hidden: yes\n")
    pad.cache.flush()
    scratch_builder.prune()
    assert not Path(artifact.dst_filename).is_file()


def test_prune_all(builder):
    pad = builder.pad
    # Don't build / — it has a zillion thumbnail images and so is slow to build
    src = pad.get("/blog/post1")
    prog, _ = builder.build(src)
    (artifact,) = prog.artifacts
    assert Path(artifact.dst_filename).is_file()
    pad.cache.flush()
    builder.prune(all=True)
    assert not Path(artifact.dst_filename).is_file()


class AssertBuildsNothingReporter(NullReporter):
    """Reporter to collect source objects which are built during a build cycle."""

    def start_artifact_build(self, is_current):
        assert is_current


def test_second_build_all_builds_nothing(scratch_builder, scratch_project_data):
    # This excercises a bug having to do with not properly tracking the alt
    # of virtual source object dependencies.  See #1108 (and #1007, #959).

    # The project should have multiple alts configured. (Scratch_project does.)

    # Add child pages to scratch_project
    for child in "child1", "child2":
        child_lr = scratch_project_data / "content" / child / "contents.lr"
        child_lr.parent.mkdir()
        child_lr.write_text("_template: child.html\n")
    # Reference siblings
    scratch_project_data.joinpath("templates/child.html").write_text(
        "{{ this.get_siblings() }}"
    )

    scratch_builder.build_all()

    with AssertBuildsNothingReporter():
        scratch_builder.build_all()


def test_BuildState_to_source_id(env, builder):
    build_state = builder.new_build_state()
    assert build_state.to_source_id("blog/.././foo.txt") == "foo.txt"


@pytest.fixture
def path_cache(env):
    return PathCache(env)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("filename", "filename"),
        ("a/../b", "b"),
    ],
)
def test_PathCache_to_source_id(path_cache, env, path, expected):
    root_path = Path(env.root_path)
    assert path_cache.to_source_id(root_path / path) == expected
    assert path_cache.to_source_id(os.fspath(root_path / path)) == expected
    assert path_cache.to_source_id(path) == expected
    assert path_cache.to_source_id(Path(path)) == expected


def test_PathCache_to_source_id_fails_for_paths_outside_tree(
    path_cache, tmp_path
):
    with pytest.raises(ValueError, match="is not below the source folder"):
        path_cache.to_source_id(tmp_path / "filename")


def test_PathCache_to_source_id_with_symlinked_project(data_path, tmp_path):
    demo_project = data_path / "demo-project"
    tree = tmp_path / "tree"
    try:
        tree.symlink_to(demo_project, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported?")

    project = Project.from_path(tree)
    assert project is not None
    path_cache = PathCache(project.make_env(load_plugins=False))

    assert path_cache.to_source_id(tree / "filename") == "filename"


################################################################


def test_Artifact_open_encoding(dummy_artifact_txn):
    with dummy_artifact_txn:
        with dummy_artifact_txn.open("w", encoding="iso-8859-1") as fp:
            fp.write("Ciarán")

    content = Path(dummy_artifact_txn.dst_filename).read_bytes()
    assert content == b"Ciar\xe1n"


def test_FileInfo_unchanged(env, tmp_path):
    file_path = tmp_path / "file"
    file_path.write_text("foo")

    file_info = FileInfo(env, file_path)
    # cache size, mtime, but *not* checksum
    assert file_info.size == 3

    file_path.write_text("foobar")
    file_info2 = FileInfo(env, file_path)
    assert file_info2.size != 3

    assert not file_info.unchanged(file_info2)


@pytest.mark.parametrize(
    "info1, info2, expected",
    [
        ((1, 1, "sum", True), (1, 1, "sum", True), True),
        ((1, 1, "sum", True), (1, 1, "sum", False), False),
        ((1, 1, "sum", True), (1, 2, "sum", True), False),
        ((1, 1, "sum", True), (2, 1, "sum", True), False),
        ((1, 1, "sum", True), (1, 1, "XXX", True), False),
        # check that checksum is not consulted for non-directories
        ((1, 1, "sum", False), (1, 1, "XXX", False), True),
    ],
)
def test_FileInfo_unchanged_cases(env, tmp_path, info1, info2, expected):
    file_info1 = FileInfo(env, tmp_path, *info1)
    file_info2 = FileInfo(env, tmp_path, *info2)
    assert file_info1.unchanged(file_info2) is expected
    assert file_info2.unchanged(file_info1) is expected


def test_FileInfo_checksum_ignores_uninteresting_files(env, tmp_path):
    checksum1 = FileInfo(env, tmp_path).checksum

    tmp_path.joinpath("file.txt").touch()

    checksum2 = FileInfo(env, tmp_path).checksum
    assert checksum2 != checksum1

    tmp_path.joinpath(".hidden.txt").touch()

    checksum3 = FileInfo(env, tmp_path).checksum
    assert checksum3 == checksum2


def test_FileInfo_checksum_notices_record_dirs(env, tmp_path):
    checksum1 = FileInfo(env, tmp_path).checksum

    page_dir = tmp_path.joinpath("page")
    page_dir.mkdir()

    checksum2 = FileInfo(env, tmp_path).checksum
    assert checksum2 != checksum1

    page_dir.joinpath("contents.lr").touch()
    checksum3 = FileInfo(env, tmp_path).checksum
    assert checksum3 != checksum2


def test_FileInfo_precached_values(env, tmp_path):
    file_info = FileInfo(
        env, tmp_path, mtime=1, size=2, checksum="cheese", is_dir=False
    )
    assert file_info.checksum == "cheese"
    assert file_info.mtime == 1
    assert file_info.size == 2
    assert not file_info.is_dir

    assert FileInfo(env, tmp_path).is_dir  # sanity check


def test_FileInfo_raises_type_error(env, tmp_path):
    file_info = FileInfo(env, tmp_path)
    with pytest.raises(TypeError, match="must be a FileInfo"):
        file_info.unchanged(tmp_path)


def test_build_state_reference_cycle(pad, builder):
    # Check that there are no reference cycles preventing build_state from
    # being immediately collected.
    collected = mock.Mock(spec=(), name="collected")

    prog, build_state = builder.build(pad.root)
    weakref.finalize(build_state, collected)
    del prog, build_state
    assert collected.call_count == 1


@pytest.mark.parametrize(
    "path, alt",
    [
        ("/@vtag/foo", "de"),
        ("/path@vtag", None),
    ],
)
def test_pack_virtual_source_path(path, alt):
    packed = _pack_virtual_source_path(path, alt)
    assert _unpack_virtual_source_path(packed) == (path, alt)
    assert _is_packed_virtual_source_path(packed)


def test_unpack_virtual_source_path_returns_none_for_primary_alt() -> None:
    path = "/path@virt"
    assert _is_packed_virtual_source_path(path)
    assert _unpack_virtual_source_path(path) == (path, None)


def test_unpack_virtual_source_path_raises() -> None:
    path = "/blog/post1/"
    with pytest.raises(ValueError, match=r"must include at least one .@"):
        assert _unpack_virtual_source_path(path)  # type: ignore[arg-type]
