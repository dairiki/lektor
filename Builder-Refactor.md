## Free-Threading Issues

### Places to check/fix thread safety

- Our jinja code
- Our markdown/mistune code

### Extension libraries

Lektor uses the following non-pure-python libraries.

- watchfiles — seems to segfault on free-threaded python
  The segfault happens with either `PYTHON_GIL=0` or `PYTHON_GIL=1`,
  but does not happen with the regular non-free-threading build
  of python 3.13.0rc1.

- markupsafe — see
  [#460](https://github.com/pallets/markupsafe/issues/460),
  [#462](https://github.com/pallets/markupsafe/pull/462)

- Pillow — [#8199](https://github.com/python-pillow/Pillow/issues/8199)

  Free-threading work in master but not released?

  Free-threading builds may be available from:
  https://anaconda.org/scientific-python-nightly-wheels/pillow

## API Changes

### builder.BuildState

- Removed `BuildState.connect_to_database()`

- Added (optional) `parent_artifact` parameter to `BuildState.new_artifact`

  This is used for declaring sub-artifacts. The `parent_artifact` should reference
  the artifact that generated the sub-artifact.

- Deprecated `BuildState.artifact_exists()`
- Deprecated `BuildState.get_artifact_dependency_infos()`
- Deprecated `BuildState.iter_artifacts()`

- Changed parameters of `BuildState.check_artifact_is_current()`
  from `(artifact_name, sources, config_hash)` to `(artifact)`.

### builder.FileInfo

- Deprecated `FileInfo.filename_and_checksum` (should just remove?)

### builder.Artifact

- Removed the `build_state` attribute and constructor parameter.
- Added the `parent` attribute and constructor parameter.
- Removed all methods. They've mostly been moved to the new
  `ArtifactTransaction` class.

### builder.ArtifactTransaction

This new method contains all the methods used to create/modify an artifact.

It used to be that artifact _build progs_ were passed an `Artifact` instance. Now
they are passed a `ArtifactTransaction` instance.

Methods moved from `Artifact`:

- `ArtifactTransaction.clear_dirty_flag()`
- `ArtifactTransaction.set_dirty_flag()`
- `ArtifactTransaction.ensure_dir()`
- `ArtifactTransaction.open()`
- `ArtifactTransaction.replace_with_file()`
- `ArtifactTransaction.render_template_into()`

Methods removed from `Artifact` that were not preserved:

- the `Artifact.is_current` property
- `Artifact.get_dependency_infos()`

### builder.Builder

The Builder has been extensively refactored to work towards supporting
multi-threaded builds.

- `Builder.build` used to return a `(prog, build_state)` tuple. This
  has been upgraded to a `NamedTuple` which has two additional
  properties: `failures` and `primary_artifact`.
- Removed `Builder.connect_to_database()`. This has been replaced by a
  `Builder.build_db` attribute.
- Removed `Builder.get_build_program()`.
  (If needed this should be part of `Environment` anyway?)

- Removed `Builder.build_artifact()`
- Removed `Builder.update_source_info()`
- Removed `Builder.get_initial_build_queue()`
- Removed `Builder.extend_build_queue()`

### build_programs.BuildProgram

- Added `BuildProgram.get_artifacts()`.

  This replaces `BuildProgram.produce_artifacts()` and `BuildProgram.artifacts` as the
  primary extension point for declaring what artifact(s) are to be produced.

- Removed `BuildProgram.build()` (internal, non-API)

### lektor.compat

- Added a (not fully functional) polyfill for `itertools.batched`
  (which is only available in py312+).

### lektor.context.Context

- Removed the ability to construct a Context without a currently-being-built artifact.

  That seemed to be used only for testing, and as a hacky way to
  temporarily disable dependency collection.

  Now the constructor takes a single parameter of type `ArtifactTransaction`.

  Note that the lifetime of the `Context` has always been the construction of a single artifact.

- Deprecated the `Context.artifact` attribute.
- Added a `Context.artifact_txn` attribute

- `Context.referenced_dependencies` has been moved to
  `ArtifactTransaction._referenced_source_files`.

- `Context.referenced_virtual_dependencies` has been moved to
  `ArtifactTransaction._referenced_virtual_sources`.
- Added a `disable_dependency_tracking()` context manager to temporarily
  disable dependency tracking. (This was copied from `lektorlib`.)

### db.Record

- Added a (pure virtual) `Record.parent` attribute.
  As a `DBSourceObject`, all records should have a `parent` attribute.

### lektor.environment

- Deleted `lektor.environment.any_fnmatch`. (inlined)

## BuildState functionality should be split up

However note that BuildState is part of the "before-build", "after-build" plugin API,
so we will need to provide a proxy or some such for that?
(Plugins that appear to use build_state: lektor-minify, lektor-diazotheme, lektor-amp, lektor-groupby.)

- Database actions
  (Need one connection per thread/context. Lifetime is arbitrary (data long-lived on disk.))

- Project source path handling and caching (PathCache)
  (Lifetime: one upper-level Builder method call, e.g. buildall)

- State of build. Current lifetime of BuildState is the building of
  one source record (i.e. the execution of one BuildFunction). The
  BuildState appears to be used to track number of successful and
  failed artifact builds within that process.

  (The need for this function of the BuildState could probably be
  eliminated or rolled into BuildProgram.build() or something.)

## Context

Note lifetime is the build of a single artifact.

### [DONE] Don't store \_exc_info on context.

Just keep raising exception, catching it where needed to mark failiure.

Perhaps raise a custom exception.

    raise BuildFailed(...) from exc

### [DONE] Refactor the BuildProgram API

Once build_func is part of Artifact, we can replace the
`produce_artifacts`, `declare_artifact`, and `build_artifact` API
methods with just `get_artifacts`:

    def get_artifacts(self) -> Iterable[Artifact]: ...
    """Return an iterable of artifacts to be build for this record.

    Note that each artifact, when built, may produce additional sub-artifacts which
    also are to be built as part of the build process for this record.
    """

## Comments

Improve docstrings on the various parts of the build system.
