## Ideas

### Track Dirty Artifacts rather than Dirty Sources

The point of `set/clear_dirty_flag()` and the `dirty_sources` table is
to force artifacts to be rebuilt, even if their dependencies are
up-to-date.

Currently, they work by marking all of the source files for the artifact
as "dirty".

Why not just keep a table of "dirty" artifacts?

### Normalize the `source_info` table

The `source_info` table should be normalize from to split out (`lang`, `title`) from the rest of the data.
(There can be multiple lang-title pairs for each source Record.)

Also, we track _a_ source file per `source_info` record in order to be able to prune stale `source_info` records.
It might work better to track the SourceObject (I think the SourceObject for `source_info`s is always a `Record`).
Then prune `source_info`s for non-existent source objects.

### Pruning

Currently, Lektor prunes artifacts that do not have an extant primary source file.

For artifacts from virtual sources, this can be problematic. Some virtual sources do
not really depend on any source file. Currently, however they have to declare a bogus
dependency on some source file or their artifact will be pruned immediately.

The criteria could be changed:

Track the SourceObject that generated each artifact.

Every time a source object is build, we remember what artifacts and sub-artifacts it
produces. (Even if those artifacts were not actually rebuilt due to unchanged dependencies.)

After a build_all, we can delete any memory of artifacts associated with non-existing
SourceObjects. (The `build_all` involves iterating over all SourceObjects, so this
is cheap.)

Now we can prune those artifacts that don't have a recorded source object.

## TODOs

### lektor.builder, multithreading

- [Done] Don't build a source object more than once per build_all
- [Done] Maybe don't even attempt to build a given artifact more than once per build_all.
- I'm not maxing out CPU usage. Figure out what needs to change for that to happened.

- The SqliteConnectionPool needs attention.
  - One connection per asyncio task, as well as one per thread?
  - Make sure the connections are being closed properly on thread/task termination

### Type annotations

- rebase them back earlier
- make a PR

### Docstrings

Improve docstrings on the various parts of the build system.

Note lifetime is the build of a single artifact.

====

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
- Removed `Context.exc_info`. Now we just raise the exception and catch it where desired.

### db.Record

- Added a (pure virtual) `Record.parent` attribute.
  As a `DBSourceObject`, all records should have a `parent` attribute.

### lektor.environment

- Deleted `lektor.environment.any_fnmatch`. (inlined)
