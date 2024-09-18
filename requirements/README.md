# Test Requirements

This directory contains PDM lock files used for running tests. These
are used by our tox configuration.

The script in `update.sh` may be used to update (or regenerate) these lock files.

## tests.lock

This is a [multi-target lock file], with separate locks run for each
minor python version. This is done to ensure that each python version
gets the most recent compatible version of each dependency.

It includes the _default_ and `tests` groups of dependencies. (But _not_ the `dev` group.)

[multi-target lock file]: https://pdm-project.org/latest/usage/lock-targets/#separate-lock-files-or-merge-into-one

## old-deps.lock

This is the same a tests.lock, except that PDM's `[direct_minimal_versions]` lock strategy is used.
This means it uses the oldest compatible version of each direct dependency.

This is used in an attempt to make sure that our lower-bound pins are actually correct.

[direct_miminal_versions]: https://pdm-project.org/latest/usage/lockfile/#direct-minimal-versions
