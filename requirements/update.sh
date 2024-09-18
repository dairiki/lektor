#!/bin/bash

set -eEu -o pipefail
here="$(dirname "$0")"
python_targets=(">=3.12" "~="3.{11,10,9,8}.0)

run () {
    echo "$*"
    "$@"
}

create_lockfile () {
    local lockfile="$1"
    shift

    run pdm lock -L "$lockfile" "$@" --python "${python_targets[0]}"
    for python in "${python_targets[@]:1}"; do
        run pdm lock -L "$lockfile" --python "$python" --append
    done
}

update_lockfile () {
    local lockfile="$1"

    if [[ -f "$lockfile" ]]; then
        run pdm lock -L "$lockfile"
    else
        create_lockfile "$@"
    fi
}


update_lockfile "${here}/tests.lock" -G tests
update_lockfile "${here}/old-deps.lock" -G tests -S direct_minimal_versions
