#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s OUTPUT_DIRECTORY [LOCAL_WORK_DIRECTORY]\n' "${0##*/}"
    printf '       %s OUTPUT_DIRECTORY LOCAL_WORK_DIRECTORY --detector-size PIXELS [MATRIX_OPTIONS]\n' "${0##*/}"
    printf 'Run the GPFS benchmark with detector staging on node-local scratch.\n'
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 1 )) || { (( $# > 2 )) && [[ $3 != --detector-size && $3 != --detector-size=* ]]; }; then
    usage >&2
    exit 2
fi

output=$(realpath -m -- "$1")
if [[ -e $1 || -L $1 ]]; then
    printf 'Choose a new output directory. Already exists: %s\n' "$1" >&2
    exit 2
fi

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
work_dir=${2:-${SLURM_TMPDIR:-/tmp/$(id -un)}/voxel-io}
(umask 077; mkdir -p -- "$work_dir")
work_dir=$(cd -- "$work_dir" && pwd -P)
scratch="$work_dir/scratch"
mkdir -p -- "$scratch"

export XDG_CACHE_HOME="$work_dir/cache"
export XDG_CONFIG_HOME="$work_dir/config"
export XDG_DATA_HOME="$work_dir/share"
export MPLBACKEND=Agg

tango_deploy="/tmp/$(id -un)/tango-ucx"
export PIXI_HOME="$work_dir/pixi"
export PIXI_CACHE_DIR="$XDG_CACHE_HOME/pixi"
if [[ -x $tango_deploy/pixi/bin/pixi ]]; then
    export PIXI_HOME="$tango_deploy/pixi"
    export PIXI_CACHE_DIR="$tango_deploy/cache/pixi"
    pixi="$PIXI_HOME/bin/pixi"
elif [[ -x $PIXI_HOME/bin/pixi ]]; then
    pixi="$PIXI_HOME/bin/pixi"
elif ! pixi=$(command -v pixi); then
    if [[ $(uname -m) != x86_64 ]]; then
        printf 'The locked environment requires a Linux x86_64 node.\n' >&2
        exit 2
    fi
    pixi_version=0.83.0
    pixi="$PIXI_HOME/bin/pixi"
    mkdir -p -- "$PIXI_HOME/bin"
    download_dir=$(mktemp -d "$PIXI_HOME/download.XXXXXXXX")
    trap 'rm -rf -- "$download_dir"' EXIT
    curl -fsSL --retry 3 \
        "https://github.com/prefix-dev/pixi/releases/download/v$pixi_version/pixi-x86_64-unknown-linux-musl.tar.gz" \
        -o "$download_dir/pixi.tar.gz"
    tar -xzf "$download_dir/pixi.tar.gz" -C "$download_dir" pixi
    mv -f -- "$download_dir/pixi" "$pixi"
fi

cd -- "$source_dir"
printf 'Compute node: %s\nPixi: %s\nLocal scratch: %s\nBenchmark output: %s\n' \
    "$(hostname)" "$pixi" "$scratch" "$output"
"$pixi" install --locked
if (( $# > 2 )); then
    "$pixi" run --locked write-matrix "$output" --scratch "$scratch" "${@:3}"
else
    "$pixi" run --locked experiment "$output" --profile gpfs --scratch "$scratch"
fi
