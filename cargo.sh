#!/bin/sh
# cargo with everything kept inside this folder: its package cache, and the cmake and nasm that compiling the bundled JPEG decoder needs.
# Use it as cargo: ./cargo.sh run --release
here=$(cd "$(dirname "$0")" && pwd)
export CARGO_HOME="$here/.cargo-home"
export PATH="$here/.venv/bin:$here/.tools/nasm/usr/bin:$PATH"
exec cargo "$@"
