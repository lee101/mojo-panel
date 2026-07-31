#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
mojo build --emit shared-lib "$repo_dir/src/capi.mojo" -o "$repo_dir/dist/libmojo-panel.so"

if mojo build --emit shared-lib "$repo_dir/src/gpu.mojo" \
     -o "$repo_dir/dist/libmojo-panel-gpu.so" 2>"$repo_dir/dist/gpu-build.log"; then
  echo "built dist/libmojo-panel-gpu.so"
else
  echo "GPU kernels not built (see dist/gpu-build.log); CPU path unaffected" >&2
  rm -f "$repo_dir/dist/libmojo-panel-gpu.so"
fi
