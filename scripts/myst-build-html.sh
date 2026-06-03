#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shim_dir="${TMPDIR:-/tmp}/myst-npm-shim"
real_npm="$(command -v npm || true)"
npm_version="$(npm -v 2>/dev/null || true)"

if [ -z "$real_npm" ]; then
  echo "npm bulunamadı" >&2
  exit 1
fi

if [ -z "$npm_version" ]; then
  npm_version="11.11.0"
fi

mkdir -p "$shim_dir"
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' 'if [ "${1:-}" = "--version" ]; then'
  printf '  printf '"'"'%%s\\n'"'"' %q\n' "$npm_version"
  printf '%s\n' '  exit 0'
  printf '%s\n' 'fi'
  printf 'exec %q "$@"\n' "$real_npm"
} >"$shim_dir/npm"
chmod +x "$shim_dir/npm"

cd "$repo_root/thesis"
NODE_OPTIONS="--require $repo_root/scripts/myst-node-preload.cjs ${NODE_OPTIONS:-}" \
  PATH="$shim_dir:$PATH" \
  myst build --site "$@"

node "$repo_root/scripts/render-myst-html.cjs"
