#!/usr/bin/env bash
set -euo pipefail

script_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"

src="$script_dir/qasmviz.sh"
dst="/usr/local/bin/qasmviz"

if [[ ! -f "$src" ]]; then
    "$script_dir/make_qasmviz.sh"
fi

"$script_dir/setup-venv.sh"

if [[ ! -f "$src" ]]; then
  echo "Error: source script not found: $src" >&2
  exit 1
fi

if [[ -L "$dst" ]]; then
  sudo rm -- "$dst"
elif [[ -e "$dst" ]]; then
  echo "Error: $dst exists and is not a symlink; refusing to replace it." >&2
  exit 1
fi

sudo ln -s -- "$src" "$dst"
echo "Installed $dst -> $src"
