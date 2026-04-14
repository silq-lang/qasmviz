#!/usr/bin/env bash
set -euo pipefail

cat > "qasmviz.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
DIR=$(printf '%q' "$PWD")
. "\$DIR/.venv/bin/activate"
exec env PYTHONPATH="\$DIR" python "\$DIR/qasmviz.py" "\$@"
EOF

chmod +x qasmviz.sh
