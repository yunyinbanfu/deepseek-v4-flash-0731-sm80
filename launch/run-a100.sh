#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias kept for old notes. The maintained source-tree launcher is
# run-pp-dspark.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run-pp-dspark.sh" "$@"
