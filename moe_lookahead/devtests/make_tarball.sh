#!/usr/bin/env bash
# Build ../moe_lookahead.tar.gz from this project, without data, caches, checkpoints, logs or results.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$HERE")"
NAME="$(basename "$HERE")"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -C "$PARENT" -czf "$PARENT/$NAME.tar.gz" \
  --exclude="$NAME/data" --exclude="$NAME/cache" --exclude="$NAME/ckpt" --exclude="$NAME/logs" \
  --exclude="$NAME/results" --exclude="$NAME/smoke" --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.DS_Store' --exclude='._*' --exclude="$NAME/setup_login.log" \
  "$NAME"
echo "wrote $PARENT/$NAME.tar.gz ($(du -h "$PARENT/$NAME.tar.gz" | cut -f1))"
tar -tzf "$PARENT/$NAME.tar.gz" | sort
