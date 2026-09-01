#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DESKTOP_VENV=${CHATLAB_DESKTOP_VENV:-"$SCRIPT_DIR/.desktop-venv"}

if [ ! -x "$DESKTOP_VENV/bin/python" ]; then
    python3 -m venv "$DESKTOP_VENV"
fi

"$DESKTOP_VENV/bin/python" -m pip install --upgrade pip
"$DESKTOP_VENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-desktop.txt"
"$DESKTOP_VENV/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --distpath "$SCRIPT_DIR/dist" \
    --workpath "$SCRIPT_DIR/build" \
    "$SCRIPT_DIR/ChatLab.spec"

printf 'Built %s\n' "$SCRIPT_DIR/dist/ChatLab.app"
