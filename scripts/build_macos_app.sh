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

APP="$SCRIPT_DIR/dist/ChatLab.app"

# macOS keys the Local Network permission to the bundle's designated
# requirement. An ad-hoc signature's requirement is its hash, so every build
# counts as a new app and the permission granted to the last one lapses. A
# certificate's requirement names the certificate and survives rebuilds, and
# a self-signed one is enough. Only the outer bundle is re-signed: given an
# identity, PyInstaller would also turn on the hardened runtime, which needs
# entitlements this bundle does not carry.
CODESIGN_IDENTITY=${CHATLAB_CODESIGN_IDENTITY:-"ChatLab Local"}
if security find-identity -p codesigning | grep -Fq "\"$CODESIGN_IDENTITY\""; then
    codesign --force --timestamp=none --sign "$CODESIGN_IDENTITY" "$APP"
    codesign --verify --strict "$APP"
    printf 'Signed with "%s"\n' "$CODESIGN_IDENTITY"
elif [ -n "${CHATLAB_CODESIGN_IDENTITY:-}" ]; then
    printf 'No code signing identity named "%s" in the keychain.\n' "$CODESIGN_IDENTITY" >&2
    exit 1
else
    printf 'No "%s" certificate; the bundle keeps its ad-hoc signature.\n' "$CODESIGN_IDENTITY"
fi

printf 'Built %s\n' "$APP"
