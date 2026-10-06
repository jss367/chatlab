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
#
# CHATLAB_CODESIGN_IDENTITY takes a certificate name or its SHA-1 fingerprint.
# Signing is always by fingerprint, so a name two certificates share (an old
# "ChatLab Local" left behind by a new one) stops the build rather than
# leaving codesign to refuse or guess. Only the first section of the listing
# is read: a trusted certificate appears again under "Valid identities only".
CODESIGN_IDENTITY=${CHATLAB_CODESIGN_IDENTITY:-"ChatLab Local"}
MATCHES=$(security find-identity -p codesigning | sed '/Valid identities only/,$d' | awk -v id="$CODESIGN_IDENTITY" '
    $1 ~ /^[0-9]+\)$/ {
        name = $0
        sub(/^[^"]*"/, "", name)
        sub(/".*$/, "", name)
        if (toupper($2) == toupper(id) || name == id) print $2
    }' | sort -u)
COUNT=$(printf '%s' "$MATCHES" | grep -c . || true)
if [ "$COUNT" -eq 1 ]; then
    codesign --force --timestamp=none --sign "$MATCHES" "$APP"
    codesign --verify --strict "$APP"
    printf 'Signed with "%s" (%s)\n' "$CODESIGN_IDENTITY" "$MATCHES"
elif [ "$COUNT" -gt 1 ]; then
    printf 'Several code signing identities match "%s":\n%s\n' "$CODESIGN_IDENTITY" "$MATCHES" >&2
    printf 'Delete the stale one, or set CHATLAB_CODESIGN_IDENTITY to one fingerprint.\n' >&2
    exit 1
elif [ -n "${CHATLAB_CODESIGN_IDENTITY:-}" ]; then
    printf 'No code signing identity "%s" in the keychain.\n' "$CODESIGN_IDENTITY" >&2
    exit 1
else
    printf 'No "%s" certificate; the bundle keeps its ad-hoc signature.\n' "$CODESIGN_IDENTITY"
fi

printf 'Built %s\n' "$APP"
