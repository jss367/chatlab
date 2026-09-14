#!/bin/bash
# release_harness.sh [path-to-release.sh]
#
# Exercises scripts/release.sh end to end against a local bare repository
# standing in for GitHub, with gh, the build and the test run stubbed out.
# Nothing here touches the real repository, the real remote, or the
# installed app. tests/test_release_script.py runs this on macOS.
set -uo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
real=${1:-"$here/../scripts/release.sh"}
[ -x "$real" ] || { echo "no release.sh at $real" >&2; exit 2; }
root=$(mktemp -d /tmp/relharness.XXXX)
export PATH="$root/bin:$PATH"
export RELHARNESS_STATE="$root/releases.txt"
export CHATLAB_RELEASE_REPO=fake/repo
mkdir -p "$root/bin"

cat > "$root/bin/gh" <<'GH'
#!/bin/bash
state="$RELHARNESS_STATE"
case "$1 $2" in
  "auth status") exit 0 ;;
  "release view")
      if grep -qx "$3" "$state"; then echo "{\"tagName\":\"$3\"}"; exit 0; fi
      echo "release not found" >&2; exit 1 ;;
  "release list") tail -1 "$state" ;;
  "release create")
      echo "$3" >> "$state"
      echo "fake gh: published $3 (${*:4})" >&2
      echo "https://example.invalid/releases/tag/$3" ;;
  *) echo "fake gh: unhandled: $*" >&2; exit 1 ;;
esac
GH
printf '#!/bin/sh\nexit 0\n' > "$root/bin/open"
chmod +x "$root/bin/gh" "$root/bin/open"

git init -q --bare "$root/origin.git"
git clone -q "$root/origin.git" "$root/work" 2>/dev/null
cd "$root/work"
git config user.email t@example.invalid; git config user.name Test
mkdir -p scripts tests .desktop-venv/bin
printf '__version__ = "0.15.0"\nBUNDLE_IDENTIFIER = "build.chatlab.app"\n' > version.py
cp "$real" scripts/release.sh; chmod +x scripts/release.sh
cat > scripts/build_macos_app.sh <<'B'
#!/bin/sh
set -eu
version=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' version.py)
rm -rf dist/ChatLab.app
mkdir -p dist/ChatLab.app/Contents/MacOS
cat > dist/ChatLab.app/Contents/Info.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleShortVersionString</key><string>$version</string>
<key>CFBundleExecutable</key><string>ChatLab</string>
<key>CFBundleName</key><string>ChatLab</string>
</dict></plist>
PLIST
printf '#!/bin/sh\nexit 0\n' > dist/ChatLab.app/Contents/MacOS/ChatLab
chmod +x dist/ChatLab.app/Contents/MacOS/ChatLab
echo "stub build: $version"
B
chmod +x scripts/build_macos_app.sh
printf '#!/bin/sh\nexit 0\n' > .desktop-venv/bin/python
chmod +x .desktop-venv/bin/python
printf 'dist/\n.desktop-venv/\n' > .gitignore
git add -A >/dev/null && git commit -qm base && git push -q origin HEAD:main
echo "v0.15.0" > "$RELHARNESS_STATE"

pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: expected [$3], got [$2]"; fail=$((fail+1)); fi; }
run() { ./scripts/release.sh --skip-install "$@" > "$root/out.txt" 2> "$root/err.txt"; echo $?; }

echo "== 1. dirty tree aborts =="
echo x >> version.py
code=$(run); check "exit 1" "$code" 1
check "message" "$(grep -c 'working tree has changes' "$root/err.txt")" 1
git checkout -- version.py

echo "== 2. normal release bumps the minor version =="
code=$(run); check "exit 0" "$code" 0
git fetch -q origin main
check "version on main" "$(git show origin/main:version.py | sed -n 's/^__version__ = "\(.*\)"$/\1/p')" 0.16.0
check "commit subject" "$(git log -1 --format=%s origin/main)" "Release ChatLab 0.16.0"
check "tag on remote" "$(git ls-remote --tags "$root/origin.git" 'v0.16.0^{}' | wc -l | tr -d ' ')" 1
check "tag points at main" "$(git ls-remote --tags "$root/origin.git" 'v0.16.0^{}' | awk '{print $1}')" "$(git rev-parse origin/main)"
check "release published" "$(tail -1 "$RELHARNESS_STATE")" v0.16.0
check "asset uploaded" "$(grep -c 'ChatLab-macos-arm64.zip' "$root/err.txt")" 1

echo "== 3. rerun bumps again from the published release =="
code=$(run); check "exit 0" "$code" 0
git fetch -q origin main
check "version on main" "$(git show origin/main:version.py | sed -n 's/^__version__ = "\(.*\)"$/\1/p')" 0.17.0

echo "== 4. an interrupted release is finished, not bumped past =="
# main carries 0.18.0 with no tag and no release, as if a build had failed
git checkout -q --detach origin/main
sed -i '' 's/^__version__ = .*/__version__ = "0.18.0"/' version.py
git commit -qam "Release ChatLab 0.18.0" && git push -q origin HEAD:main
before=$(git rev-parse HEAD)
code=$(run); check "exit 0" "$code" 0
git fetch -q origin main
check "no extra commit" "$(git rev-parse origin/main)" "$before"
check "released 0.18.0" "$(tail -1 "$RELHARNESS_STATE")" v0.18.0
check "no 0.19.0" "$(grep -c v0.19.0 "$RELHARNESS_STATE")" 0

echo "== 5. an already published --version aborts =="
code=$(run --version 0.18.0); check "exit 1" "$code" 1
check "message" "$(grep -c 'already published' "$root/err.txt")" 1

echo "== 6. a local commit that is not on main aborts =="
git checkout -q --detach origin/main
echo "# local" >> version.py; git commit -qam "local work"
code=$(run); check "exit 1" "$code" 1
check "message" "$(grep -c 'not on origin/main' "$root/err.txt")" 1

echo "== 7. a push that loses a race rebases and retries =="
git checkout -q --detach origin/main
git clone -q "$root/origin.git" "$root/other" 2>/dev/null
git -C "$root/other" config user.email o@example.invalid
git -C "$root/other" config user.name Other
mkdir -p .git/hooks
cat > .git/hooks/pre-push <<'HOOK'
#!/bin/sh
# Once, land a competing commit on main just before our push reaches it.
[ -e "$RELHARNESS_RACE" ] && exit 0
touch "$RELHARNESS_RACE"
echo "raced" >> "$RELHARNESS_OTHER/README.md"
git -C "$RELHARNESS_OTHER" add -A
git -C "$RELHARNESS_OTHER" commit -qm "Competing commit"
git -C "$RELHARNESS_OTHER" push -q origin HEAD:main
exit 0
HOOK
chmod +x .git/hooks/pre-push
export RELHARNESS_RACE="$root/raced" RELHARNESS_OTHER="$root/other"
before_race=$(git rev-parse origin/main)
code=$(run); check "exit 0" "$code" 0
check "retried once" "$(grep -c 'rebasing onto it' "$root/out.txt")" 1
git fetch -q origin main
check "competing commit kept" "$(git log origin/main --format=%s | grep -c 'Competing commit')" 1
check "version commit on top" "$(git log -1 --format=%s origin/main)" "Release ChatLab 0.19.0"
check "tag follows the rebase" "$(git ls-remote --tags "$root/origin.git" 'v0.19.0^{}' | awk '{print $1}')" "$(git rev-parse origin/main)"
rm -f .git/hooks/pre-push

echo
echo "$pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
    rm -rf "$root"
else
    echo "the scratch repository is left at $root"
fi
[ "$fail" -eq 0 ]
