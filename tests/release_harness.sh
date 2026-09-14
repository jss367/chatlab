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
# Each line of the state file is "<tag> <missing|draft|incomplete|published>".
# release view prints the status directly: the script's own jq is exercised
# against the real GitHub API, not here.
state="$RELHARNESS_STATE"
status_of() { awk -v t="$1" '$1==t{s=$2} END{print s}' "$state"; }
case "$1 $2" in
  "auth status") exit 0 ;;
  "release view")
      st=$(status_of "$3")
      [ -n "$st" ] || { echo "release not found" >&2; exit 1; }
      echo "$st" ;;
  "release list") awk '$2=="published"{t=$1} END{print t}' "$state" ;;
  "release create")
      echo "$3 published" >> "$state"
      echo "fake gh: created $3 (${*:4})" >&2
      echo "https://example.invalid/releases/tag/$3" ;;
  "release upload") echo "fake gh: uploaded to $3 (${*:4})" >&2 ;;
  "release edit")
      awk -v t="$3" '$1==t{$2="published"} {print}' "$state" > "$state.tmp"
      mv "$state.tmp" "$state"
      echo "fake gh: published $3" >&2 ;;
  *) echo "fake gh: unhandled: $*" >&2; exit 1 ;;
esac
GH
# Launch Services and the process table, enough for the install step.
cat > "$root/bin/open" <<'OPEN'
#!/bin/sh
[ -n "${RELHARNESS_OPEN_FAILS:-}" ] && exit 1
touch "$RELHARNESS_RUNNING"
exit 0
OPEN
cat > "$root/bin/pgrep" <<'PGREP'
#!/bin/sh
[ -e "$RELHARNESS_RUNNING" ]
PGREP
cat > "$root/bin/osascript" <<'OSA'
#!/bin/sh
rm -f "$RELHARNESS_RUNNING"
OSA
cat > "$root/bin/pkill" <<'PKILL'
#!/bin/sh
rm -f "$RELHARNESS_RUNNING"
PKILL
chmod +x "$root/bin/gh" "$root/bin/open" "$root/bin/pgrep" "$root/bin/osascript" "$root/bin/pkill"
export RELHARNESS_RUNNING="$root/app-running"

git init -q --bare "$root/origin.git"
# The default branch of a fresh repository is master on an unconfigured Git,
# and every clone below wants main.
git -C "$root/origin.git" symbolic-ref HEAD refs/heads/main
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
echo "v0.15.0 published" > "$RELHARNESS_STATE"

pass=0; fail=0
last_release() { awk 'END{print $1}' "$RELHARNESS_STATE"; }
status_of() { awk -v t="$1" '$1==t{s=$2} END{print s}' "$RELHARNESS_STATE"; }
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
check "release published" "$(last_release)" v0.16.0
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
check "released 0.18.0" "$(last_release)" v0.18.0
check "no 0.19.0" "$(grep -c v0.19.0 "$RELHARNESS_STATE")" 0

echo "== 5. a published version that is not the one on main aborts =="
code=$(run --version 0.17.0); check "exit 1" "$code" 1
check "message" "$(grep -c 'already published' "$root/err.txt")" 1

echo "== 6. a local commit that is not on main aborts =="
git checkout -q --detach origin/main
echo "# local" >> version.py; git commit -qam "local work"
code=$(run); check "exit 1" "$code" 1
check "message" "$(grep -c 'not on origin/main' "$root/err.txt")" 1

echo "== 7. a push that loses a race rebases and retries =="
git checkout -q --detach origin/main
git clone -q --branch main "$root/origin.git" "$root/other" 2>/dev/null
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

echo "== 8. a lightweight tag from an earlier hand-made release is accepted =="
git checkout -q --detach origin/main
sed -i '' 's/^__version__ = .*/__version__ = "0.20.0"/' version.py
git commit -qam "Release ChatLab 0.20.0" && git push -q origin HEAD:main
git tag v0.20.0 && git push -q origin v0.20.0
before=$(git rev-parse HEAD)
code=$(run); check "exit 0" "$code" 0
git fetch -q origin main
check "no extra commit" "$(git rev-parse origin/main)" "$before"
check "released 0.20.0" "$(last_release)" v0.20.0

echo "== 9. a version the updater could not read is refused =="
code=$(run --version 1.2.3beta); check "exit 1" "$code" 1
check "message" "$(grep -c 'Not a MAJOR.MINOR.PATCH' "$root/err.txt")" 1

echo "== 10. a run interrupted after publishing installs without cutting another =="
before=$(git rev-parse origin/main)
code=$(run --version 0.20.0); check "exit 0" "$code" 0
check "said so" "$(grep -c 'published already' "$root/out.txt")" 1
git fetch -q origin main
check "no extra commit" "$(git rev-parse origin/main)" "$before"
check "no second release" "$(grep -c '^v0.20.0 ' "$RELHARNESS_STATE")" 1
check "no 0.21.0" "$(grep -c v0.21.0 "$RELHARNESS_STATE")" 0

echo "== 11. an older published version is still refused =="
code=$(run --version 0.19.0); check "exit 1" "$code" 1
check "message names the one on main" "$(grep -c 'pass --version 0.20.0' "$root/err.txt")" 1

echo "== 12. the tests run in the environment the build used =="
mkdir -p "$root/altvenv/bin"
printf '#!/bin/sh\ntouch "$RELHARNESS_ALT_MARKER"\nexit 0\n' > "$root/altvenv/bin/python"
chmod +x "$root/altvenv/bin/python"
export RELHARNESS_ALT_MARKER="$root/altvenv-ran"
code=$(CHATLAB_DESKTOP_VENV="$root/altvenv" ./scripts/release.sh --skip-install --version 0.20.0 \
    > "$root/out.txt" 2> "$root/err.txt"; echo $?)
check "exit 0" "$code" 0
check "the override ran the tests" "$([ -e "$root/altvenv-ran" ] && echo yes || echo no)" yes

release_from_main() {  # <version>: land it on main as an interrupted run would
    git checkout -q --detach origin/main
    sed -i '' "s/^__version__ = .*/__version__ = \"$1\"/" version.py
    git commit -qam "Release ChatLab $1"
    git push -q origin HEAD:main
}

echo "== 13. a draft left by a failed publish is finished, not bumped past =="
release_from_main 0.21.0
git tag -a v0.21.0 -m "ChatLab 0.21.0" && git push -q origin v0.21.0
echo "v0.21.0 draft" >> "$RELHARNESS_STATE"
code=$(run); check "exit 0" "$code" 0
check "said so" "$(grep -c 'exists as draft' "$root/out.txt")" 1
check "now published" "$(status_of v0.21.0)" published
check "no 0.22.0" "$(grep -c v0.22.0 "$RELHARNESS_STATE")" 0

echo "== 14. a release whose uploads did not finish gets its assets =="
release_from_main 0.22.0
echo "v0.22.0 incomplete" >> "$RELHARNESS_STATE"
code=$(run); check "exit 0" "$code" 0
check "said so" "$(grep -c 'exists as incomplete' "$root/out.txt")" 1
check "now published" "$(status_of v0.22.0)" published
check "only one" "$(grep -c '^v0.22.0 ' "$RELHARNESS_STATE")" 1

echo "== 15. a resumed release builds the commit its tag names =="
release_from_main 0.23.0
git tag -a v0.23.0 -m "ChatLab 0.23.0" && git push -q origin v0.23.0
tagged=$(git rev-parse HEAD)
echo "later work" >> notes.txt
git add notes.txt && git commit -qm "Unrelated work" && git push -q origin HEAD:main
code=$(run); check "exit 0" "$code" 0
check "said so" "$(grep -c 'which v0.23.0 already names' "$root/out.txt")" 1
check "built the tagged commit" "$(git -C "$root/work" rev-parse HEAD)" "$tagged"
check "released 0.23.0" "$(status_of v0.23.0)" published
check "no 0.24.0" "$(grep -c v0.24.0 "$RELHARNESS_STATE")" 0

echo "== 16. a launch that fails puts the previous bundle back =="
dest="$root/Applications/ChatLab.app"
mkdir -p "$dest/Contents"
cat > "$dest/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleShortVersionString</key><string>0.1.0</string>
<key>CFBundleExecutable</key><string>ChatLab</string>
<key>CFBundleName</key><string>ChatLab</string>
</dict></plist>
PLIST
export CHATLAB_INSTALL_DEST="$dest"
release_from_main 0.24.0
code=$(RELHARNESS_OPEN_FAILS=1 ./scripts/release.sh > "$root/out.txt" 2> "$root/err.txt"; echo $?)
check "exit 1" "$code" 1
check "said so" "$(grep -c 'putting the previous bundle back' "$root/err.txt")" 1
check "the old bundle is back" "$(plutil -extract CFBundleShortVersionString raw "$dest/Contents/Info.plist")" 0.1.0
check "the release was still published" "$(status_of v0.24.0)" published

echo "== 17. rerunning that finishes the install without a second release =="
code=$(./scripts/release.sh --version 0.24.0 > "$root/out.txt" 2> "$root/err.txt"; echo $?)
check "exit 0" "$code" 0
check "installed" "$(plutil -extract CFBundleShortVersionString raw "$dest/Contents/Info.plist")" 0.24.0
check "only one release" "$(grep -c '^v0.24.0 ' "$RELHARNESS_STATE")" 1
check "no 0.25.0" "$(grep -c v0.25.0 "$RELHARNESS_STATE")" 0

echo
echo "$pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
    rm -rf "$root"
else
    echo "the scratch repository is left at $root"
fi
[ "$fail" -eq 0 ]
