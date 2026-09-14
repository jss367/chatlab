#!/bin/bash
# release.sh - cut a ChatLab release from origin/main and install it.
#
# Bumps the minor version, lands it on main, builds the bundle from the
# landed commit, runs the tests and the smoke test, tags, publishes the
# GitHub release with the zip and its checksum, then swaps the new bundle
# into /Applications and relaunches.
#
# The build needs a real Metal GPU, so the release is made here rather than
# on a GitHub runner. See "Releasing a new version" in the README.
#
# An interrupted run is resumed by running it again: a version already on
# main without a published release is finished rather than bumped past.
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_slug=${CHATLAB_RELEASE_REPO:-jss367/chatlab}
dest=${CHATLAB_INSTALL_DEST:-/Applications/ChatLab.app}
asset_name="ChatLab-macos-arm64.zip"
target=""
notes_file=""
skip_install=0

usage() {
    cat <<'USAGE'
usage: scripts/release.sh [--version X.Y.Z] [--notes-file FILE] [--skip-install]

  --version X.Y.Z   release this version instead of bumping the minor one
  --notes-file FILE release notes; without it GitHub generates them
  --skip-install    publish the release but leave the installed app alone
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version) target=${2:?--version needs a number}; shift 2 ;;
        --notes-file) notes_file=${2:?--notes-file needs a path}; shift 2 ;;
        --skip-install) skip_install=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$repo_root"

step() { printf '\n==> %s\n' "$1"; }
die() { echo "$*" >&2; exit 1; }

version_in() { sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$1"; }
bump_minor() { echo "$1" | awk -F. '{printf "%s.%s.0\n", $1, $2 + 1}'; }
higher() { printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1; }
plist_version() { plutil -extract CFBundleShortVersionString raw "$1/Contents/Info.plist"; }
released() { gh release view "v$1" --repo "$repo_slug" --json tagName >/dev/null 2>&1; }

step "Checking the working tree and GitHub access"
[ "$(uname -m)" = arm64 ] || die "ChatLab releases are Apple Silicon only; this is $(uname -m)."
command -v gh >/dev/null || die "gh is not installed."
gh auth status >/dev/null 2>&1 || die "gh is not logged in; run gh auth login."
[ -z "$(git status --porcelain)" ] || die "The working tree has changes; commit or remove them first."

git fetch --quiet origin main --tags
git merge-base --is-ancestor HEAD origin/main \
    || die "HEAD holds commits that are not on origin/main; release from a checkout of main."
git checkout --quiet --detach origin/main
echo "at $(git log --oneline -1)"

step "Choosing the version"
current=$(version_in version.py)
[ -n "$current" ] || die "Could not read __version__ from version.py."
if [ -n "$target" ]; then
    :
elif released "$current"; then
    latest=$(gh release list --repo "$repo_slug" --exclude-drafts --exclude-pre-releases \
        --limit 1 --json tagName --jq '.[0].tagName' | sed 's/^v//')
    target=$(bump_minor "$(higher "$current" "${latest:-$current}")")
    echo "v$current is published; releasing $target"
else
    target=$current
    echo "v$current is not published; finishing that release rather than bumping"
fi
# updater.parse_version accepts numbers and dots and nothing else, so a
# version it cannot read is a release every installed app would skip.
echo "$target" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
    || die "Not a MAJOR.MINOR.PATCH version: $target"
if released "$target" && [ "$target" != "$current" ]; then
    die "v$target is already published, and main is on $current; pass --version $current to reinstall that release, or a version that has not been cut."
fi
tag="v$target"

if [ "$target" != "$current" ]; then
    step "Landing $tag on main"
    sed -i '' "s/^__version__ = \".*\"$/__version__ = \"$target\"/" version.py
    [ "$(version_in version.py)" = "$target" ] || die "Failed to write the version into version.py."
    git add version.py
    git commit --quiet -m "Release ChatLab $target"
    attempt=1
    until git push --quiet origin HEAD:main; do
        [ "$attempt" -lt 3 ] || die "Could not push the version commit to main."
        echo "main moved while we were preparing; rebasing onto it and pushing again"
        git fetch --quiet origin main
        git rebase --quiet origin/main \
            || { git rebase --abort; die "Someone else changed version.py; sort that out first."; }
        attempt=$((attempt + 1))
    done
fi
release_commit=$(git rev-parse HEAD)

step "Building ChatLab.app"
./scripts/build_macos_app.sh
built=dist/ChatLab.app
[ "$(plist_version "$built")" = "$target" ] \
    || die "The built bundle says $(plist_version "$built"), not $target."

step "Running the tests"
# scripts/build_macos_app.sh honours this too, so the tests run against the
# environment the bundle was built from.
desktop_venv=${CHATLAB_DESKTOP_VENV:-"$repo_root/.desktop-venv"}
"$desktop_venv/bin/python" -m unittest discover -s tests

step "Smoke testing the bundle"
"$built/Contents/MacOS/ChatLab" --smoke-test

if released "$target"; then
    step "v$target is published already; rebuilt it to install rather than cutting another"
else
    step "Tagging $tag"
    if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
        [ "$(git rev-list -n1 "$tag")" = "$release_commit" ] \
            || die "$tag already names a different commit; resolve that before releasing."
    else
        git tag -a "$tag" -m "ChatLab $target" "$release_commit"
    fi
    git push --quiet origin "$tag"
    # Only an annotated tag has a peeled ref; a lightweight one, as an earlier
    # release made by hand may have left behind, is itself the commit.
    remote_tag=$(git ls-remote origin "refs/tags/$tag^{}" | awk '{print $1}')
    [ -n "$remote_tag" ] || remote_tag=$(git ls-remote origin "refs/tags/$tag" | awk '{print $1}')
    [ "$remote_tag" = "$release_commit" ] \
        || die "$tag on GitHub names ${remote_tag:-nothing}, not $release_commit."

    step "Packaging the bundle"
    staging=$(mktemp -d)
    trap 'rm -rf "$staging"' EXIT
    ditto -c -k --sequesterRsrc --keepParent "$built" "$staging/$asset_name"
    size=$(stat -f %z "$staging/$asset_name")
    echo "$asset_name is $size bytes"
    [ "$size" -le 2147483648 ] \
        || die "The asset is over GitHub's 2 GB release limit; the updater could not use it."
    shasum -a 256 "$staging/$asset_name" | awk '{print $1}' > "$staging/$asset_name.sha256"

    step "Publishing the release"
    if [ -n "$notes_file" ]; then
        set -- --notes-file "$notes_file"
    else
        set -- --generate-notes
    fi
    gh release create "$tag" --repo "$repo_slug" --verify-tag --title "ChatLab $tag" "$@" \
        "$staging/$asset_name" "$staging/$asset_name.sha256"
fi

if [ "$skip_install" -eq 1 ]; then
    step "Leaving $dest alone (--skip-install)"
else
    step "Installing into $dest"
    old_version=$(plist_version "$dest" 2>/dev/null || echo none)
    if pgrep -xq ChatLab; then
        osascript -e 'tell application "ChatLab" to quit' >/dev/null 2>&1 || true
        for _ in $(seq 1 20); do pgrep -xq ChatLab || break; sleep 0.5; done
        if pgrep -xq ChatLab; then pkill -x ChatLab || true; sleep 1; fi
    fi
    backup=""
    if [ -e "$dest" ]; then
        backup=$(mktemp -d)/$(basename "$dest")
        mv "$dest" "$backup"
    fi
    restore() {
        rm -rf "$dest"
        if [ -n "$backup" ] && [ -e "$backup" ]; then mv "$backup" "$dest"; fi
        open "$dest" >/dev/null 2>&1 || true
    }
    if ! ditto "$built" "$dest"; then
        echo "The copy failed; putting the previous bundle back." >&2
        restore; exit 1
    fi
    open "$dest"
    sleep 5
    if ! pgrep -xq ChatLab; then
        echo "ChatLab exited within 5s of launch; putting the previous bundle back." >&2
        restore; exit 1
    fi
    if [ -n "$backup" ]; then rm -rf "$(dirname "$backup")"; fi
    echo "installed $dest: $old_version -> $target"
fi

step "Released ChatLab $target"
echo "commit  $release_commit"
echo "release https://github.com/$repo_slug/releases/tag/$tag"
