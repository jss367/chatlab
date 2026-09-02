"""Single source of truth for the ChatLab release version.

Bump this, commit, then tag the commit ``v<version>`` and push the tag. The
release workflow refuses to publish when the tag and this value disagree.
"""

__version__ = "0.2.0"
BUNDLE_IDENTIFIER = "build.chatlab.token-explorer"
