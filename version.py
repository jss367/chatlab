"""Single source of truth for the ChatLab release version.

``scripts/release.sh`` writes this, tags the commit ``v<version>`` and
publishes the release; nothing else should need to set it by hand.
"""

__version__ = "0.17.0"
BUNDLE_IDENTIFIER = "build.chatlab.app"
