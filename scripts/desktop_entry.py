"""The script PyInstaller freezes into ChatLab.app; see ChatLab.spec."""

import sys

from chatlab.desktop_launcher import main

if __name__ == "__main__":
    sys.exit(main())
