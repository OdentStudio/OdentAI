"""Entry point for PyInstaller. A separate file because `-m dental9` does
not work once frozen: the binary has no package to import by name."""
import multiprocessing
import sys

from dental9.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()   # otherwise child processes on Windows
    sys.exit(main())                   # re-launch the whole binary
