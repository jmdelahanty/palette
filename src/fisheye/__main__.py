"""
Main entry point for the fisheye package.

Allows running as: python -m fisheye
"""

import sys
from .core.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
