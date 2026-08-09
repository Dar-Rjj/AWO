#!/usr/bin/env python3
"""Run the repository OpenRouter preflight from a source checkout."""

from __future__ import annotations

import sys

from awo.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["preflight", *sys.argv[1:]]))
