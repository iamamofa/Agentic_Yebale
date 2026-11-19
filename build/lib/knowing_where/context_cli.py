"""Compatibility entry point; prefer python -m knowing_where.cli.context."""

from .cli.context import build_report, main

if __name__ == "__main__":
    main()
