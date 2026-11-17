"""Console plumbing shared by every benchmark CLI entry point.

``sys.stdout`` is opened with the platform's default codec, which on Windows is
GBK — printing the Chinese report text then raises ``UnicodeEncodeError`` and the
CLI dies after having done all the work. Every entry point therefore forces UTF-8
on stdout first; the helper keeps that fix in one place instead of six.
"""

from __future__ import annotations

import contextlib
import sys


def force_utf8_stdout() -> None:
    """Best-effort UTF-8 stdout; a console that refuses the switch still prints."""
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
