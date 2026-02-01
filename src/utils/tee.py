"""Utilities for capturing and mirroring output."""

import io
import sys


class TeeOutput:
    """Capture stdout while still printing to console."""

    def __init__(self):
        self.buffer = io.StringIO()
        self.stdout = sys.stdout

    def write(self, text):
        self.buffer.write(text)
        self.stdout.write(text)

    def flush(self):
        self.stdout.flush()

    def getvalue(self):
        return self.buffer.getvalue()
