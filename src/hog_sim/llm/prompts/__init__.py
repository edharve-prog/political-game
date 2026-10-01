"""Versioned prompt templates.

Each module has a ``VERSION`` string, a ``SYSTEM`` prompt and a ``render(...)`` function for
the user turn. Bump ``VERSION`` whenever the wording changes: it is part of every recorded
response's cache key and every usage log line, so old recordings are never replayed against
a new prompt.
"""
