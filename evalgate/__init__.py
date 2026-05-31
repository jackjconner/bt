"""evalgate — capture and diff PipelineSummary golden values.

Public surface:
- ``serialize``        : convert a PipelineSummary (or any dataclass) to a JSON-safe dict
- ``load_golden``      : read a golden dict from a JSON file
- ``save_golden``      : write a dict to a JSON file (pretty-printed)
- ``diff_summaries``   : compare two serialized summaries field-by-field
- ``DiffRow``          : one field's comparison result
- ``format_diff_table``: render diff rows as a human-readable table
"""

from __future__ import annotations

from evalgate._core import (
    DiffRow,
    diff_summaries,
    format_diff_table,
    load_golden,
    save_golden,
    serialize,
)

__all__ = [
    "DiffRow",
    "diff_summaries",
    "format_diff_table",
    "load_golden",
    "save_golden",
    "serialize",
]
