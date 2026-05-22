"""Pytest configuration for the microssim test subpackage.

Excludes the offline fixture generator from collection. Pytest's default
``python_files = "test_*.py *_test.py"`` already skips ``_generate_golden.py``
by filename, but ``collect_ignore`` survives any future config change.
"""

collect_ignore = ["_generate_golden.py"]
