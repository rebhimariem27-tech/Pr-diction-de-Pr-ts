"""Compatibility layer for Azure SQL driver imports.

The project historically used the ``mssql_python`` package, but this driver is
not always available in local or CI environments. The application should still
import cleanly so that model evaluation and validation code can run without a
live Azure SQL connection.
"""
from __future__ import annotations

from typing import Any

try:
    from mssql_python import connect as _connect
except ModuleNotFoundError:
    _connect = None
except ImportError:
    _connect = None


def connect(*args: Any, **kwargs: Any):
    if _connect is None:
        raise RuntimeError(
            "Azure SQL support requires the 'mssql-python' package or an equivalent "
            "compatible driver. Install it before using Azure SQL-backed flows."
        )
    return _connect(*args, **kwargs)
