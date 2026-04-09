"""Shared path resolution for replication scripts.

Adapts the argparse-based path patterns from the private repo to the
public repo layout where data lives at ``<root>/data/`` and outputs go
to ``<root>/output/{tables,figures}/``.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root(arg_root: Path | None = None) -> Path:
    if arg_root is not None:
        return arg_root.expanduser().resolve()
    env_root = os.environ.get("ODTE_REPO_ROOT") or os.environ.get("ODTEPIPE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_data_dir(root: Path | None = None) -> Path:
    return (root or get_project_root()) / "data"


def get_tables_dir(root: Path | None = None) -> Path:
    d = (root or get_project_root()) / "output" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir(root: Path | None = None) -> Path:
    d = (root or get_project_root()) / "output" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d
