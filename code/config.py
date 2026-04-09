"""
Path configuration for the 0DTE replication package.

Replaces the internal zEnvmt/LocalSetupSTVRP.py used in development.
All paths resolve relative to the repository root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    """Return repository root: env override or inferred from this file's location."""
    env = os.environ.get("ODTE_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


@dataclass
class RepoConfig:
    """Centralised path configuration for the public replication repo."""

    root: Path = field(default_factory=_repo_root)

    @property
    def data_dir(self) -> Path:
        """Shipped interpolated data panels."""
        return self.root / "data"

    @property
    def figures_dir(self) -> Path:
        return self.root / "output" / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.root / "output" / "tables"

    # ----- Compatibility aliases matching the private-repo Paths attributes -----
    @property
    def pathOL_figs_strats(self) -> Path:
        return self.figures_dir

    @property
    def pathOL_table_strats(self) -> Path:
        return self.tables_dir

    # Data files
    @property
    def structures_path(self) -> Path:
        return self.data_dir / "data_structures.parquet"

    @property
    def opt_path(self) -> Path:
        return self.data_dir / "data_opt.parquet"

    @property
    def vix_path(self) -> Path:
        return self.data_dir / "vix.parquet"

    @property
    def slopes_path(self) -> Path:
        return self.data_dir / "slopes.parquet"

    @property
    def moments_spx_path(self) -> Path:
        return self.data_dir / "future_moments_SPX.parquet"

    @property
    def moments_vix_path(self) -> Path:
        return self.data_dir / "future_moments_VIX.parquet"

    @property
    def moments_path(self) -> Path:
        """Legacy alias — prefer moments_spx_path / moments_vix_path."""
        return self.data_dir / "future_moments_SPX.parquet"

    @property
    def eod_path(self) -> Path:
        return self.data_dir / "ALL_eod.csv"

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)


CFG = RepoConfig()


# ---------------------------------------------------------------------------
# Adapter for argparse-style scripts that use --project-root / --data-version
# ---------------------------------------------------------------------------

def resolve_paths(
    project_root: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Return (data_dir, tables_dir, figures_dir, repo_root).

    Works both in the public repo layout (data/ at root) and when overridden
    via CLI arguments.
    """
    root = project_root or CFG.root
    dd = data_dir or CFG.data_dir
    tables = root / "output" / "tables"
    figures = root / "output" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return dd, tables, figures, root
