#!/usr/bin/env python3
"""Environment and data health check for the 0DTE replication package.

Usage:
    python tools/doctor.py            # full check
    python tools/doctor.py --quick    # Python + dependencies only
    python tools/doctor.py --verbose  # show passing checks in detail
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mpl_doctor_")

REPO_ROOT = Path(__file__).resolve().parents[1]

MIN_PYTHON = (3, 10)

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("statsmodels", "statsmodels"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("pyarrow", "pyarrow"),
    ("h5py", "h5py"),
    ("jinja2", "jinja2"),
    ("seaborn", "seaborn"),
]

MODELZOO_PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("catboost", "catboost"),
]

REQUIRED_DATA_FILES = [
    "data_opt.parquet",
    "data_structures.parquet",
    "vix.parquet",
    "slopes.parquet",
    "future_moments_SPX.parquet",
    "future_moments_VIX.parquet",
    "ALL_eod.csv",
]

LFS_POINTER_SIGNATURE = b"version https://git-lfs.github.com/spec/"
LFS_POINTER_MAX_BYTES = 200


def _col(text: str, code: int) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _pass(msg: str) -> str:
    return _col(f"  PASS  {msg}", 32)


def _fail(msg: str) -> str:
    return _col(f"  FAIL  {msg}", 31)


def _warn(msg: str) -> str:
    return _col(f"  WARN  {msg}", 33)


def _info(msg: str) -> str:
    return _col(f"  INFO  {msg}", 36)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(LFS_POINTER_MAX_BYTES)
        return head.startswith(LFS_POINTER_SIGNATURE)
    except OSError:
        return False


def _validate_parquet(path: Path) -> tuple[bool, str]:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        return True, f"{meta.num_rows:,} rows x {meta.num_columns} cols"
    except Exception as exc:
        return False, str(exc)


def _validate_csv(path: Path) -> tuple[bool, str]:
    try:
        with open(path) as f:
            header = f.readline()
            first = f.readline()
        if not header.strip():
            return False, "empty file"
        if not first.strip():
            return False, "header only, no data rows"
        ncols = len(header.split(","))
        return True, f"{ncols} columns"
    except Exception as exc:
        return False, str(exc)


def check_python() -> list[str]:
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        print(_pass(label))
        return []
    msg = f"{label} — requires {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+"
    print(_fail(msg))
    return [msg]


def check_packages(pkgs: list[tuple[str, str]], tier: str) -> list[str]:
    failures: list[str] = []
    for import_name, display_name in pkgs:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(_pass(f"[{tier}] {display_name} ({ver})"))
        except ImportError:
            msg = f"[{tier}] {display_name} — not installed"
            if tier == "model-zoo":
                print(_warn(msg))
            else:
                print(_fail(msg))
                failures.append(msg)
        except Exception as exc:
            msg = f"[{tier}] {display_name} — import error: {exc}"
            if tier == "model-zoo":
                print(_warn(msg))
            else:
                print(_fail(msg))
                failures.append(msg)
    return failures


def check_data(verbose: bool) -> list[str]:
    data_dir = REPO_ROOT / "data"
    failures: list[str] = []

    for fname in REQUIRED_DATA_FILES:
        fpath = data_dir / fname

        if not fpath.exists():
            msg = f"{fname} — missing (run: git lfs pull)"
            print(_fail(msg))
            failures.append(msg)
            continue

        if _is_lfs_pointer(fpath):
            msg = f"{fname} — unresolved LFS pointer (run: git lfs pull)"
            print(_fail(msg))
            failures.append(msg)
            continue

        if fname.endswith(".parquet"):
            ok, detail = _validate_parquet(fpath)
        elif fname.endswith(".csv"):
            ok, detail = _validate_csv(fpath)
        else:
            ok, detail = fpath.stat().st_size > 0, "non-empty"

        size_mb = fpath.stat().st_size / 1_000_000
        if ok:
            if verbose:
                print(_pass(f"{fname:<35s} {size_mb:>7.1f} MB  ({detail})"))
            else:
                print(_pass(fname))
        else:
            msg = f"{fname} — corrupt ({detail})"
            print(_fail(msg))
            failures.append(msg)

    return failures


def check_output_dirs() -> None:
    for subdir in ("tables", "figures"):
        d = REPO_ROOT / "output" / subdir
        if d.exists():
            print(_pass(f"output/{subdir}/"))
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(_info(f"output/{subdir}/ created"))


def check_optional_keys() -> None:
    for key in ("MASSIVE_API_KEY", "THETADATA_USERNAME"):
        val = os.environ.get(key)
        if val:
            print(_pass(f"{key} set"))
        else:
            print(_info(f"{key} not set (Tier 2 rebuild only)"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="0DTE replication package — environment and data health check."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check Python and dependencies only; skip data and output validation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed info for passing checks (file sizes, row counts).",
    )
    args = parser.parse_args()

    print()
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Python:    {sys.executable}")
    print()

    all_failures: list[str] = []

    print("── Python ──")
    all_failures.extend(check_python())
    print()

    print("── Base packages (Tier 1) ──")
    all_failures.extend(check_packages(REQUIRED_PACKAGES, "base"))
    print()

    print("── Model-zoo packages (optional) ──")
    check_packages(MODELZOO_PACKAGES, "model-zoo")
    print()

    if not args.quick:
        print("── Data files ──")
        all_failures.extend(check_data(verbose=args.verbose))
        print()

        print("── Output directories ──")
        check_output_dirs()
        print()

        print("── API keys (Tier 2) ──")
        check_optional_keys()
        print()

    print("─" * 55)
    if all_failures:
        print(_fail(f"{len(all_failures)} issue(s) found:"))
        for f in all_failures:
            print(f"    - {f}")
        print()
        return 1
    else:
        print(_pass("All checks passed. Ready to replicate."))
        print(f"  Run:  python code/run_replication.py")
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
