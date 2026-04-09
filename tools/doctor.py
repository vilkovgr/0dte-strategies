#!/usr/bin/env python3
"""Environment and data health check for the 0DTE replication package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("statsmodels", "statsmodels"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("pyarrow", "pyarrow"),
    ("h5py", "h5py"),
]

REQUIRED_DATA_FILES = {
    "data_opt.parquet": 200_000_000,
    "data_structures.parquet": 10_000_000,
    "vix.parquet": 500_000,
    "slopes.parquet": 100_000,
    "future_moments_SPX.parquet": 50_000_000,
    "future_moments_VIX.parquet": 10_000_000,
    "ALL_eod.csv": 1_000_000,
}


def check_python() -> bool:
    ok = sys.version_info >= (3, 10)
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "OK" if ok else "FAIL (need >= 3.10)"
    print(f"  Python {ver:>10s}  {status}")
    return ok


def check_packages() -> tuple[int, int]:
    passed = 0
    total = len(REQUIRED_PACKAGES)
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pip_name:<20s} {ver:<12s} OK")
            passed += 1
        except ImportError:
            print(f"  {pip_name:<20s} {'—':<12s} MISSING  (pip install {pip_name})")
    return passed, total


def check_data() -> tuple[int, int]:
    data_dir = REPO_ROOT / "data"
    passed = 0
    total = len(REQUIRED_DATA_FILES)
    for fname, min_bytes in REQUIRED_DATA_FILES.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  {fname:<35s} MISSING  (run: git lfs pull)")
        elif fpath.stat().st_size < min_bytes:
            size_mb = fpath.stat().st_size / 1_000_000
            print(f"  {fname:<35s} TOO SMALL ({size_mb:.1f} MB) — LFS pointer not resolved?")
        else:
            size_mb = fpath.stat().st_size / 1_000_000
            print(f"  {fname:<35s} {size_mb:>7.1f} MB  OK")
            passed += 1
    return passed, total


def check_output_dirs() -> None:
    for subdir in ("tables", "figures"):
        d = REPO_ROOT / "output" / subdir
        if d.exists():
            print(f"  output/{subdir}/  exists")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  output/{subdir}/  created")


def check_optional_keys() -> None:
    import os
    for key in ("MASSIVE_API_KEY", "THETADATA_USERNAME"):
        val = os.environ.get(key)
        if val:
            print(f"  {key:<25s} set")
        else:
            print(f"  {key:<25s} not set (Tier 2 rebuild only)")


def main() -> None:
    print("=" * 60)
    print("0DTE Replication Package — Environment Check")
    print("=" * 60)

    print("\n[1] Python Version")
    py_ok = check_python()

    print("\n[2] Required Packages")
    pkg_ok, pkg_total = check_packages()

    print("\n[3] Data Files")
    data_ok, data_total = check_data()

    print("\n[4] Output Directories")
    check_output_dirs()

    print("\n[5] Optional API Keys (Tier 2)")
    check_optional_keys()

    print("\n" + "=" * 60)
    all_ok = py_ok and (pkg_ok == pkg_total) and (data_ok == data_total)
    if all_ok:
        print("All checks passed. Ready to replicate.")
        print("  Run:  python code/run_replication.py")
    else:
        issues = []
        if not py_ok:
            issues.append("Python < 3.10")
        if pkg_ok < pkg_total:
            issues.append(f"{pkg_total - pkg_ok} missing package(s)")
        if data_ok < data_total:
            issues.append(f"{data_total - data_ok} missing/incomplete data file(s)")
        print(f"Issues found: {', '.join(issues)}")
        print("Fix the above before running replication.")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
