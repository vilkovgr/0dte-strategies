#!/usr/bin/env python3
"""Test harness: verify adapted scripts produce identical outputs to reference.

Usage:
    python tests/test_replication.py              # run all tests
    python tests/test_replication.py --script structural_break  # single test
    python tests/test_replication.py --generate-refs             # regen references
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_TABLES = REPO_ROOT / "tests" / "reference" / "tables"
OUT_TABLES = REPO_ROOT / "output" / "tables"
OUT_FIGURES = REPO_ROOT / "output" / "figures"
CODE_DIR = REPO_ROOT / "code" / "analysis"

os.environ.setdefault("MPLBACKEND", "Agg")

# Each entry: (script_name, list of output .tex files to compare)
PARITY_TESTS = [
    (
        "compute_structural_break_2022.py",
        ["0dte_structbreak_post2022.tex"],
    ),
    (
        "compute_vix_regime_conditioning.py",
        ["0dte_vix_regime_1000.tex"],
    ),
    (
        "compute_tail_risk_diagnostics.py",
        ["0dte_tail_risk_diagnostics.tex"],
    ),
    (
        "compute_implementable_pnl.py",
        ["0dte_implementable_pnl.tex"],
    ),
    (
        "compute_clustered_inference_mht.py",
        ["0dte_inference_cluster_mht.tex"],
    ),
]

# Scripts that must run without error (no parity check, just exit-code = 0)
SMOKE_TESTS = [
    "option_strats_uncond_analysis.py",
    "compute_conditional_oos_protocol.py",
    "figs_strats.py",
]


def run_script(script_name: str, extra_args: list[str] | None = None) -> tuple[bool, str]:
    """Run a script and return (success, stderr)."""
    script = CODE_DIR / script_name
    cmd = [sys.executable, str(script), "--project-root", str(REPO_ROOT)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0, result.stderr


def compare_tex(filename: str) -> tuple[bool, str]:
    """Compare generated .tex to reference. Returns (match, diff_summary)."""
    ref = REF_TABLES / filename
    gen = OUT_TABLES / filename

    if not ref.exists():
        return False, f"Reference file missing: {ref}"
    if not gen.exists():
        return False, f"Generated file missing: {gen}"

    ref_lines = ref.read_text().splitlines(keepends=True)
    gen_lines = gen.read_text().splitlines(keepends=True)

    if ref_lines == gen_lines:
        return True, "IDENTICAL"

    diff = list(difflib.unified_diff(ref_lines, gen_lines, fromfile="reference", tofile="generated", n=2))
    n_changes = sum(1 for line in diff if line.startswith("+") or line.startswith("-"))
    summary = f"{n_changes} changed lines"
    if len(diff) < 40:
        summary += "\n" + "".join(diff)
    return False, summary


def run_parity_tests(args, backup: Path) -> tuple[int, int, int]:
    """Run parity tests (script output must match reference .tex)."""
    passed = failed = errors = 0

    for script_name, tex_files in PARITY_TESTS:
        if args.script and args.script not in script_name:
            continue

        print(f"\n{'='*60}")
        print(f"PARITY: {script_name}")
        print(f"{'='*60}")

        for tf in tex_files:
            src = OUT_TABLES / tf
            if src.exists():
                shutil.copy2(src, backup / tf)

        ok, stderr = run_script(script_name)
        if not ok:
            print(f"  RUN FAILED: {stderr[:500]}")
            errors += 1
            for tf in tex_files:
                bk = backup / tf
                if bk.exists():
                    shutil.copy2(bk, OUT_TABLES / tf)
            continue

        all_match = True
        for tf in tex_files:
            match, detail = compare_tex(tf)
            status = "PASS" if match else "DIFF"
            print(f"  {tf}: {status}")
            if not match:
                print(f"    {detail[:300]}")
                all_match = False

        if all_match:
            passed += 1
        else:
            failed += 1

        for tf in tex_files:
            bk = backup / tf
            if bk.exists():
                shutil.copy2(bk, OUT_TABLES / tf)

    return passed, failed, errors


def run_smoke_tests(args) -> tuple[int, int]:
    """Run smoke tests (script must exit cleanly, no parity check)."""
    passed = failed = 0

    for script_name in SMOKE_TESTS:
        if args.script and args.script not in script_name:
            continue

        print(f"\n{'='*60}")
        print(f"SMOKE: {script_name}")
        print(f"{'='*60}")

        ok, stderr = run_script(script_name)
        if ok:
            print(f"  PASS (exit 0)")
            passed += 1
        else:
            print(f"  FAIL: {stderr[:500]}")
            failed += 1

    return passed, failed


def generate_refs() -> None:
    """Regenerate all reference tables from current environment."""
    print("Regenerating reference tables...")
    REF_TABLES.mkdir(parents=True, exist_ok=True)

    for script_name, tex_files in PARITY_TESTS:
        print(f"\n  Running {script_name}...")
        ok, stderr = run_script(script_name)
        if not ok:
            print(f"  FAILED: {stderr[:300]}")
            continue
        for tf in tex_files:
            src = OUT_TABLES / tf
            if src.exists():
                shutil.copy2(src, REF_TABLES / tf)
                print(f"  Updated: {tf}")
            else:
                print(f"  WARNING: {tf} not produced")

    print("\nDone. Reference tables updated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=None, help="Run only tests matching this substring")
    parser.add_argument("--generate-refs", action="store_true",
                        help="Regenerate reference tables instead of testing")
    args = parser.parse_args()

    if args.generate_refs:
        generate_refs()
        return

    backup = REPO_ROOT / "tests" / "_output_backup"
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)

    p_passed, p_failed, p_errors = run_parity_tests(args, backup)
    s_passed, s_failed = run_smoke_tests(args)

    total_passed = p_passed + s_passed
    total_failed = p_failed + s_failed + p_errors

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Parity tests:  {p_passed} passed, {p_failed} diffs, {p_errors} errors")
    print(f"  Smoke tests:   {s_passed} passed, {s_failed} failed")
    print(f"  TOTAL:         {total_passed} passed, {total_failed} failed")
    print(f"{'='*60}")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
