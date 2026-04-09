#!/usr/bin/env python3
"""Test harness: verify adapted scripts produce identical outputs to reference.

Usage:
    python tests/test_replication.py              # run all tests
    python tests/test_replication.py --script structural_break  # single test
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_TABLES = REPO_ROOT / "tests" / "reference" / "tables"
OUT_TABLES = REPO_ROOT / "output" / "tables"
OUT_FIGURES = REPO_ROOT / "output" / "figures"
CODE_DIR = REPO_ROOT / "code" / "analysis"

# Each entry: (script_name, list of output .tex files to compare)
TESTS = [
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


def run_script(script_name: str) -> tuple[bool, str]:
    """Run a script and return (success, stderr)."""
    script = CODE_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
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
    n_changes = sum(1 for l in diff if l.startswith("+") or l.startswith("-"))
    summary = f"{n_changes} changed lines"
    if len(diff) < 40:
        summary += "\n" + "".join(diff)
    return False, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=None, help="Run only tests matching this substring")
    args = parser.parse_args()

    # Back up current outputs before overwriting
    backup = REPO_ROOT / "tests" / "_output_backup"
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)

    passed = 0
    failed = 0
    errors = 0

    for script_name, tex_files in TESTS:
        if args.script and args.script not in script_name:
            continue

        print(f"\n{'='*60}")
        print(f"TEST: {script_name}")
        print(f"{'='*60}")

        # Back up the tex files that will be overwritten
        for tf in tex_files:
            src = OUT_TABLES / tf
            if src.exists():
                shutil.copy2(src, backup / tf)

        # Run the adapted script
        ok, stderr = run_script(script_name)
        if not ok:
            print(f"  RUN FAILED: {stderr[:500]}")
            errors += 1
            # Restore backups
            for tf in tex_files:
                bk = backup / tf
                if bk.exists():
                    shutil.copy2(bk, OUT_TABLES / tf)
            continue

        # Compare outputs
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

        # Restore reference versions
        for tf in tex_files:
            bk = backup / tf
            if bk.exists():
                shutil.copy2(bk, OUT_TABLES / tf)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} diffs, {errors} errors")
    print(f"{'='*60}")

    sys.exit(0 if (failed == 0 and errors == 0) else 1)


if __name__ == "__main__":
    main()
