#!/usr/bin/env python3
"""Sync updated data and outputs from the private dev repo to the public repo.

Usage (run from the private 0DTEpipe repo):
    python /path/to/0dte-strategies/tools/sync_to_public.py \
        --private-root /path/to/0DTEpipe \
        --public-root  /path/to/0dte-strategies \
        --data-version ver-2024-05-01

This copies:
    1. Derived data panels (parquet/csv) from Data/temp_strats/<version>/
    2. Generated tables from 0DTE-Strategies/tables/
    3. Generated figures from 0DTE-Strategies/figures/
    4. Updated analysis scripts from Code/Analysis/

It does NOT copy:
    - Raw Cboe data or proprietary source files
    - Environment config files
    - Internal pipeline scripts
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DERIVED_DATA_FILES = [
    "data_opt.parquet",
    "data_structures.parquet",
    "vix.parquet",
    "slopes.parquet",
    "ALL_eod.csv",
]

ANALYSIS_SCRIPT_MAP = {
    "compute_implementable_pnl.py": "compute_implementable_pnl.py",
    "compute_clustered_inference_mht.py": "compute_clustered_inference_mht.py",
    "compute_conditional_model_zoo.py": "compute_conditional_model_zoo.py",
    "compute_conditional_oos_protocol.py": "compute_conditional_oos_protocol.py",
    "compute_structural_break_2022.py": "compute_structural_break_2022.py",
    "compute_tail_risk_diagnostics.py": "compute_tail_risk_diagnostics.py",
    "compute_vix_regime_conditioning.py": "compute_vix_regime_conditioning.py",
    "compute_conditional_oos_investment_ts.py": "compute_conditional_oos_investment_ts.py",
    "build_conditional_target_choice_table.py": "build_conditional_target_choice_table.py",
    "derive_binary_decision_summary.py": "derive_binary_decision_summary.py",
    "moneyness_selection.py": "moneyness_selection.py",
    "plot_conditional_topk_basket_legs.py": "plot_conditional_topk_basket_legs.py",
    "option_strats_uncond_analysis.py": "option_strats_uncond_analysis.py",
}


def sync_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        print(f"  SKIP (missing): {src}")
        return False
    if dry_run:
        print(f"  DRY-RUN: {src} → {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  COPIED: {src.name} ({src.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync private → public repo.")
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--data-version", default="ver-2024-05-01")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-scripts", action="store_true")
    args = parser.parse_args()

    priv = args.private_root.resolve()
    pub = args.public_root.resolve()
    data_src = priv / "Data" / "temp_strats" / args.data_version

    copied = 0

    if not args.skip_data:
        print("\n[1] Derived data panels")
        for fname in DERIVED_DATA_FILES:
            if sync_file(data_src / fname, pub / "data" / fname, args.dry_run):
                copied += 1

        # Re-convert moments from h5 to parquet
        h5_path = data_src / "ex_post_moments.h5"
        if h5_path.exists():
            print("  Converting ex_post_moments.h5 → parquet...")
            if not args.dry_run:
                import pandas as pd
                for key in ["future_moments_SPX", "future_moments_VIX"]:
                    df = pd.read_hdf(h5_path, key=key)
                    df["time"] = df["time"].astype(str)
                    df["date"] = pd.to_datetime(df["date"])
                    out = pub / "data" / f"{key}.parquet"
                    df.to_parquet(out, index=False)
                    print(f"  CONVERTED: {key}.parquet ({len(df)} rows)")
                    copied += 1

    if not args.skip_tables:
        print("\n[2] Output tables")
        tables_src = priv / "0DTE-Strategies" / "tables"
        tables_dst = pub / "output" / "tables"
        ref_dst = pub / "tests" / "reference" / "tables"
        if tables_src.exists():
            for tex in sorted(tables_src.glob("*.tex")):
                sync_file(tex, tables_dst / tex.name, args.dry_run)
                sync_file(tex, ref_dst / tex.name, args.dry_run)
                copied += 1

    if not args.skip_figures:
        print("\n[3] Output figures")
        figs_src = priv / "0DTE-Strategies" / "figures"
        figs_dst = pub / "output" / "figures"
        if figs_src.exists():
            for pdf in sorted(figs_src.glob("*.pdf")):
                if sync_file(pdf, figs_dst / pdf.name, args.dry_run):
                    copied += 1

    if not args.skip_scripts:
        print("\n[4] Analysis scripts (manual review recommended)")
        scripts_src = priv / "Code" / "Analysis"
        scripts_dst = pub / "code" / "analysis"
        for src_name, dst_name in ANALYSIS_SCRIPT_MAP.items():
            src_path = scripts_src / src_name
            if src_path.exists():
                print(f"  NOTE: {src_name} available for manual diff/update")

    print(f"\nSync complete: {copied} files {'would be ' if args.dry_run else ''}copied.")
    if args.dry_run:
        print("(Re-run without --dry-run to execute.)")


if __name__ == "__main__":
    main()
