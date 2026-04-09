# Quick Start Guide

## Prerequisites

- **Python ≥ 3.10** (3.11 or 3.12 recommended)
- **Git LFS** — the data files are stored with Git Large File Storage
- ~500 MB disk space for data panels

## Step 1: Install Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt-get install git-lfs

# Then initialize (once per machine)
git lfs install
```

## Step 2: Clone the Repository

```bash
git clone https://github.com/vilkovgr/0dte-strategies.git
cd 0dte-strategies
```

If you already cloned without LFS, pull the data:
```bash
git lfs pull
```

## Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Verify Your Setup

```bash
python tools/doctor.py
```

This checks Python version, required packages, data file presence, and file sizes.

## Step 6: Reproduce All Exhibits

```bash
python code/run_replication.py
```

Tables appear in `output/tables/`, figures in `output/figures/`.

## Step 7: Verify Parity

```bash
# Table parity: generated tables vs. reference
python tests/test_replication.py

# Tooling: doctor, converter, documentation structure
python tests/test_tools.py
```

`test_replication.py` compares generated tables byte-for-byte against `tests/reference/tables/`.
`test_tools.py` verifies the environment checker, LaTeX converter, and documentation completeness.

---

## Troubleshooting

### "File not found" errors for data files

Git LFS pointers were not resolved. Run:
```bash
git lfs pull
```

### Missing `statsmodels` or other package

```bash
pip install -r requirements.txt --upgrade
```

### Scripts fail with import errors

Make sure you run scripts from the repo root or set `ODTE_REPO_ROOT`:
```bash
export ODTE_REPO_ROOT=/path/to/0dte-strategies
```

### Large memory usage

`data_opt.parquet` is ~250 MB on disk and expands to ~2 GB in memory. Ensure at least 4 GB of free RAM.

---

## Optional: Rebuild Data from Source (Tier 2)

If you hold a Massive or ThetaData API subscription:

1. Copy `.env.example` to `.env` and fill in your API key
2. Run the appropriate ingest adapter:

```bash
# Massive
python code/ingest/massive/download_spxw.py --start 2016-09-01 --end 2026-02-01

# ThetaData
python code/ingest/thetadata/download_spxw.py --start 2016-09-01 --end 2026-02-01
```

3. Build derived panels:
```bash
python code/build_data.py --source massive   # or --source thetadata
```
