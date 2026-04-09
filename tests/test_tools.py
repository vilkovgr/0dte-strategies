#!/usr/bin/env python3
"""Smoke tests for documentation and tooling layer.

Verifies that:
  - doctor.py --quick exits 0 on a properly configured environment
  - latex2md.py --dry-run produces the expected section structure
  - paper-annotated.md contains required annotation blocks
  - reading-guide.md contains all five AI Reading Guide sections
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=REPO_ROOT, timeout=60, **kwargs,
    )


def test_doctor_quick():
    r = _run([sys.executable, "tools/doctor.py", "--quick"])
    assert "Base packages" in r.stdout, f"Missing 'Base packages' in output:\n{r.stdout}"
    assert "Model-zoo" in r.stdout, f"Missing 'Model-zoo' in output:\n{r.stdout}"
    assert "Python" in r.stdout, f"Missing Python check in output:\n{r.stdout}"
    if r.returncode != 0:
        assert "FAIL" in r.stdout, (
            f"doctor.py --quick exited {r.returncode} without any FAIL lines"
        )


def test_latex2md_dry_run():
    tex = REPO_ROOT / "tests" / "fixtures" / "sample_paper.tex"
    r = _run([sys.executable, "tools/latex2md.py", str(tex), "--dry-run"])
    assert r.returncode == 0, f"latex2md.py --dry-run failed:\n{r.stderr}"
    lines = r.stdout.strip().splitlines()
    section_lines = [l for l in lines if l.strip().startswith("##")]
    assert len(section_lines) >= 5, (
        f"Expected >=5 section headers, got {len(section_lines)}"
    )


def test_latex2md_full_conversion():
    tex = REPO_ROOT / "tests" / "fixtures" / "sample_paper.tex"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out = Path(f.name)
    try:
        r = _run([sys.executable, "tools/latex2md.py", str(tex), "-o", str(out)])
        assert r.returncode == 0, f"latex2md.py failed:\n{r.stderr}"
        text = out.read_text(encoding="utf-8")

        assert "<!-- @document-metadata" in text, "Missing @document-metadata"

        import re
        blocks = re.findall(r"<!-- @section-type:", text)
        assert len(blocks) >= 5, f"Expected >=5 @section-type blocks, got {len(blocks)}"

        assert "$$" in text, "No math blocks in output"
        assert "> **Table**" in text, "No table references in output"
        assert "> **Figure**" in text, "No figure references in output"
        assert "\\begin{figure" not in text, "Raw LaTeX figure environment leaked"
        assert "\\begin{table" not in text, "Raw LaTeX table environment leaked"
        assert "\\appendix" not in text, "Raw \\appendix command leaked"
        assert "\\newpage" not in text, "Raw \\newpage command leaked"
    finally:
        out.unlink(missing_ok=True)


def test_paper_annotated_structure():
    path = REPO_ROOT / "docs" / "paper" / "paper-annotated.md"
    if not path.exists():
        return  # Not yet generated
    text = path.read_text(encoding="utf-8")

    assert "<!-- @document-metadata" in text, "Missing @document-metadata block"

    import re
    section_blocks = re.findall(r"<!-- @section-type:", text)
    assert len(section_blocks) >= 15, (
        f"Expected >=15 @section-type blocks, got {len(section_blocks)}"
    )

    for field in ("@key-claim:", "@importance:", "@equations:"):
        count = text.count(field)
        assert count >= 10, f"Expected >=10 occurrences of {field}, got {count}"


def test_reading_guide_sections():
    path = REPO_ROOT / "docs" / "agent-context" / "reading-guide.md"
    text = path.read_text(encoding="utf-8")

    required = [
        "Three Core Claims",
        "Structure",
        "Key Methodological Choices",
    ]
    for section in required:
        assert section in text, f"reading-guide.md missing section: {section}"


def test_variables_sections():
    path = REPO_ROOT / "docs" / "agent-context" / "variables.md"
    text = path.read_text(encoding="utf-8")
    assert "Implied Measures" in text, "variables.md missing Implied Measures"
    assert "Strategy-Level Conditional Features" in text, (
        "variables.md missing Strategy-Level Conditional Features"
    )


def test_requirements_upper_bounds():
    path = REPO_ROOT / "requirements.txt"
    text = path.read_text(encoding="utf-8")
    pkg_lines = [
        l.strip() for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    for line in pkg_lines:
        assert ">=" in line, f"Missing lower bound in: {line}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = skipped = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)
