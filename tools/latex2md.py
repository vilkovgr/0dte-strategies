#!/usr/bin/env python3
r"""LaTeX-to-annotated-markdown converter for AI-augmented replication packages.

Parses an academic LaTeX paper and produces a markdown file annotated with
machine-readable HTML comment blocks (@document-metadata, @section-type)
following the schema used by the 0DTE replication companion papers.

Usage:
    python tools/latex2md.py paper.tex -o docs/paper/paper-annotated.md
    python tools/latex2md.py paper.tex --reading-guide docs/agent-context/reading-guide.md
    python tools/latex2md.py paper.tex --dry-run   # preview sections only

The converter handles:
  - Section/subsection/paragraph structure with @section-type annotations
  - LaTeX math to markdown math ($...$, $$...$$)
  - Figure/table environments to captioned references
  - Citations (\cite, \citealp, \citet) to (Author, Year) placeholders
  - Footnotes to markdown [^n] syntax
  - \textbf, \emph, \texttt to **, *, `
  - Cross-references (\ref, \eqref) to human-readable labels
"""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ── Section-type classifier ─────────────────────────────────────────

SECTION_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("introduction", ["introduction", "intro"]),
    ("results", ["unconditional", "option strat", "individual option",
                 "variance risk premium", "vrp", "performance",
                 "implementab", "tail risk", "capital"]),
    ("drivers", ["driver", "what drives", "moment measure"]),
    ("data", ["data,", "data and", "data on", "variable construction",
              "underlying market"]),
    ("methodology", ["methodology", "method", "model", "protocol",
                      "conditional features", "feature construction",
                      "out-of-sample trading", "out-of-sample protocol"]),
    ("conditional", ["conditional signal", "conditional", "signal",
                     "vix regime", "filter", "model zoo", "target choice",
                     "oos portfolio", "oos investment",
                     "portfolio implementation"]),
    ("robustness", ["robust", "inference", "multiple testing", "clustering",
                    "structural break", "regime stability"]),
    ("conclusion", ["conclusion", "practical implication", "summary"]),
    ("acknowledgments", ["acknowledgment", "ai use disclosure"]),
    ("appendix", ["appendix", "additional table", "additional figure",
                  "feature dictionary"]),
]

IMPORTANCE_RULES: dict[str, str] = {
    "introduction": "core",
    "data": "core",
    "results": "core",
    "drivers": "core",
    "conditional": "core",
    "conclusion": "core",
    "methodology": "supporting",
    "robustness": "supporting",
    "acknowledgments": "peripheral",
    "appendix": "supporting",
}


def classify_section(title: str) -> str:
    title_lower = title.lower()
    for stype, keywords in SECTION_TYPE_RULES:
        if any(kw in title_lower for kw in keywords):
            return stype
    return "results"


# ── LaTeX-to-markdown conversion engine ─────────────────────────────

@dataclass
class LatexSection:
    level: int  # 1=section, 2=subsection, 3=paragraph
    title: str
    label: str = ""
    body: str = ""
    section_type: str = ""
    importance: str = ""
    equations: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    key_claim: str = ""


def _expand_user_macros(tex: str) -> str:
    """Expand simple \\newcommand{\\foo}{bar} macros in the document body."""
    macros: dict[str, str] = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", tex):
        macros[m.group(1)] = m.group(2)
    for name, expansion in macros.items():
        tex = re.sub(r"\\" + name + r"(?![a-zA-Z])", lambda _: expansion, tex)
    return tex


def _extract_metadata(tex: str) -> dict[str, str]:
    """Pull document-level metadata from the LaTeX preamble."""
    meta: dict[str, str] = {}

    m = re.search(r"\\newcommand\{\\titleText\}\{(.+?)\}", tex)
    if m:
        meta["title"] = _clean_latex_inline(m.group(1))

    m = re.search(r"\\date\{\{\\large (.+?)\}", tex)
    if m:
        meta["date"] = m.group(1).strip()

    m = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", tex, re.DOTALL)
    if m:
        meta["abstract"] = _clean_latex_inline(m.group(1)).strip()

    m = re.search(r"\\textbf\{Keywords:\}(.+?)$", tex, re.MULTILINE)
    if m:
        meta["keywords"] = _clean_latex_inline(m.group(1)).strip()

    m = re.search(r"\\emph\{Portfolio takeaway:\}(.+?)$", tex, re.MULTILINE)
    if m:
        meta["portfolio_takeaway"] = _clean_latex_inline(m.group(1)).strip()

    authors = re.findall(r"\{\\large (.+?)\}", tex)
    if authors:
        meta["authors"] = "; ".join(_clean_latex_inline(a) for a in authors)

    return meta


_CLEAN_PATTERNS: list[tuple[str, str]] = [
    (r"\\&", "&"),
    (r"\\%", "%"),
    (r"\\#", "#"),
    (r"~", " "),
    (r"\\,", " "),
    (r"\\ ", " "),
    (r"\\noindent\s*", ""),
    (r"\\medskip", ""),
    (r"\\bigskip", ""),
    (r"\\clearpage", ""),
    (r"\\openup\s+\d+pt", ""),
    (r"\\singlespacing", ""),
    (r"\\doublespacing", ""),
    (r"\\onehalfspacing", ""),
    (r"\\thispagestyle\{[^}]*\}", ""),
    (r"\\pagestyle\{[^}]*\}", ""),
    (r"\\setcounter\{[^}]*\}\{\d+\}", ""),
    (r"\\renewcommand\{[^}]*\}\{[^}]*\}", ""),
    (r"\\hypersetup\{[^}]*\}", ""),
]


def _clean_latex_inline(text: str) -> str:
    """Strip simple LaTeX formatting commands, preserving math."""
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\cite(?:alp|t|p|year)?\*?\{([^}]*)\}", _format_cite, text)
    text = re.sub(r"\\citealp\{([^}]*)\}", _format_cite, text)
    text = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", text)
    text = re.sub(r"\\eqref\{([^}]*)\}", r"(\1)", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\red\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\blue\{([^}]*)\}", r"\1", text)
    for pat, repl in _CLEAN_PATTERNS:
        text = re.sub(pat, repl, text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _format_cite(m: re.Match) -> str:
    keys = m.group(1).split(",")
    return "(" + ", ".join(k.strip() for k in keys) + ")"


def _convert_math_environments(text: str) -> str:
    """Convert LaTeX math environments to markdown math blocks."""
    def _align_repl(m: re.Match) -> str:
        content = m.group(1)
        content = re.sub(r"\\label\{([^}]*)\}", r"  \\tag{\1}", content)
        content = content.replace("&", " ")
        lines = [l.strip().rstrip("\\\\").strip()
                 for l in content.split("\n") if l.strip()]
        return "\n$$\n" + "\n".join(lines) + "\n$$\n"

    text = re.sub(
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        _align_repl, text, flags=re.DOTALL
    )
    text = re.sub(
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        _align_repl, text, flags=re.DOTALL
    )
    text = re.sub(
        r"\\begin\{cases\}(.*?)\\end\{cases\}",
        lambda m: "\\begin{cases}" + m.group(1) + "\\end{cases}",
        text, flags=re.DOTALL,
    )
    return text


def _extract_braced(text: str, start: int) -> str:
    """Extract content from a balanced {...} group starting at position start."""
    if start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
        i += 1
    return text[start + 1 :]


def _find_caption(content: str) -> str:
    """Find \\caption{...} handling nested braces."""
    idx = content.find("\\caption{")
    if idx < 0:
        return ""
    return _extract_braced(content, idx + len("\\caption"))


def _convert_figure(m: re.Match) -> str:
    """Convert a figure/figure* environment to markdown."""
    content = m.group(1)
    caption = _find_caption(content)
    label_m = re.search(r"\\label\{([^}]+)\}", content)

    caption_md = _clean_latex_inline(caption) if caption else "Figure"
    label = label_m.group(1) if label_m else ""

    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content)
    subfig_caps = re.findall(r"\\subfigure\[([^\]]*)\]", content)

    lines = [f"\n> **Figure** ({label}): {caption_md}\n"]
    if subfig_caps:
        for i, sc in enumerate(subfig_caps):
            file_ref = graphics[i] if i < len(graphics) else ""
            lines.append(f">   Panel {chr(65+i)}: {_clean_latex_inline(sc)} [{file_ref}]")
    elif graphics:
        for g in graphics:
            lines.append(f">   File: {g}")
    lines.append("")
    return "\n".join(lines)


def _convert_table(m: re.Match) -> str:
    """Convert a table/table* environment to markdown."""
    content = m.group(1)
    caption = _find_caption(content)
    label_m = re.search(r"\\label\{([^}]+)\}", content)
    input_m = re.search(r"\\input\{([^}]+)\}", content)

    caption_md = _clean_latex_inline(caption) if caption else "Table"
    label = label_m.group(1) if label_m else ""
    input_file = input_m.group(1).strip() if input_m else ""

    source_line = f"\n> Source: `{input_file}`" if input_file else ""
    return f"\n> **Table** ({label}): {caption_md}{source_line}\n\n"


def _convert_footnotes(text: str) -> tuple[str, int]:
    """Convert \\footnote{{...}} to markdown [^n] syntax."""
    footnotes: list[str] = []
    counter = [0]

    def _repl(m: re.Match) -> str:
        counter[0] += 1
        n = counter[0]
        content = _clean_latex_inline(m.group(1))
        footnotes.append(f"[^{n}]: {content}")
        return f"[^{n}]"

    text = re.sub(r"\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}", _repl, text)

    if footnotes:
        text = text + "\n\n" + "\n\n".join(footnotes) + "\n"
    return text, counter[0]


def _strip_latex_preamble(tex: str) -> str:
    """Return only the document body."""
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex, re.DOTALL)
    return m.group(1) if m else tex


def _strip_comments(tex: str) -> str:
    """Remove LaTeX comment lines (but not \\% escapes)."""
    lines = []
    for line in tex.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        idx = 0
        while True:
            idx = line.find("%", idx)
            if idx < 0:
                break
            if idx == 0 or line[idx - 1] != "\\":
                line = line[:idx]
                break
            idx += 1
        lines.append(line)
    return "\n".join(lines)


def _split_sections(body: str) -> list[LatexSection]:
    """Split the LaTeX body into sections/subsections/paragraphs."""
    pattern = re.compile(
        r"\\(section|subsection|paragraph)\*?\{(.+?)\}\s*"
        r"(?:\\label\{([^}]*)\})?"
    )
    sections: list[LatexSection] = []
    last_end = 0

    for m in pattern.finditer(body):
        if sections:
            sections[-1].body = body[last_end:m.start()]

        level_map = {"section": 1, "subsection": 2, "paragraph": 3}
        level = level_map.get(m.group(1), 1)
        title = _clean_latex_inline(m.group(2))
        label = m.group(3) or ""

        sec = LatexSection(level=level, title=title, label=label)
        sec.section_type = classify_section(title)
        sec.importance = IMPORTANCE_RULES.get(sec.section_type, "supporting")
        sections.append(sec)
        last_end = m.end()

    if sections:
        sections[-1].body = body[last_end:]

    return sections


def _extract_section_metadata(sec: LatexSection) -> None:
    """Extract equations, tables, figures from a section's body."""
    sec.equations = re.findall(r"\\label\{(eq:[^}]+)\}", sec.body)
    sec.tables = re.findall(r"\\label\{(tab[^}]*)\}", sec.body)
    sec.figures = re.findall(r"\\label\{(fig[^}]*)\}", sec.body)

    claim = sec.body.replace("\n", " ").strip()
    trunc = re.search(r"\\begin\{(figure|table)\*?\}", claim)
    if trunc:
        claim = claim[: trunc.start()]
    claim = re.sub(r"\\(appendix|newpage|clearpage)\b", "", claim)
    claim = claim[:500]
    claim = _clean_latex_inline(claim).strip()
    if len(claim) > 200:
        claim = claim[:200].rsplit(" ", 1)[0] + "..."
    sec.key_claim = claim


def _convert_body(body: str) -> str:
    """Convert a section body from LaTeX to markdown."""
    body = _convert_math_environments(body)

    for env in ("figure", "figure*"):
        body = re.sub(
            r"\\begin\{" + re.escape(env) + r"\}(?:\[!?[hHtTbBpP!]*\])?\s*(.*?)\\end\{" + re.escape(env) + r"\}",
            _convert_figure, body, flags=re.DOTALL,
        )
    for env in ("table", "table*"):
        body = re.sub(
            r"\\begin\{" + re.escape(env) + r"\}(?:\[!?[hHtTbBpP!]*\])?\s*(.*?)\\end\{" + re.escape(env) + r"\}",
            _convert_table, body, flags=re.DOTALL,
        )

    body = re.sub(r"\\begin\{spacing\}\{[^}]*\}(.*?)\\end\{spacing\}",
                  r"\1", body, flags=re.DOTALL)
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  lambda m: "\n**Abstract:** " + m.group(1).strip() + "\n",
                  body, flags=re.DOTALL)

    _strip_envs = [
        "center", "landscape", "threeparttable",
        "adjustbox", "tabularx", "longtable", "tabular",
    ]
    for env in _strip_envs:
        body = re.sub(
            r"\\begin\{" + re.escape(env) + r"\}(?:\{[^}]*\}|\[[^\]]*\])*"
            r"(.*?)"
            r"\\end\{" + re.escape(env) + r"\}",
            r"\1", body, flags=re.DOTALL,
        )

    body = re.sub(r"\\appendix", "", body)
    body = re.sub(r"\\newpage", "", body)
    body = re.sub(r"\\clearpage", "", body)
    body = re.sub(r"\\vspace\{[^}]*\}", "", body)
    body = re.sub(r"\\hspace\{[^}]*\}", "", body)
    body = re.sub(r"\\maketitle", "", body)
    body = re.sub(r"\\title\{[^}]*\}", "", body)
    body = re.sub(r"\\author\{[^}]*\}", "", body)
    body = re.sub(r"\\date\{[^}]*\}", "", body)

    body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
    body = re.sub(r"\\bibliography\{[^}]*\}", "", body)

    body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
    body = re.sub(r"\\small\b", "", body)
    body = re.sub(r"\\scriptsize\b", "", body)
    body = re.sub(r"\\centering\b", "", body)
    body = re.sub(r"\\subfiguretopcaptrue\b", "", body)
    body = re.sub(r"\\input\{[^}]*\}", "", body)
    body = re.sub(r"\\adjustbox\{[^}]*\}\{", "", body)
    body = re.sub(r"\\begin\{[a-zA-Z*]+\}(?:\{[^}]*\}|\[[^\]]*\])*", "", body)
    body = re.sub(r"\\end\{[a-zA-Z*]+\}", "", body)

    body, _ = _convert_footnotes(body)
    body = _clean_latex_inline(body)

    paragraphs = re.split(r"\n\s*\n", body)
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith("$$") or p.startswith(">"):
            cleaned.append(p)
        else:
            cleaned.append(textwrap.fill(p, width=100))
    return "\n\n".join(cleaned)


# ── Assembly ────────────────────────────────────────────────────────

def build_document_metadata(
    meta: dict[str, str],
    sections: list[LatexSection],
    overrides: dict[str, str] | None = None,
) -> str:
    """Build @document-metadata from parsed preamble fields and section inventory."""
    ov = overrides or {}

    all_eqs = sorted({e for s in sections for e in s.equations})
    all_tabs = sorted({t for s in sections for t in s.tables})
    all_figs = sorted({f for s in sections for f in s.figures})

    eq_summary = f"({all_eqs[0]})-({all_eqs[-1]})" if all_eqs else "none"
    tab_count = len(all_tabs)
    fig_count = len(all_figs)

    abstract = meta.get("abstract", "")
    core_q = abstract[:120].rsplit(" ", 1)[0] + "..." if len(abstract) > 120 else abstract
    keywords = meta.get("keywords", "")

    lines = [
        "<!-- @document-metadata",
        f'  @title: {ov.get("title", meta.get("title", ""))}',
        f'  @type: {ov.get("type", "academic-paper")}',
        f'  @core-question: {ov.get("core-question", core_q)}',
        f'  @core-answer: {ov.get("core-answer", "")}',
        f'  @keywords: {ov.get("keywords", keywords)}',
        f'  @datasets: {ov.get("datasets", "")}',
        f'  @key-equations: {ov.get("key-equations", eq_summary)}',
        f'  @key-tables: {tab_count} tables ({", ".join(all_tabs[:5])}{"..." if tab_count > 5 else ""})',
        f'  @key-figures: {fig_count} figures ({", ".join(all_figs[:5])}{"..." if fig_count > 5 else ""})',
        "-->",
    ]
    return "\n".join(lines)


def build_section_annotation(sec: LatexSection) -> str:
    eqs = ", ".join(sec.equations) if sec.equations else "none"
    tables = ", ".join(sec.tables) if sec.tables else "none"
    figures = ", ".join(sec.figures) if sec.figures else "none"

    claim = sec.key_claim.replace("\n", " ")
    if len(claim) > 250:
        claim = claim[:250].rsplit(" ", 1)[0] + "..."

    data_sources = []
    body_lower = sec.body.lower()
    if "data_opt" in body_lower or "data_structures" in body_lower:
        data_sources.append("data_opt, data_structures")
    if "vix" in body_lower and "parquet" in body_lower:
        data_sources.append("vix")
    if "future_moments" in body_lower:
        data_sources.append("future_moments")
    if "slopes" in body_lower and "parquet" in body_lower:
        data_sources.append("slopes")
    ds = ", ".join(data_sources) if data_sources else "none"

    lines = [
        f"<!-- @section-type: {sec.section_type}",
        f"  @key-claim: {claim}",
        f"  @importance: {sec.importance}",
        f"  @data-source: {ds}",
        f"  @equations: {eqs}",
        f"  @tables: {tables}",
        f"  @figures: {figures}",
        "-->",
    ]
    return "\n".join(lines)


def convert(
    tex_path: Path,
    reading_guide_path: Path | None = None,
    output_path: Path | None = None,
    dry_run: bool = False,
    meta_overrides: dict[str, str] | None = None,
) -> str:
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    meta = _extract_metadata(tex)
    tex = _expand_user_macros(tex)

    body = _strip_latex_preamble(tex)
    body = _strip_comments(body)

    sections = _split_sections(body)
    for sec in sections:
        _extract_section_metadata(sec)

    if dry_run:
        lines = ["# Section Structure (dry run)", ""]
        for sec in sections:
            indent = "  " * (sec.level - 1)
            eqs = f" [eqs: {', '.join(sec.equations)}]" if sec.equations else ""
            tabs = f" [tabs: {', '.join(sec.tables)}]" if sec.tables else ""
            figs = f" [figs: {', '.join(sec.figures)}]" if sec.figures else ""
            lines.append(
                f"{indent}{'#' * (sec.level + 1)} {sec.title}  "
                f"({sec.section_type}, {sec.importance}){eqs}{tabs}{figs}"
            )
        return "\n".join(lines)

    parts: list[str] = []

    parts.append(f"# {meta.get('title', 'Paper')}\n")
    parts.append(build_document_metadata(meta, sections, meta_overrides))
    parts.append("")

    if reading_guide_path and reading_guide_path.exists():
        guide = reading_guide_path.read_text(encoding="utf-8")
        header_line = guide.split("\n")[0]
        if header_line.startswith("# "):
            guide = "\n".join(guide.split("\n")[1:])
        parts.append("---\n")
        parts.append("## AI Reading Guide\n")
        parts.append(guide.strip())
        parts.append("\n---\n")

    for sec in sections:
        parts.append(build_section_annotation(sec))
        parts.append("")
        hashes = "#" * (sec.level + 1)
        parts.append(f"{hashes} {sec.title}")
        parts.append("")

        converted = _convert_body(sec.body)
        if converted.strip():
            parts.append(converted)
        parts.append("")

    result = "\n".join(parts)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
        print(f"Wrote {len(result):,} chars to {output_path}")
    return result


def _parse_meta_args(raw: list[str] | None) -> dict[str, str]:
    """Parse --meta key=value pairs into a dict."""
    if not raw:
        return {}
    out: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--meta values must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LaTeX paper to annotated markdown for AI agents."
    )
    parser.add_argument("tex_file", type=Path, help="Path to the main .tex file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output markdown path (default: stdout)")
    parser.add_argument("--reading-guide", type=Path, default=None,
                        help="Path to reading-guide.md to prepend")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print section structure only; do not convert body text")
    parser.add_argument("--meta", nargs="*", metavar="KEY=VALUE",
                        help="Override @document-metadata fields "
                             "(e.g. --meta 'core-answer=...' 'datasets=...')")
    args = parser.parse_args()

    result = convert(
        tex_path=args.tex_file,
        reading_guide_path=args.reading_guide,
        output_path=args.output,
        dry_run=args.dry_run,
        meta_overrides=_parse_meta_args(args.meta),
    )

    if not args.output:
        print(result)


if __name__ == "__main__":
    main()
