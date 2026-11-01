from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "submission"
SOURCE = ROOT / "report" / "main_report_draft.md"
GENERATED_MD = OUTPUT_DIR / "submission_source.md"
RAW_DOCX = OUTPUT_DIR / "submission_raw.docx"
FINAL_DOCX = OUTPUT_DIR / "Quek Fu Sheng (A0149798B) FE5110 Term 2610.docx"

TITLE = "Do Data Driven Stock Clusters and Vine Copulas Improve Portfolio Tail Risk Forecasts"
SUBTITLE = "Evidence from a Provisional S&P 100 Research Universe"

TOC_HEADINGS = [
    "Abstract",
    "1. Introduction",
    "2. Literature Review",
    "3. Data and Research Design",
    "4. Methodology",
    "5. Forecast Evaluation and Inference",
    "6. Core Empirical Results",
    "7. ML Extension and Robustness",
    "8. Discussion, Limitations, and Extensions",
    "9. Conclusion",
    "References",
    "Appendix A Reproducibility and Data Availability",
]


def page_break() -> str:
    return """```{=openxml}
<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:r><w:br w:type="page"/></w:r></w:p>
```"""


def cover_markdown() -> str:
    return f"""::: {{custom-style=\"Title\"}}
{TITLE}
:::

::: {{custom-style=\"Subtitle\"}}
{SUBTITLE}
:::

::: {{custom-style=\"Cover Meta\"}}
FE5110 Financial Engineering Project  
Semester 1 AY 2026/2027  
Term 2610
:::

| Project information | Details |
|---|---|
| Name | Quek Fu Sheng |
| Student ID | A0149798B |
| Reporting status | Provisional research universe |

**Itemised workload**

| Work category | Hours |
|---|---:|
| Data collection and validation | 27 hours |
| Literature review | 21 hours |
| Programming and testing | 60 hours |
| Statistical analysis | 27 hours |
| Report writing and presentation | 21 hours |
| Project planning and reproducibility | 6 hours |
| **Total workload** | **162 hours** |

**Original contributions**

1. A controlled comparison of GICS and data-driven stock groupings that holds the daily equal-weight stock portfolio exactly constant across representations.
2. A leakage-free Gaussian-versus-R-vine comparison using identical marginal models, information sets, Monte Carlo draws, and prespecified loss functions.
3. An unsupervised-learning extension comparing hierarchical, spectral, and PCA-plus-k-means groupings under identical downstream risk evaluation.
4. An audit framework that stops on data or modelling errors and records security lifecycles, corporate actions, missing sessions, extreme returns, model fallbacks, seeded simulation, and reproducibility links.

{page_break()}
"""


def contents_markdown(page_map: dict[str, int] | None) -> str:
    lines = ["# Contents", "", "| Section | Page |", "|---|---:|"]
    for heading in TOC_HEADINGS:
        page = str(page_map[heading]) if page_map and heading in page_map else ""
        lines.append(f"| {heading} | {page} |")
    lines.extend(["", page_break(), ""])
    return "\n".join(lines)


def appendix_markdown() -> str:
    return f"""
# Appendix A Reproducibility and Data Availability

The reproducible project repository is available at
<https://github.com/Drakeral/stock-clustering-vine-copula-risk>. This report is
bound to artifact-producing Git commit `a40c000d5615ab1a2c6305cf0ad67136a33cd652`.
This hash identifies the committed code and report figures; the later packaging
commit only updates this reference and assembles the submission files.

The canonical environment uses Python 3.12 and is restored with
`uv sync --frozen`. The README documents the complete execution order, and
`scripts/run_pipeline.py --profile reproduce` is the single main program and
stops immediately if any derivation stage fails. The stages prepare and verify
the universe, validate provider inputs, construct
the return panels, validate portfolio arithmetic, estimate annual groupings and
monthly marginal/copula models, evaluate the core and exploratory forecasts,
build report artifacts, and verify every recorded file hash. The public test command is
`uv run python -m unittest discover -s tests -v`.

The repository tracks source code, fixed configuration, schemas, tests,
sanitised file inventories, audit summaries, report tables, and figures. Credentials,
licensed or provider-restricted raw market files, locally detailed event
records, and large derived panels are excluded. The exact public source
snapshots and permissible hashes are retained so an authorised user can
reconstruct the analysis after obtaining the required market data. The intended
2 January 2020 S&P 100 membership and GICS labels remain provisional because a
licensed point-in-time universe extract has not been obtained.

The report figures are produced from the passed report-artifact pipeline:
`scripts/build_report_artifacts.py` creates all report tables and figures,
including the annual dependence gaps, GICS-cluster composition heatmaps,
model-rank chart, VaR exception rates, and daily loss-versus-VaR chart. The run
manifest and `scripts/verify_artifact_lineage.py` connect the displayed results
to the fixed configuration and audited outputs using cryptographic file hashes.
"""


def build_markdown(page_map: dict[str, int] | None) -> None:
    manuscript = SOURCE.read_text(encoding="utf-8")
    manuscript = re.sub(
        r"\A# .*?\n\n\*Evidence from.*?\*\n\n", "", manuscript, count=1, flags=re.S
    )
    manuscript = re.sub(
        r"^(#{2,3}) ", lambda match: "#" * (len(match.group(1)) - 1) + " ", manuscript, flags=re.M
    )
    manuscript = manuscript.replace(r"\[", "$$").replace(r"\]", "$$")
    manuscript = manuscript.replace(r"\(", "$").replace(r"\)", "$")
    manuscript = manuscript.replace(
        "![Annual out-of-sample dependence gaps](../reports/figures/annual_dependence_gaps.svg)",
        "![](reports/figures/annual_dependence_gaps.svg){width=92%}",
    )
    manuscript = manuscript.replace(
        "![Average primary-score ranks](../reports/figures/core_model_score_ranks.svg)",
        "![](reports/figures/core_model_score_ranks.svg){width=92%}",
    )
    manuscript = manuscript.replace(
        "![Full-period VaR exception rates](../reports/figures/core_var_exception_rates.svg)",
        "![](reports/figures/core_var_exception_rates.svg){width=92%}",
    )
    manuscript = manuscript.replace(
        "![GICS composition of hierarchical clusters](../reports/figures/gics_hierarchical_contingency.svg)",
        "![](reports/figures/gics_hierarchical_contingency.svg){width=98%}",
    )
    manuscript = manuscript.replace(
        "![Daily realised losses and 99% VaR forecasts](../reports/figures/daily_losses_var_99.svg)",
        "![](reports/figures/daily_losses_var_99.svg){width=92%}",
    )
    full = cover_markdown() + contents_markdown(page_map) + manuscript + appendix_markdown()
    GENERATED_MD.write_text(full, encoding="utf-8")


def run_pandoc() -> None:
    subprocess.run(
        [
            "/opt/homebrew/bin/pandoc",
            str(GENERATED_MD),
            "--from=markdown+raw_attribute+tex_math_dollars",
            "--to=docx",
            f"--resource-path={ROOT}",
            "--standalone",
            "--citeproc",
            "--output",
            str(RAW_DOCX),
        ],
        cwd=ROOT,
        check=True,
    )


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), "D9D9D9")


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    """Keep a table row intact when Word paginates the document."""
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = tr_pr.find(qn("w:cantSplit"))
    if cant_split is None:
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.name = "Arial"
    run.font.size = Pt(9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), "Arial")
    r_fonts.set(qn("w:hAnsi"), "Arial")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "18")
    r_pr.extend([r_fonts, sz])
    t = OxmlElement("w:t")
    t.text = "1"
    r.extend([r_pr, t])
    fld.append(r)
    paragraph._p.append(fld)


def style_docx() -> None:
    doc = Document(RAW_DOCX)
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.72)
        section.bottom_margin = Inches(0.72)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        section.header_distance = Inches(0.3)
        section.footer_distance = Inches(0.35)
        footer = section.footer
        footer.is_linked_to_previous = False
        paragraph = footer.paragraphs[0]
        paragraph.clear()
        add_page_field(paragraph)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08

    for name, size, before, after in (
        ("Title", 21, 0, 10),
        ("Subtitle", 12, 0, 8),
        ("Heading 1", 15, 14, 6),
        ("Heading 2", 12, 10, 4),
        ("Heading 3", 10.5, 8, 3),
    ):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name != "Subtitle"
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    if "Cover Meta" not in [style.name for style in styles]:
        cover_meta = styles.add_style("Cover Meta", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cover_meta = styles["Cover Meta"]
    cover_meta.font.name = "Arial"
    cover_meta.font.size = Pt(10.5)
    cover_meta.font.color.rgb = RGBColor(0, 0, 0)
    cover_meta.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cover_meta.paragraph_format.space_after = Pt(10)

    in_references = False
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text == "References":
            in_references = True
        elif text == "Appendix A Reproducibility and Data Availability":
            in_references = False
        elif in_references:
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.0
            for run in paragraph.runs:
                run.font.size = Pt(9.5)
        if paragraph.style.name in {"Title", "Subtitle", "Cover Meta"}:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if text == "Contents":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if text.startswith("Figure "):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = False
            for run in paragraph.runs:
                run.italic = True
                run.font.size = Pt(9)
        if text.startswith("Table "):
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9.5)
        if text in {"References", "Appendix A Reproducibility and Data Availability"}:
            paragraph.paragraph_format.page_break_before = True
        if text.startswith("http") or "github.com/Drakeral" in text:
            paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    for table_index, table in enumerate(doc.tables):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        set_table_borders(table)
        if table.rows:
            repeat_table_header(table.rows[0])
        for row_index, row in enumerate(table.rows):
            prevent_row_split(row)
            for cell_index, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                set_cell_margins(cell)
                if row_index == 0:
                    set_cell_shading(cell, "1F4E78")
                elif row_index % 2 == 0:
                    set_cell_shading(cell, "EAF2F8")
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(1.5)
                    paragraph.paragraph_format.space_before = Pt(1.5)
                    paragraph.paragraph_format.line_spacing = 1.0
                    if cell_index > 0 and len(cell.text) < 30:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.name = "Arial"
                        run.font.size = Pt(8.5 if table_index > 1 else 9)
                        if row_index == 0:
                            run.font.color.rgb = RGBColor(255, 255, 255)
                            run.bold = True

    for shape in doc.inline_shapes:
        if shape.width > Inches(6.4):
            ratio = Inches(6.4) / shape.width
            shape.width = Inches(6.4)
            shape.height = int(shape.height * ratio)

    props = doc.core_properties
    props.title = TITLE
    props.subject = "FE5110 Financial Engineering Project"
    props.author = "Quek Fu Sheng"
    props.keywords = "stock clustering, vine copula, VaR, expected shortfall, financial risk"
    props.comments = "Provisional S&P 100 research universe"
    doc.save(FINAL_DOCX)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toc-json", type=Path)
    args = parser.parse_args()
    page_map = None
    if args.toc_json:
        page_map = json.loads(args.toc_json.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_markdown(page_map)
    run_pandoc()
    style_docx()
    print(FINAL_DOCX)


if __name__ == "__main__":
    main()
