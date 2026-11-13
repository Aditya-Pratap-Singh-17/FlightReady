"""Build the two submission PDFs from their checked-in Markdown sources."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    PageBreak, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GREEN = colors.HexColor("#173A32")
TEAL = colors.HexColor("#267A6B")
PALE = colors.HexColor("#EAF3F0")
GRAY = colors.HexColor("#5E6B66")


def styles(summary=False):
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="TitleFlight", parent=s["Title"], fontName="Helvetica-Bold", fontSize=24,
        leading=28, textColor=GREEN, alignment=TA_CENTER, spaceAfter=16))
    s.add(ParagraphStyle(name="SubTitle", parent=s["Normal"], fontSize=10, leading=14, textColor=GRAY,
        alignment=TA_CENTER, spaceAfter=20))
    s.add(ParagraphStyle(name="H1Flight", parent=s["Heading1"], fontName="Helvetica-Bold", fontSize=15,
        leading=19, textColor=GREEN, spaceBefore=12, spaceAfter=7))
    s.add(ParagraphStyle(name="H2Flight", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=5))
    s.add(ParagraphStyle(name="BodyFlight", parent=s["BodyText"], fontName="Helvetica", fontSize=9,
        leading=13, textColor=colors.HexColor("#1F2925"), spaceAfter=6))
    s.add(ParagraphStyle(name="BulletFlight", parent=s["BodyFlight"], leftIndent=14, firstLineIndent=-8,
        bulletIndent=5, spaceAfter=3))
    s.add(ParagraphStyle(name="CaptionFlight", parent=s["BodyFlight"], fontSize=8, leading=10,
        textColor=GRAY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8))
    s.add(ParagraphStyle(name="SmallFlight", parent=s["BodyFlight"], fontSize=7.4, leading=9.4, spaceAfter=2))
    s.add(ParagraphStyle(name="TableHeader", parent=s["SmallFlight"], fontName="Helvetica-Bold", textColor=colors.white))
    if summary:
        s["TitleFlight"].fontSize=18; s["TitleFlight"].leading=21; s["TitleFlight"].spaceAfter=8
        s["H1Flight"].fontSize=10.5; s["H1Flight"].leading=12.5; s["H1Flight"].spaceBefore=6; s["H1Flight"].spaceAfter=3
        s["BodyFlight"].fontSize=7.4; s["BodyFlight"].leading=9.4; s["BodyFlight"].spaceAfter=3
        s["BulletFlight"].fontSize=7.2; s["BulletFlight"].leading=9; s["BulletFlight"].spaceAfter=1
    return s


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B7C9C3")); canvas.line(0.65*inch, 0.52*inch, 7.85*inch, 0.52*inch)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GRAY)
    canvas.drawString(0.65*inch, 0.34*inch, "FlightReady - Academic Research Submission")
    canvas.drawRightString(7.85*inch, 0.34*inch, f"Page {doc.page}")
    canvas.restoreState()


def parse_table(lines, s, compact=False):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        style = s["TableHeader"] if not rows else s["SmallFlight" if compact else "BodyFlight"]
        rows.append([Paragraph(inline(c), style) for c in cells])
    widths = [7.15*inch/len(rows[0])] * len(rows[0])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), GREEN), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B7C9C3")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
        ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return table


def markdown_story(path: Path, *, summary=False):
    s = styles(summary=summary); lines = path.read_text(encoding="utf-8").splitlines(); story=[]; i=0
    while i < len(lines):
        line = lines[i].strip()
        if not line: story.append(Spacer(1, 3)); i += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), s["TitleFlight"]));
            if not summary: story.append(Paragraph("Final research report | 12 November 2026", s["SubTitle"]))
        elif line.startswith("## "):
            if line.startswith("## 12. Statistical analysis"):
                story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]), s["H1Flight"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), s["H2Flight"]))
        elif line.startswith("!["):
            match=re.match(r"!\[.*?\]\((.*?)\)", line)
            if match:
                img=(path.parent/match.group(1)).resolve(); im=Image(str(img)); im._restrictSize(6.8*inch, 4.25*inch)
                story.append(KeepTogether([Spacer(1,5), im]))
        elif line.startswith("|"):
            block=[]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            story.append(parse_table(block, s, compact=not summary)); story.append(Spacer(1,7)); continue
        elif re.match(r"^[-*] ", line):
            story.append(Paragraph("- " + inline(line[2:]), s["BulletFlight"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline(line), s["BulletFlight"]))
        elif line.startswith("**Figure"):
            story.append(Paragraph(inline(line), s["CaptionFlight"]))
        else:
            text=line
            while i+1 < len(lines) and lines[i+1].strip() and not re.match(r"^(#|\||[-*] |\d+\. |!\[)", lines[i+1].strip()):
                i += 1; text += " " + lines[i].strip()
            story.append(Paragraph(inline(text), s["BodyFlight"]))
        i += 1
    return story


def build(source: str, output: str, *, summary=False):
    out=DOCS/output
    margin=0.48*inch if summary else 0.65*inch
    frame=Frame(margin, 0.58*inch, 8.5*inch-2*margin, 9.82*inch, id="normal", leftPadding=3, rightPadding=3, topPadding=3, bottomPadding=3)
    doc=BaseDocTemplate(str(out), pagesize=letter, leftMargin=.65*inch, rightMargin=.65*inch,
        topMargin=.55*inch, bottomMargin=.65*inch, title=source.replace("_", " "))
    doc.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=page_number))
    doc.build(markdown_story(DOCS/source, summary=summary))


if __name__ == "__main__":
    build("FlightReady_Improved_Research_Report.md", "FlightReady_Improved_Research_Report.pdf")
    build("FlightReady_Improved_Professor_Summary.md", "FlightReady_Improved_Professor_Summary.pdf", summary=True)
