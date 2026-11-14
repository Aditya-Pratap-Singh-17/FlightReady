"""Render the independent validation reports to reviewable PDFs."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, PageBreak, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "experiments" / "independent_validation" / "final_reports"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#246B8E")
PALE = colors.HexColor("#EAF2F7")
GRAY = colors.HexColor("#59636B")


def styles(ieee=False):
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="DocTitle", parent=s["Title"], fontName="Helvetica-Bold", fontSize=18 if ieee else 23, leading=22 if ieee else 27, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
    s.add(ParagraphStyle(name="H1Doc", parent=s["Heading1"], fontName="Helvetica-Bold", fontSize=12.5 if ieee else 15, leading=15 if ieee else 19, textColor=NAVY, spaceBefore=10, spaceAfter=6))
    s.add(ParagraphStyle(name="H2Doc", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=10.5 if ieee else 12, leading=13 if ieee else 15, textColor=BLUE, spaceBefore=8, spaceAfter=4))
    s.add(ParagraphStyle(name="BodyDoc", parent=s["BodyText"], fontName="Helvetica", fontSize=8.2 if ieee else 9, leading=10.5 if ieee else 12.5, textColor=colors.HexColor("#20282E"), spaceAfter=5, alignment=4 if ieee else 0))
    s.add(ParagraphStyle(name="BulletDoc", parent=s["BodyDoc"], leftIndent=12, firstLineIndent=-7, bulletIndent=4, spaceAfter=2))
    s.add(ParagraphStyle(name="SmallDoc", parent=s["BodyDoc"], fontSize=6.3, leading=7.5, spaceAfter=1))
    s.add(ParagraphStyle(name="TableHead", parent=s["SmallDoc"], fontName="Helvetica-Bold", textColor=colors.white))
    return s


def inline(value):
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    value = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", value)
    return value


def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#B8C5CE")); canvas.line(.62*inch, .5*inch, 7.88*inch, .5*inch)
    canvas.setFont("Helvetica", 7); canvas.setFillColor(GRAY); canvas.drawString(.62*inch, .32*inch, "FlightReady independent scientific validation")
    canvas.drawRightString(7.88*inch, .32*inch, f"Page {doc.page}"); canvas.restoreState()


def table(block, s, width):
    rows=[]
    for line in block:
        cells=[c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells): continue
        style=s["TableHead"] if not rows else s["SmallDoc"]
        rows.append([Paragraph(inline(c), style) for c in cells])
    t=Table(rows, colWidths=[width/len(rows[0])]*len(rows[0]), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#B8C5CE")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,PALE]),("LEFTPADDING",(0,0),(-1,-1),2.5),("RIGHTPADDING",(0,0),(-1,-1),2.5),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    return t


def story(path, ieee=False):
    s=styles(ieee); lines=path.read_text(encoding="utf-8").splitlines(); result=[]; i=0; width=7.1*inch
    while i < len(lines):
        line=lines[i].strip()
        if not line: result.append(Spacer(1,2)); i+=1; continue
        if line.startswith("# "):
            result.append(Paragraph(inline(line[2:]),s["DocTitle"])); result.append(Paragraph("Independent validation | 12 November 2026", ParagraphStyle(name=f"Sub{i}", parent=s["BodyDoc"], alignment=TA_CENTER, textColor=GRAY, spaceAfter=12)))
        elif line.startswith("## "):
            if line in {"## References", "## 8. Statistical analysis", "## V. Results"}: result.append(PageBreak())
            result.append(Paragraph(inline(line[3:]),s["H1Doc"]))
        elif line.startswith("### "): result.append(Paragraph(inline(line[4:]),s["H2Doc"]))
        elif line.startswith("!["):
            match=re.match(r"!\[.*?\]\((.*?)\)",line)
            if match:
                image=(path.parent/match.group(1)).resolve(); item=Image(str(image)); item._restrictSize(6.8*inch,4.4*inch); result.append(KeepTogether([Spacer(1,4),item,Spacer(1,5)]))
        elif line.startswith("|"):
            block=[]
            while i<len(lines) and lines[i].strip().startswith("|"): block.append(lines[i]); i+=1
            result.extend([table(block,s,width),Spacer(1,6)]); continue
        elif re.match(r"^[-*] ",line): result.append(Paragraph("- "+inline(line[2:]),s["BulletDoc"]))
        elif re.match(r"^\d+\. ",line): result.append(Paragraph(inline(line),s["BulletDoc"]))
        else:
            text=line
            while i+1<len(lines) and lines[i+1].strip() and not re.match(r"^(#|\||[-*] |\d+\. |!\[)",lines[i+1].strip()): i+=1; text+=" "+lines[i].strip()
            result.append(Paragraph(inline(text),s["BodyDoc"]))
        i+=1
    return result


def build(source, output, ieee=False):
    frame=Frame(.62*inch,.58*inch,7.26*inch,9.82*inch,id="main",leftPadding=3,rightPadding=3,topPadding=3,bottomPadding=3)
    doc=BaseDocTemplate(str(REPORTS/output),pagesize=letter,title=source.replace("_"," "),author="FlightReady research project")
    doc.addPageTemplates(PageTemplate(id="main",frames=frame,onPage=footer)); doc.build(story(REPORTS/source,ieee=ieee))


if __name__ == "__main__":
    build("FlightReady_Final_Independent_Validation_Report.md","FlightReady_Final_Independent_Validation_Report.pdf")
    build("IEEE_Publication_Readiness_Review.md","IEEE_Publication_Readiness_Review.pdf")
    build("FlightReady_IEEE_Conference_Paper_Draft.md","FlightReady_IEEE_Conference_Paper_Draft.pdf",ieee=True)
