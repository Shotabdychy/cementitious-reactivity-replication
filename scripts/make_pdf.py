"""PART 5 -- render docs/REPLICATION_REPORT.md to a submittable PDF.

Markdown -> HTML (python-markdown, tables + fenced code) -> PDF (fpdf2 write_html).
Unicode text comes from the system Arial faces so that superscripts, arrows and Greek letters
survive.

usage: python make_pdf.py [--md docs/REPLICATION_REPORT.md] [--out docs/REPLICATION_REPORT.pdf]
"""
import os
import re
import sys
import argparse

import markdown
from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--md", default=os.path.join(ROOT, "docs", "REPLICATION_REPORT.md"))
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "REPLICATION_REPORT.pdf"))
ap.add_argument("--base", default=ROOT)
args = ap.parse_args()

FONTS = r"C:\Windows\Fonts"
src = open(args.md, encoding="utf-8").read()

# page breaks: a line containing only <!--pagebreak--> becomes a real break
src = src.replace("<!--pagebreak-->", '<p class="pb"></p>')

html = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

# fpdf2's HTML subset: unwrap <pre><code> to a single <pre>, and resolve image paths
html = re.sub(r"<pre><code[^>]*>", "<pre>", html)
html = html.replace("</code></pre>", "</pre>")
# fpdf2 draws an <img> at its pixel size unless a width is given, which would overflow A4.
# 430 px at the renderer's 72 dpi is about 152 mm, which leaves room for two wide panels a page.
html = re.sub(r'<img([^>]*?)src="([^"]+)"',
              lambda m: f'<img{m.group(1)}src="{os.path.join(args.base, m.group(2))}" '
                        f'width="430"', html)
# Arial has no subscript digits or relational glyphs; fall back to plain characters in the PDF
for a, b in (("₂", "2"), ("₃", "3"), ("₄", "4"), ("⋈", " joined with ")):
    html = html.replace(a, b)
# fpdf2 renders <th>/<td> but ignores most attributes; strip alignment rows markdown leaves
html = html.replace("<thead>\n", "<thead>").replace("align=\"right\"", "").replace(
    "align=\"left\"", "").replace("align=\"center\"", "")
# fpdf2 refuses nested tags inside a table cell, so inline code spans are flattened there only
html = re.sub(r"<(td|th)>.*?</\1>",
              lambda m: m.group(0).replace("<code>", "").replace("</code>", "")
                                  .replace("<strong>", "").replace("</strong>", "")
                                  .replace("<em>", "").replace("</em>", ""),
              html, flags=re.S)


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "", 7.5)
        self.set_text_color(110, 110, 105)
        self.cell(0, 6, "Replication of Mahjoubi et al., Communications Materials 6:99 (2025)",
                  align="L")
        self.ln(7)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-13)
        self.set_font("Arial", "", 7.5)
        self.set_text_color(110, 110, 105)
        self.cell(0, 6, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.add_font("Arial", "", os.path.join(FONTS, "arial.ttf"))
pdf.add_font("Arial", "B", os.path.join(FONTS, "arialbd.ttf"))
pdf.add_font("Arial", "I", os.path.join(FONTS, "ariali.ttf"))
pdf.add_font("Arial", "BI", os.path.join(FONTS, "arialbi.ttf"))
pdf.add_font("Courier New", "", os.path.join(FONTS, "cour.ttf"))
pdf.add_font("Courier New", "B", os.path.join(FONTS, "courbd.ttf"))
pdf.set_margins(14, 12, 12)
pdf.set_auto_page_break(True, margin=14)
pdf.set_font("Arial", "", 8.5)
pdf.add_page()

# split on the page-break marker so each chunk is written, then a real page break inserted
chunks = html.split('<p class="pb"></p>')
for i, chunk in enumerate(chunks):
    pdf.write_html(chunk, tag_styles={}, font_family="Arial")
    if i < len(chunks) - 1:
        pdf.add_page()

pdf.output(args.out)
print(f"wrote {args.out} ({pdf.page_no()} pages, {os.path.getsize(args.out) / 1024:.0f} KB)")
