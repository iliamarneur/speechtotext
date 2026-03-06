"""Generation de PDF a partir de contenu Markdown (analyses LLM)."""

import re
from fpdf import FPDF


class AnalysisPDF(FPDF):
    """PDF stylise pour les exports d'analyses."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "Audio-to-Knowledge", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_title(self, text):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, self._sanitize(text))
        self.ln(8)
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(8)

    def add_section(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(50, 50, 50)
        self.ln(4)
        self.cell(0, 10, self._sanitize(text))
        self.ln(8)

    def add_subsection(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(70, 70, 70)
        self.ln(2)
        self.cell(0, 8, self._sanitize(text))
        self.ln(6)

    def add_markdown(self, md_text):
        """Parse et rend du Markdown basique en PDF."""
        lines = md_text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Headers
            if stripped.startswith("### "):
                self.add_subsection(self._clean(stripped[4:]))
            elif stripped.startswith("## "):
                self.add_section(self._clean(stripped[3:]))
            elif stripped.startswith("# "):
                self.add_title(self._clean(stripped[2:]))
            # Horizontal rule
            elif stripped == "---":
                self.ln(4)
                self.set_draw_color(200, 200, 200)
                self.set_line_width(0.3)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
                self.ln(6)
            # Table
            elif stripped.startswith("|") and stripped.endswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                i -= 1  # will be incremented at end
                self._render_table(table_lines)
            # Checkbox (must check before bullet)
            elif stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
                checked = stripped.startswith("- [x] ")
                text = self._clean(stripped[6:])
                prefix = "[x] " if checked else "[ ] "
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 40)
                self.cell(12, 6, prefix)
                self.multi_cell(self.w - self.l_margin - self.r_margin - 12, 6, text)
                self.ln(1)
            # Bullet list
            elif stripped.startswith("- ") or stripped.startswith("* "):
                text = self._clean(stripped[2:])
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 40)
                self.cell(6, 6, "-")
                self.multi_cell(self.w - self.l_margin - self.r_margin - 6, 6, text)
                self.ln(1)
            # Numbered list
            elif re.match(r"^\d+\.\s", stripped):
                match = re.match(r"^(\d+\.)\s(.+)", stripped)
                if match:
                    num, text = match.group(1), self._clean(match.group(2))
                    self.set_font("Helvetica", "B", 10)
                    self.set_text_color(40, 40, 40)
                    self.cell(10, 6, num)
                    self.set_font("Helvetica", "", 10)
                    self.multi_cell(self.w - self.l_margin - self.r_margin - 10, 6, text)
                    self.ln(1)
            # Empty line
            elif not stripped:
                self.ln(3)
            # Normal text
            else:
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 40)
                self.multi_cell(0, 6, self._clean(stripped))
                self.ln(1)

            i += 1

    def _clean(self, text):
        """Remove Markdown formatting and sanitize for Latin-1."""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        return self._sanitize(text)

    @staticmethod
    def _sanitize(text):
        """Replace Unicode chars not in Latin-1 with safe equivalents."""
        replacements = {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
            "\u2023": ">", "\u25cf": "-", "\u25cb": "o", "\u2713": "v",
            "\u2717": "x", "\u2610": "[ ]", "\u2611": "[x]", "\u2612": "[x]",
            "\u00a0": " ", "\u200b": "", "\u2192": "->", "\u2190": "<-",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Fallback: replace any remaining non-Latin-1 chars
        try:
            text.encode("latin-1")
        except UnicodeEncodeError:
            text = text.encode("latin-1", errors="replace").decode("latin-1")
        return text

    def _render_table(self, rows):
        """Render a Markdown table."""
        if len(rows) < 2:
            return

        # Parse cells
        parsed = []
        for row in rows:
            cells = [c.strip() for c in row.split("|")[1:-1]]
            parsed.append(cells)

        # Skip separator row
        data_rows = [r for r in parsed if not all(re.match(r"^[\s\-:]+$", c) for c in r)]
        if not data_rows:
            return

        num_cols = max(len(r) for r in data_rows)
        col_width = (self.w - self.l_margin - self.r_margin) / max(num_cols, 1)

        for row_idx, cells in enumerate(data_rows):
            # Pad cells
            while len(cells) < num_cols:
                cells.append("")

            if row_idx == 0:
                # Header row
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(59, 130, 246)
                self.set_text_color(255, 255, 255)
            else:
                self.set_font("Helvetica", "", 9)
                self.set_text_color(40, 40, 40)
                if row_idx % 2 == 0:
                    self.set_fill_color(240, 245, 255)
                else:
                    self.set_fill_color(255, 255, 255)

            for cell in cells:
                self.cell(col_width, 7, self._clean(cell)[:40], border=1, fill=True)
            self.ln()

        self.ln(4)


def generate_analysis_pdf(title: str, content: str, filename: str = "audio") -> bytes:
    """Genere un PDF a partir d'une analyse."""
    pdf = AnalysisPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.add_title(title)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, AnalysisPDF._sanitize(f"Source : {filename}"))
    pdf.ln(8)
    pdf.add_markdown(content)
    return bytes(pdf.output())


def generate_all_analyses_pdf(filename: str, analyses: dict) -> bytes:
    """Genere un PDF avec toutes les analyses."""
    pdf = AnalysisPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.add_title(f"Analyses — {filename}")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, AnalysisPDF._sanitize(f"Source : {filename}"))
    pdf.ln(10)

    for label, content in analyses.items():
        if content:
            pdf.add_section(label)
            pdf.add_markdown(content)
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(6)

    return bytes(pdf.output())
