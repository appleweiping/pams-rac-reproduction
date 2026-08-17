"""Render the non-empirical ICASSP method specification figure.

The scene contains no experiment values.  It is deliberately labelled as a
proposed, synchronization-pending specification.  The editable source of the
same scene is ``icassp_framework.drawio``.  This renderer provides deterministic
SVG/PDF exports and a clean PNG preview when the draw.io desktop CLI is absent.
"""

from __future__ import annotations

import html
import math
import shutil
import subprocess
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
WIDTH = 2000
HEIGHT = 740
PDF_WIDTH_PT = 7.0 * 72.0
SCALE = PDF_WIDTH_PT / WIDTH
PDF_HEIGHT_PT = HEIGHT * SCALE

INK = "#25313C"
MUTED = "#5B6570"
BLUE = "#4477AA"
BLUE_FILL = "#EAF1F8"
GREEN = "#228833"
GREEN_FILL = "#EAF6ED"
AMBER = "#CC9900"
AMBER_FILL = "#FFF6D5"
PURPLE = "#AA3377"
PURPLE_FILL = "#F8EAF2"
PURPLE_TEXT = "#6F1D4E"
CYAN = "#2288AA"
CYAN_FILL = "#E5F5F8"
GRAY = "#747474"
GRAY_FILL = "#F2F2F2"
RED = "#A63D40"
WHITE = "#FFFFFF"


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\times.ttf")
    bold = Path(r"C:\Windows\Fonts\timesbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("RACSerif", str(regular)))
        pdfmetrics.registerFont(TTFont("RACSerif-Bold", str(bold)))
        return "RACSerif", "RACSerif-Bold"
    return "Times-Roman", "Times-Bold"


class SVGRenderer:
    def __init__(self) -> None:
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="7in" height="{PDF_HEIGHT_PT / 72.0:.4f}in" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
            'aria-label="Proposed identity-indexed local tempo routing specification">',
            "<defs>",
            '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
            'orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 12 6 L 0 12 z" fill="{INK}"/></marker>',
            '<marker id="arrow-green" markerWidth="12" markerHeight="12" refX="10" refY="6" '
            'orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 12 6 L 0 12 z" fill="{GREEN}"/></marker>',
            '<marker id="arrow-purple" markerWidth="12" markerHeight="12" refX="10" refY="6" '
            'orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 12 6 L 0 12 z" fill="{PURPLE}"/></marker>',
            '<pattern id="fastPattern" width="16" height="16" patternUnits="userSpaceOnUse">'
            f'<rect width="16" height="16" fill="{AMBER_FILL}"/>'
            f'<path d="M0 12 L12 0 M4 16 L16 4" stroke="{AMBER}" stroke-width="2"/></pattern>',
            '<pattern id="midPattern" width="12" height="12" patternUnits="userSpaceOnUse">'
            f'<rect width="12" height="12" fill="{BLUE_FILL}"/>'
            f'<path d="M0 6 H12" stroke="{BLUE}" stroke-width="2"/></pattern>',
            '<pattern id="slowPattern" width="14" height="14" patternUnits="userSpaceOnUse">'
            f'<rect width="14" height="14" fill="{GREEN_FILL}"/>'
            f'<circle cx="4" cy="4" r="1.8" fill="{GREEN}"/></pattern>',
            "</defs>",
            f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        ]

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = WHITE,
        stroke: str = INK,
        sw: float = 3,
        radius: float = 18,
        dash: str | None = None,
        pattern: str | None = None,
    ) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        actual_fill = f"url(#{pattern})" if pattern else fill
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{actual_fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def line(
        self,
        points: list[tuple[float, float]],
        stroke: str = INK,
        sw: float = 4,
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        pts = " ".join(f"{x},{y}" for x, y in points)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        marker = "arrow-green" if stroke == GREEN else "arrow-purple" if stroke == PURPLE else "arrow"
        a = f' marker-end="url(#{marker})"' if arrow else ""
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d}{a}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str = WHITE,
        stroke: str = INK,
        sw: float = 3,
    ) -> None:
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        lines: list[str] | str,
        size: float = 30,
        bold: bool = False,
        color: str = INK,
        anchor: str = "middle",
        leading: float = 1.18,
    ) -> None:
        if isinstance(lines, str):
            lines = [lines]
        weight = "700" if bold else "400"
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}">'
        )
        for idx, line in enumerate(lines):
            dy = "0" if idx == 0 else f"{size * leading:.1f}"
            self.parts.append(
                f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>'
            )
        self.parts.append("</text>")

    def finish(self) -> bytes:
        self.parts.append("</svg>")
        return ("\n".join(self.parts) + "\n").encode("utf-8")


class PDFRenderer:
    def __init__(self, path: Path) -> None:
        self.regular_font, self.bold_font = register_fonts()
        self.canvas = canvas.Canvas(
            str(path),
            pagesize=(PDF_WIDTH_PT, PDF_HEIGHT_PT),
            invariant=1,
            initialFontName=self.regular_font,
            initialFontSize=10,
        )
        self.canvas.setTitle("Proposed identity-indexed local tempo routing specification")
        self.canvas.setAuthor("RAC paper artifact generator")

    @staticmethod
    def _rgb(hex_color: str) -> tuple[float, float, float]:
        value = hex_color.lstrip("#")
        return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        return x * SCALE, PDF_HEIGHT_PT - y * SCALE

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = WHITE,
        stroke: str = INK,
        sw: float = 3,
        radius: float = 18,
        dash: str | None = None,
        pattern: str | None = None,
    ) -> None:
        del pattern
        c = self.canvas
        c.saveState()
        c.setFillColorRGB(*self._rgb(fill))
        c.setStrokeColorRGB(*self._rgb(stroke))
        c.setLineWidth(sw * SCALE)
        if dash:
            c.setDash([float(v) * SCALE for v in dash.split()])
        px, py_top = self._xy(x, y)
        c.roundRect(
            px,
            py_top - h * SCALE,
            w * SCALE,
            h * SCALE,
            radius * SCALE,
            stroke=1,
            fill=1,
        )
        c.restoreState()

    def line(
        self,
        points: list[tuple[float, float]],
        stroke: str = INK,
        sw: float = 4,
        dash: str | None = None,
        arrow: bool = False,
    ) -> None:
        c = self.canvas
        c.saveState()
        c.setStrokeColorRGB(*self._rgb(stroke))
        c.setFillColorRGB(*self._rgb(stroke))
        c.setLineWidth(sw * SCALE)
        c.setLineJoin(1)
        c.setLineCap(1)
        if dash:
            c.setDash([float(v) * SCALE for v in dash.split()])
        path = c.beginPath()
        x0, y0 = self._xy(*points[0])
        path.moveTo(x0, y0)
        for point in points[1:]:
            px, py = self._xy(*point)
            path.lineTo(px, py)
        c.drawPath(path, stroke=1, fill=0)
        if arrow and len(points) >= 2:
            (x1, y1), (x2, y2) = points[-2], points[-1]
            angle = math.atan2(-(y2 - y1), x2 - x1)
            tip_x, tip_y = self._xy(x2, y2)
            length = 14 * SCALE
            half = 6 * SCALE
            bx = tip_x - length * math.cos(angle)
            by = tip_y - length * math.sin(angle)
            perp_x = -math.sin(angle) * half
            perp_y = math.cos(angle) * half
            p = c.beginPath()
            p.moveTo(tip_x, tip_y)
            p.lineTo(bx + perp_x, by + perp_y)
            p.lineTo(bx - perp_x, by - perp_y)
            p.close()
            c.drawPath(p, stroke=0, fill=1)
        c.restoreState()

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str = WHITE,
        stroke: str = INK,
        sw: float = 3,
    ) -> None:
        c = self.canvas
        c.saveState()
        c.setFillColorRGB(*self._rgb(fill))
        c.setStrokeColorRGB(*self._rgb(stroke))
        c.setLineWidth(sw * SCALE)
        x, y = self._xy(cx, cy)
        c.circle(x, y, r * SCALE, stroke=1, fill=1)
        c.restoreState()

    def text(
        self,
        x: float,
        y: float,
        lines: list[str] | str,
        size: float = 30,
        bold: bool = False,
        color: str = INK,
        anchor: str = "middle",
        leading: float = 1.18,
    ) -> None:
        if isinstance(lines, str):
            lines = [lines]
        c = self.canvas
        c.saveState()
        c.setFillColorRGB(*self._rgb(color))
        c.setFont(self.bold_font if bold else self.regular_font, size * SCALE)
        for idx, line in enumerate(lines):
            px, py = self._xy(x, y + idx * size * leading)
            if anchor == "middle":
                c.drawCentredString(px, py, line)
            elif anchor == "end":
                c.drawRightString(px, py, line)
            else:
                c.drawString(px, py, line)
        c.restoreState()

    def finish(self) -> None:
        self.canvas.showPage()
        self.canvas.save()


def draw_scene(r: SVGRenderer | PDFRenderer) -> None:
    # Status is a scope-control badge, not a figure title.
    r.rect(1440, 14, 520, 50, fill=GRAY_FILL, stroke=GRAY, sw=2.5, radius=10)
    r.text(
        1700,
        48,
        "PROPOSED  •  SYNC-REQUIRED",
        size=32,
        bold=True,
        color=RED,
    )

    # Connectors are drawn before the modules so endpoints stay visually clean.
    r.line([(324, 258), (372, 258)], arrow=True)
    r.line([(628, 258), (682, 258)], arrow=True)
    r.line([(918, 258), (972, 258)], arrow=True)
    r.line([(1240, 258), (1288, 258)], arrow=True)
    r.line([(1612, 300), (1612, 426), (1245, 426), (1245, 476)], arrow=True)
    r.line([(1370, 542), (1390, 542)], arrow=True)
    r.line([(1625, 542), (1650, 542)], arrow=True)
    r.line([(1810, 542), (1815, 542)], arrow=True)

    # Dashed identity-indexed state path is distinct from shared-parameter flow.
    r.line(
        [(322, 441), (346, 441), (346, 632), (1730, 632), (1730, 612)],
        stroke=GREEN,
        sw=3,
        dash="13 9",
        arrow=True,
    )
    r.text(
        1005,
        626,
        "identity i  •  state sᵢ preserved",
        size=32,
        bold=True,
        color=GREEN,
    )

    # Supplied tracks: external, perception-side artifact boundary.
    r.rect(40, 96, 284, 372, fill=WHITE, stroke=GRAY, sw=3, radius=20, dash="13 9")
    r.text(182, 122, ["SUPPLIED", "TRACKS"], size=34, bold=True, color=MUTED, leading=1.0)
    for y, label, phase in [(200, "X₁, m₁", 0), (270, "X₂, m₂", 16), (340, "Xₙ, mₙ", 8)]:
        r.rect(70, y, 224, 50, fill=GRAY_FILL, stroke=GRAY, sw=2, radius=10)
        r.text(122, y + 35, label, size=32, bold=True, color=INK)
        pts = []
        for xx in range(158, 278, 12):
            yy = y + 27 - 10 * math.sin((xx + phase) / 24.0)
            pts.append((xx, yy))
        r.line(pts, stroke=MUTED, sw=2.5)
    r.text(
        182,
        425,
        ["EXTERNAL", "SUPERVISION"],
        size=32,
        leading=1.0,
        color=MUTED,
    )

    # Shared encoder with explicit per-identity state chip.
    r.rect(372, 126, 256, 264, fill=BLUE_FILL, stroke=BLUE, sw=4, radius=20)
    r.rect(392, 145, 216, 42, fill=BLUE, stroke=BLUE, sw=1, radius=8)
    r.text(500, 176, "SHARED  θ", size=32, bold=True, color=WHITE)
    r.text(500, 226, ["MASKED", "ENCODER Eθ"], size=36, bold=True)
    r.rect(410, 310, 180, 52, fill=GREEN_FILL, stroke=GREEN, sw=3, radius=8, dash="10 7")
    r.text(500, 346, "STATE sᵢ", size=32, bold=True, color=GREEN)

    # Overlapping local windows.
    r.rect(682, 126, 236, 264, fill=CYAN_FILL, stroke=CYAN, sw=4, radius=20)
    r.text(800, 156, ["OVERLAP", "W(i,ℓ)"], size=36, bold=True, color=CYAN)
    for label, x in zip(["ℓ=1", "ℓ=2", "ℓ=L"], [702, 774, 846]):
        r.rect(x, 232, 52, 72, fill=WHITE, stroke=CYAN, sw=2.5, radius=6)
        r.text(x + 26, 278, label, size=32, bold=True, color=CYAN)
    r.text(800, 350, "PER TRACK", size=32, bold=True, color=INK)

    # Local masked period evidence.
    r.rect(972, 126, 268, 264, fill=AMBER_FILL, stroke=AMBER, sw=4, radius=20)
    r.text(1106, 156, ["MASKED", "ACF + FFT"], size=36, bold=True, color=INK)
    r.text(1106, 250, "p̂(i,ℓ) • γ(i,ℓ)", size=32, bold=True)
    r.line([(1010, 315), (1044, 290), (1080, 329), (1116, 270), (1152, 310), (1196, 284)], stroke=AMBER, sw=4)
    r.text(1106, 365, "LOCAL TEMPO", size=32, bold=True, color=MUTED)

    # Soft router and three shared experts, using texture as well as color.
    r.rect(1288, 88, 324, 320, fill=WHITE, stroke=PURPLE, sw=4, radius=22)
    r.rect(1310, 106, 280, 42, fill=PURPLE, stroke=PURPLE, sw=1, radius=8)
    r.text(1450, 138, "SHARED  H(k)", size=32, bold=True, color=WHITE)
    r.circle(1350, 226, 48, fill=PURPLE_FILL, stroke=PURPLE, sw=3)
    r.text(1350, 218, "SOFT", size=32, bold=True, color=PURPLE_TEXT)
    r.text(1350, 250, "gᵢℓₖ", size=32, bold=True, color=PURPLE_TEXT)
    expert_specs = [
        (1412, 150, "FAST", AMBER_FILL, AMBER, "fastPattern", None),
        (1412, 215, "MEDIUM", BLUE_FILL, BLUE, "midPattern", "12 5"),
        (1412, 280, "SLOW", GREEN_FILL, GREEN, "slowPattern", "3 5"),
    ]
    for x, y, label, fill, stroke, pattern, dash in expert_specs:
        r.rect(
            x,
            y,
            160,
            54,
            fill=fill,
            stroke=stroke,
            sw=3,
            radius=9,
            dash=dash,
            pattern=pattern,
        )
        r.text(x + 80, y + 38, label, size=34, bold=True, color=INK)
        r.line([(1392, 226), (1402, y + 27), (1412, y + 27)], stroke=PURPLE, sw=2.5, arrow=True)
    r.text(1450, 390, "ALL TRACKS", size=32, bold=True, color=MUTED)

    # Response synthesis and a single track-level decode.
    r.rect(1120, 476, 250, 132, fill=PURPLE_FILL, stroke=PURPLE, sw=4, radius=18)
    r.text(1245, 520, "FUSION", size=36, bold=True)
    r.text(1245, 570, "Σₖ gᵢℓₖ · rᵏᵢℓ", size=32, bold=True, color=PURPLE_TEXT)

    r.rect(1390, 476, 235, 132, fill=AMBER_FILL, stroke=AMBER, sw=4, radius=18)
    r.text(1507.5, 510, "NOLA", size=36, bold=True)
    r.text(1507.5, 554, "RESPONSE", size=32, bold=True, color=MUTED)
    r.text(1507.5, 594, "Rᵢ(t)", size=36, bold=True, color=MUTED)

    r.rect(1650, 476, 160, 132, fill=GREEN_FILL, stroke=GREEN, sw=4, radius=18, dash="11 7")
    r.text(1730, 505, ["DECODE", "ONCE", "ĉᵢ"], size=32, bold=True, color=GREEN, leading=1.0)

    r.rect(1815, 468, 180, 152, fill=WHITE, stroke=INK, sw=3.5, radius=40)
    r.text(1905, 498, ["TRACK", "COUNTS"], size=32, bold=True, leading=1.0)
    r.text(1905, 570, "ĉ = [ĉᵢ]", size=32, bold=True)
    r.text(1905, 606, "i = 1 … N", size=32, bold=True)

    # Narrow training-contract strip.  Only the synchronized objective is solid.
    r.rect(340, 642, 1628, 86, fill=GRAY_FILL, stroke=GRAY, sw=2.5, radius=13)
    r.text(370, 668, ["TRAINING", "CONTRACT"], size=32, bold=True, color=MUTED, anchor="start", leading=1.0)
    r.rect(635, 650, 310, 70, fill=WHITE, stroke=INK, sw=2.5, radius=9)
    r.text(790, 676, ["PAMS-TCC", "START"], size=32, bold=True, leading=1.0)
    r.rect(975, 650, 889, 70, fill=AMBER_FILL, stroke=AMBER, sw=2.5, radius=9, dash="10 7")
    r.text(1419.5, 676, "λᵣ Lroute + λₜ Ltrack", size=32, bold=True, color=INK)
    r.text(
        1419.5,
        708,
        "SYNC-REQUIRED  •  DELETE IF ABSENT",
        size=32,
        bold=True,
        color=RED,
    )


def render_svg(path: Path) -> None:
    renderer = SVGRenderer()
    draw_scene(renderer)
    path.write_bytes(renderer.finish())


def render_pdf(path: Path) -> None:
    renderer = PDFRenderer(path)
    draw_scene(renderer)
    renderer.finish()


def render_png(pdf_path: Path, png_path: Path) -> None:
    command = shutil.which("pdftoppm") or shutil.which("pdftoppm.cmd")
    if not command:
        raise RuntimeError("pdftoppm is required to generate the clean PNG preview")
    command_path = Path(command)
    if command_path.suffix.lower() == ".cmd" and len(command_path.parents) >= 3:
        native = (
            command_path.parents[2]
            / "native"
            / "poppler"
            / "Library"
            / "bin"
            / "pdftoppm.exe"
        )
        if native.exists():
            command = str(native)
    prefix = png_path.with_suffix("")
    subprocess.run(
        [
            command,
            "-png",
            "-singlefile",
            "-r",
            "285.714",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
    )


def main() -> None:
    svg_path = HERE / "icassp_framework.svg"
    pdf_path = HERE / "icassp_framework.pdf"
    png_path = HERE / "icassp_framework.png"
    render_svg(svg_path)
    render_pdf(pdf_path)
    render_png(pdf_path, png_path)
    print(f"wrote {svg_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
