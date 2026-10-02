from __future__ import annotations

import os
from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
# Keep a clean checkout self-contained.  An explicit export directory may be
# supplied by a release workflow when the upload-facing package is rebuilt.
UPLOAD = Path(os.environ["GRAPHICAL_ABSTRACT_UPLOAD_DIR"]).resolve() if os.environ.get("GRAPHICAL_ABSTRACT_UPLOAD_DIR") else None

NAVY = "#17324d"
BLUE = "#2166ac"
TEAL = "#1b9e77"
AMBER = "#d89000"
RED = "#c33c3c"
GRAY = "#6f7b85"
LIGHT = "#f5f7f9"
MID = "#d9e1e8"
DARK = "#1f2933"


def strip_svg_metadata(path: Path) -> None:
    """Remove generator/date metadata from the upload-facing SVG master."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\s*<metadata>.*?</metadata>\s*", "\n", text, count=1, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def box(ax, xy, width, height, title, body, edge, face=LIGHT, title_color=NAVY):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.3, edgecolor=edge, facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.16, y + height - 0.24, title, ha="left", va="top",
            fontsize=11.1, fontweight="bold", color=title_color, zorder=3)
    ax.text(x + 0.16, y + height - 0.58, body, ha="left", va="top",
            fontsize=9.1, color=DARK, linespacing=1.35, zorder=3)
    return patch


def arrow(ax, start, end, color=NAVY, label=None, label_offset=(0, 0), lw=1.8):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, shrinkA=7, shrinkB=7, zorder=4,
    )
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=8.7,
                color=color, bbox=dict(boxstyle="round,pad=0.16",
                facecolor="white", edgecolor="none", alpha=0.88), zorder=5)


def draw_frame(ax, x0, y0, width, height, hinge_mode, color):
    ncol, nlev = 3, 4
    dx = width / (ncol - 1)
    dy = height / nlev
    # Columns and beams.
    for i in range(ncol):
        ax.plot([x0 + i * dx, x0 + i * dx], [y0, y0 + height],
                color=GRAY, lw=2.0, zorder=3)
    for j in range(nlev + 1):
        ax.plot([x0, x0 + width], [y0 + j * dy, y0 + j * dy],
                color=GRAY, lw=1.8, zorder=3)
    # Hinge marks.
    marks = []
    if hinge_mode == "base":
        marks = [(i, 0) for i in range(ncol)]
    elif hinge_mode == "ends":
        marks = [(i, j) for i in range(ncol) for j in range(nlev + 1)]
    else:
        marks = [(i, j) for i in range(ncol) for j in range(nlev + 1)]
    for i, j in marks:
        ax.add_patch(Circle((x0 + i * dx, y0 + j * dy), 0.065,
                            facecolor=color, edgecolor="white", lw=0.5, zorder=5))
    if hinge_mode == "ends":
        ax.text(x0 + width + 0.14, y0 + height * 0.50,
                "hinge map", fontsize=8.2, color=color, rotation=90,
                ha="center", va="center")
    elif hinge_mode == "budget":
        ax.text(x0 + width + 0.16, y0 + height * 0.50,
                "capacity split", fontsize=8.2, color=color, rotation=90,
                ha="center", va="center")


def phase_inset(ax, x0, y0, width, height):
    ax.add_patch(Rectangle((x0, y0), width, height, facecolor="white",
                           edgecolor=MID, lw=1.0, zorder=2))
    ax.plot([x0 + 0.20, x0 + width - 0.12], [y0 + 0.23, y0 + 0.23],
            color=GRAY, lw=0.8, zorder=3)
    ax.plot([x0 + 0.20, x0 + 0.20], [y0 + 0.18, y0 + height - 0.12],
            color=GRAY, lw=0.8, zorder=3)
    ax.text(x0 + width - 0.20, y0 + 0.05, "drift u", fontsize=7.8,
            color=GRAY, ha="right", va="bottom")
    ax.text(x0 + 0.03, y0 + height - 0.14, "v", fontsize=8.0,
            color=GRAY, ha="left", va="top")
    # Recoverable basin and committed side.
    ax.fill_between([x0 + 0.20, x0 + width - 0.12],
                    [y0 + 0.23, y0 + 0.44], [y0 + 0.23, y0 + height - 0.18],
                    color="#dcebf7", alpha=0.80, zorder=1)
    ax.fill_between([x0 + 0.20, x0 + width - 0.12],
                    [y0 + 0.23, y0 + 0.23], [y0 + 0.23, y0 + 0.44],
                    color="#f7dddd", alpha=0.95, zorder=1)
    x = [x0 + 0.20 + (width - 0.32) * q for q in (0.0, 0.22, 0.43, 0.63, 0.82, 1.0)]
    y = [y0 + 0.44, y0 + 0.54, y0 + 0.71, y0 + 0.90, y0 + 1.01, y0 + 1.04]
    ax.plot(x, y, color=RED, lw=2.2, zorder=4)
    ax.text(x0 + 0.29, y0 + height - 0.25, "recoverable basin",
            fontsize=7.9, color=BLUE, ha="left", va="top")
    ax.text(x0 + 0.29, y0 + 0.33, "committed side",
            fontsize=7.9, color=RED, ha="left", va="bottom")


def main():
    fig, ax = plt.subplots(figsize=(12.0, 5.7))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.7)
    ax.axis("off")

    ax.text(0.35, 5.35, "Recoverability is a state-space crossing",
            fontsize=18, fontweight="bold", color=NAVY, ha="left", va="top")
    ax.text(0.35, 5.02,
            "The hinge map changes the path by which a frame reaches the barrier.",
            fontsize=10.4, color=GRAY, ha="left", va="top")

    # Object / state.
    box(ax, (0.35, 2.30), 2.55, 2.25, "1  Current nonlinear state",
        "frame + hinge map\nstate: $u, v, S, F_y$", NAVY)
    draw_frame(ax, 0.63, 2.48, 1.10, 0.95, "ends", AMBER)
    ax.text(1.93, 2.48, "P-Delta load path", fontsize=7.8, color=GRAY,
            ha="left", va="center")
    ax.text(2.14, 3.18, "released at\nthe current state", fontsize=8.5,
            color=DARK, ha="left", va="center")

    # Driver / continuation.
    box(ax, (3.35, 2.30), 2.25, 2.25, "2  Remove future input",
        "ground acceleration -> 0\ncontinue the same model\nfor a fixed horizon", BLUE)
    ax.plot([3.72, 4.09, 4.42, 4.70, 5.03], [3.28, 3.62, 3.03, 3.47, 3.21],
            color=BLUE, lw=1.4, zorder=3)
    ax.text(3.72, 2.72, "zero-input continuation", fontsize=8.5,
            color=BLUE, ha="left")

    # State-space barrier.
    box(ax, (6.05, 1.76), 2.45, 2.79, "3  Barrier in (u, v)",
        "static saddle = v = 0 point\nof the separatrix", TEAL)
    phase_inset(ax, 6.36, 1.96, 1.83, 1.36)
    arrow(ax, (2.95, 3.42), (3.28, 3.42), color=NAVY)
    arrow(ax, (5.64, 3.42), (6.02, 3.42), color=BLUE)

    # Outcome branch.
    box(ax, (9.15, 3.16), 2.42, 1.39, "4a  Recoverable",
        "state returns to the\nelastic neighborhood", BLUE, face="#eaf3fb")
    box(ax, (9.15, 1.26), 2.42, 1.39, "4b  Committed",
        "state escapes after\ninput removal", RED, face="#fff0f0")
    arrow(ax, (8.56, 3.68), (9.08, 3.88), color=BLUE, label="inside", label_offset=(0, 0.10))
    arrow(ax, (8.56, 2.66), (9.08, 2.03), color=RED, label="crossed", label_offset=(0.02, -0.12))

    # Mechanism strip.
    ax.add_patch(FancyBboxPatch((0.35, 0.42), 11.22, 0.63,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor="#f8fafc", edgecolor=MID, lw=1.0))
    ax.text(0.60, 0.88, "Hinge topology and capacity allocation",
            fontsize=9.5, fontweight="bold", color=NAVY, ha="left", va="center")
    cells = [
        (3.66, "beam ends + bases", AMBER, "local hinge map"),
        (6.05, "column ends added", RED, "admissible path"),
        (8.42, "multi-bay budget", TEAL, "capacity split"),
    ]
    for x, label, col, small in cells:
        ax.add_patch(FancyBboxPatch((x, 0.53), 1.75, 0.38,
                                    boxstyle="round,pad=0.015,rounding_size=0.04",
                                    facecolor="white", edgecolor=col, lw=1.0))
        ax.text(x + 0.875, 0.76, label, fontsize=8.2, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 0.875, 0.60, small, fontsize=7.2, color=GRAY,
                ha="center", va="center")
    arrow(ax, (5.43, 0.74), (5.95, 0.74), color=GRAY, lw=1.1)
    arrow(ax, (7.82, 0.74), (8.32, 0.74), color=GRAY, lw=1.1)
    ax.text(11.02, 0.74, "barrier shifts", fontsize=9.3,
            color=RED, ha="center", va="center", fontweight="bold")

    fig.text(0.5, 0.015,
             "Mechanism schematic; quantitative evidence is provided by the phase-plane and frame census figures.",
             ha="center", va="bottom", fontsize=8.5, color=GRAY)
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT]
    if UPLOAD is not None and UPLOAD != OUT:
        UPLOAD.mkdir(parents=True, exist_ok=True)
        targets.append(UPLOAD)
    for target in targets:
        fig.savefig(target / "graphical_abstract.pdf", bbox_inches="tight")
        fig.savefig(target / "graphical_abstract.svg", bbox_inches="tight")
        fig.savefig(target / "graphical_abstract.png", dpi=240, bbox_inches="tight")
        strip_svg_metadata(target / "graphical_abstract.svg")
    plt.close(fig)
    print("wrote graphical_abstract.pdf/.svg/.png")


if __name__ == "__main__":
    main()
