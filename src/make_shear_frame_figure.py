"""Build the story-shear commitment figure from the current result ledger."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def strip_svg_metadata(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\s*<metadata>.*?</metadata>\s*", "\n", text, count=1, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    rows = json.loads((RES / "frame_rows.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = {4: "#B64342", 8: "#0F4D92"}
    for nstory, dx in ((4, -0.12), (8, 0.12)):
        sub = sorted((r for r in rows if r["nstory"] == nstory), key=lambda r: r["id"])
        y = np.array([r["drift_at_commitment"] * 100 for r in sub])
        x = np.full(y.shape, dx) + np.linspace(-0.06, 0.06, y.size)
        ax.scatter(x, y, s=16, c=colors[nstory], zorder=3)
        ax.hlines(np.median(y), dx - 0.16, dx + 0.16, color=colors[nstory], lw=1.3)
    ax.axhline(0.8, color="#2D9BA8", lw=0.8, ls=":", label="Yield, 0.8%")
    ax.axhline(5.5, color="#C47B00", lw=0.8, ls="--", label="Static net-zero, 5.5%")
    ax.axhline(10, color="#767676", lw=0.6, ls="-.", label="10% drift")
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(0, 10.5)
    ax.set_xticks([-0.12, 0.12])
    ax.set_xticklabels(["4-story", "8-story"])
    ax.set_ylabel("Story drift at first commitment (%)")
    ax.legend(loc="upper right")
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 200})):
        target = FIG / f"fig5_frame_commitment.{suffix}"
        fig.savefig(target, bbox_inches="tight", **kwargs)
        if suffix == "svg":
            strip_svg_metadata(target)
    plt.close(fig)
    print("wrote fig5_frame_commitment")


if __name__ == "__main__":
    main()
