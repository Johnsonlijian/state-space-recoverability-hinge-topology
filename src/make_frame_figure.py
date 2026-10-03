"""Moment-frame commitment-drift figure. Python/matplotlib only."""

import json
from pathlib import Path

import matplotlib as mpl
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

rows = json.loads((RES / "moment_bracket.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(3.5, 2.8))
colors = {4: "#B64342", 8: "#0F4D92"}
for nstory, dx in ((4, -0.12), (8, 0.12)):
    sub = sorted(
        (
            r
            for r in rows
            if r["nstory"] == nstory and r.get("status") == "committed"
        ),
        key=lambda r: r["id"],
    )
    y = np.array([r["drift_at_cut"] * 100 for r in sub])
    x = np.full(y.shape, dx) + np.linspace(-0.06, 0.06, y.size)
    left = np.array([r.get("last_recover_level") is None for r in sub])
    if np.any(~left):
        ax.scatter(x[~left], y[~left], s=16, c=colors[nstory], zorder=3)
    if np.any(left):
        ax.scatter(
            x[left],
            y[left],
            s=18,
            facecolors="none",
            edgecolors=colors[nstory],
            linewidths=0.7,
            zorder=3,
        )
    ax.hlines(np.median(y), dx - 0.16, dx + 0.16, color=colors[nstory], lw=1.3)
ax.axhline(3.2, color="#C47B00", lw=0.8, ls="--", label="Pushover peak, 3.2%")
ax.axhline(10, color="#767676", lw=0.6, ls="-.", label="10% drift")
ax.set_xlim(-0.45, 0.45)
ax.set_ylim(0, 12)
ax.set_xticks([-0.12, 0.12])
ax.set_xticklabels(["4-story", "8-story"])
ax.set_ylabel("Story drift at first commitment (%)")
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
fig.savefig(FIG / "fig6_moment_frame.svg", bbox_inches="tight")
fig.savefig(FIG / "fig6_moment_frame.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig6_moment_frame.png", dpi=200, bbox_inches="tight")
print("wrote fig6", "n", sum(1 for r in rows if r.get("status") == "committed"))
plt.close(fig)

rows7 = json.loads((RES / "column_hinge_rows.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(3.5, 2.8))
peaks = {4: 1.61, 8: 1.42}
for nstory, dx in ((4, -0.12), (8, 0.12)):
    sub = sorted(
        (
            r
            for r in rows7
            if r["nstory"] == nstory and r.get("status") == "committed"
        ),
        key=lambda r: r["id"],
    )
    y = np.array(
        [r.get("drift_fine", r["drift_at_cut"]) * 100 for r in sub]
    )
    x = np.full(y.shape, dx) + np.linspace(-0.06, 0.06, y.size)
    ax.scatter(x, y, s=16, c=colors[nstory], zorder=3)
    ax.hlines(np.median(y), dx - 0.16, dx + 0.16, color=colors[nstory], lw=1.3)
    ax.plot(
        [dx - 0.18, dx + 0.18],
        [peaks[nstory], peaks[nstory]],
        color="#C47B00",
        lw=0.8,
        ls="--",
        zorder=2,
    )
ax.plot([], [], color="#C47B00", lw=0.8, ls="--", label="Pushover peak")
ax.axhline(7, color="#767676", lw=0.6, ls=":", label="Net-zero, near 7%")
ax.axhline(10, color="#767676", lw=0.6, ls="-.", label="10% drift")
ax.set_xlim(-0.45, 0.45)
ax.set_ylim(0, 12)
ax.set_xticks([-0.12, 0.12])
ax.set_xticklabels(["4-story", "8-story"])
ax.set_ylabel("Story drift at first commitment (%)")
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
fig.savefig(FIG / "fig7_column_hinges.svg", bbox_inches="tight")
fig.savefig(FIG / "fig7_column_hinges.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig7_column_hinges.png", dpi=200, bbox_inches="tight")
print("wrote fig7", "n", sum(1 for r in rows7 if r.get("status") == "committed"))

