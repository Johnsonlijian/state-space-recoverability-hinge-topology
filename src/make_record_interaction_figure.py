"""Create the record-interaction diagnostic for the reproducibility package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures"


def main() -> None:
    reference = {
        row["id"]: float(row["drift_fine_recomputed"])
        for row in json.loads((RESULTS / "column_hinge_rows_recomputed.json").read_text(encoding="utf-8"))
        if row["nstory"] == 4
    }
    rows = json.loads((ROOT / "results" / "multibay_dynamic_full.json").read_text(encoding="utf-8"))
    labels = [
        ("2bay_free_member", "2-bay member", "#2166ac"),
        ("3bay_free_member", "3-bay member", "#1b9e77"),
        ("2bay_free_normalized", "2-bay normalized", "#d95f02"),
        ("2bay_rigid_designinformed", "2-bay design", "#7570b3"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), sharex=True, sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (key, title, color) in zip(axes.flat, labels):
        by_id = {row["id"]: float(row["drift_at_cut"]) for row in rows if row["label"] == key}
        ids = sorted(reference)
        x = [100.0 * reference[item] for item in ids]
        y = [100.0 * (by_id[item] - reference[item]) for item in ids]
        rho = float(spearmanr(x, y).statistic)
        ax.scatter(x, y, s=31, color=color, edgecolor="white", linewidth=0.55, alpha=0.92, zorder=3)
        ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=1)
        ax.axvline(100.0 * sorted(reference.values())[10], color="#999999", linestyle=(0, (3, 3)), linewidth=0.75, zorder=1)
        ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold", pad=5)
        ax.text(0.98, 0.94, rf"$\rho_s={rho:.2f}$", transform=ax.transAxes, ha="right", va="top", fontsize=9.2, color="#222222")
        ax.grid(axis="y", color="#e7e7e7", linewidth=0.65)
        ax.set_xlim(1.7, 7.1)
        ax.set_ylim(-4.6, 4.6)
        ax.tick_params(labelsize=8.6, length=3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].set_ylabel("Paired change (percentage points)", fontsize=9.2)
    axes[1, 0].set_ylabel("Paired change (percentage points)", fontsize=9.2)
    axes[1, 0].set_xlabel("One-bay commitment drift (%)", fontsize=9.2)
    axes[1, 1].set_xlabel("One-bay commitment drift (%)", fontsize=9.2)
    fig.suptitle("Same-record commitment shifts across topology cells", fontsize=12.6, fontweight="bold", y=1.015)
    fig.text(0.5, -0.035, "Each point is one of the same 21 records; dashed vertical line is the one-bay median.", ha="center", fontsize=8.8, color="#444444")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig10_record_interaction.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10_record_interaction.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig10_record_interaction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig10_record_interaction.{pdf,svg,png}", flush=True)


if __name__ == "__main__":
    main()
