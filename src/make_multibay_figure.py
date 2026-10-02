"""Create the multi-bay topology and capacity-allocation synthesis figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COLORS = {
    "one_bay": "#314e6e",
    "2bay_free_member": "#168c8a",
    "3bay_free_member": "#42a5a5",
    "2bay_free_normalized": "#d28a2e",
    "2bay_rigid_designinformed": "#7752a8",
}
LABELS = {
    "one_bay": "1-bay reference",
    "2bay_free_member": "2-bay / member",
    "3bay_free_member": "3-bay / member",
    "2bay_free_normalized": "2-bay / normalized",
    "2bay_rigid_designinformed": "2-bay / design-informed",
}


def load() -> tuple[list[dict], list[dict], dict, dict]:
    dyn = json.loads((RES / "multibay_dynamic_full.json").read_text(encoding="utf-8"))
    static = json.loads((RES / "multibay_static.json").read_text(encoding="utf-8"))
    paired = json.loads((RES / "multibay_paired_comparison.json").read_text(encoding="utf-8"))
    baseline = json.loads((RES / "column_hinge_rows_recomputed.json").read_text(encoding="utf-8"))
    return dyn, static, paired, {"baseline": baseline}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dfe5ea", linewidth=0.5, zorder=0)
    ax.tick_params(length=3, width=0.6)


def main() -> None:
    dyn, static, paired, refs = load()
    baseline_rows = [r for r in refs["baseline"] if r["nstory"] == 4 and r.get("drift_fine_recomputed") is not None]
    baseline = np.asarray([r["drift_fine_recomputed"] * 100 for r in baseline_rows])
    models = ["one_bay", "2bay_free_member", "3bay_free_member", "2bay_free_normalized", "2bay_rigid_designinformed"]
    values = {"one_bay": baseline}
    for label in models[1:]:
        values[label] = np.asarray([r["drift_at_cut"] * 100 for r in dyn if r["label"] == label and r["status"] == "committed"])
    ratios = {
        label: np.asarray(
            [r["max_column_rotation_at_cut"] / max(r["max_beam_rotation_at_cut"], 1e-12) for r in dyn if r["label"] == label and r["status"] == "committed"]
        )
        for label in models[1:]
    }
    static_lookup = {}
    for label, query in {
        "2bay_free_member": dict(n_bays=2, rigid_diaphragm=False, capacity_mode="member", column_ratio=1.0, first_story_factor=1.0),
        "3bay_free_member": dict(n_bays=3, rigid_diaphragm=False, capacity_mode="member", column_ratio=1.0, first_story_factor=1.0),
        "2bay_free_normalized": dict(n_bays=2, rigid_diaphragm=False, capacity_mode="frame_normalized", column_ratio=1.0, first_story_factor=1.0),
        "2bay_rigid_designinformed": dict(n_bays=2, rigid_diaphragm=True, capacity_mode="frame_normalized", column_ratio=1.5, first_story_factor=1.25),
    }.items():
        static_lookup[label] = next(r for r in static if r["nstory"] == 4 and r["pattern"] == "triangular" and all(r[k] == v for k, v in query.items()))

    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.75), constrained_layout=True)
    ax = axs[0, 0]
    style_axes(ax)
    x = np.arange(1, 6)
    peak = [1.609, static_lookup["2bay_free_member"]["peak_story_drift"] * 100, static_lookup["3bay_free_member"]["peak_story_drift"] * 100, static_lookup["2bay_free_normalized"]["peak_story_drift"] * 100, static_lookup["2bay_rigid_designinformed"]["peak_story_drift"] * 100]
    labels = ["1-bay", "2-bay\nmember", "3-bay\nmember", "2-bay\nnormalized", "2-bay\ndesign"]
    c = [COLORS["one_bay"], COLORS["2bay_free_member"], COLORS["3bay_free_member"], COLORS["2bay_free_normalized"], COLORS["2bay_rigid_designinformed"]]
    ax.scatter(x, peak, s=34, color=c, marker="o", edgecolor="white", linewidth=0.5, label="static peak", zorder=3)
    for idx, label in enumerate(labels):
        key = ["one_bay", "2bay_free_member", "3bay_free_member", "2bay_free_normalized", "2bay_rigid_designinformed"][idx]
        if key == "one_bay":
            z = 0.0694
        else:
            z = static_lookup[key]["net_zero_story_drift"]
        if z is not None:
            ax.scatter(x[idx], z * 100, s=34, color=c[idx], marker="D", edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(10, color="#9aa6b2", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Story drift (%)")
    ax.set_title("Static reference points")
    ax.set_ylim(0, 12)
    ax.text(0.02, 0.04, "○ peak   ◆ net-zero\nstatic paths stop on nonconvergence", transform=ax.transAxes, fontsize=7, color="#4f5963")

    ax = axs[0, 1]
    style_axes(ax)
    positions = np.arange(len(models))
    bp = ax.boxplot([values[k] for k in models], positions=positions, widths=0.52, patch_artist=True, showfliers=False, medianprops={"color": "#1f2933", "linewidth": 1.2}, whiskerprops={"linewidth": 0.75}, capprops={"linewidth": 0.75})
    for patch, k in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS[k]); patch.set_alpha(0.24); patch.set_edgecolor(COLORS[k])
    rng = np.random.default_rng(4609)
    for idx, k in enumerate(models):
        jitter = rng.uniform(-0.14, 0.14, size=len(values[k]))
        ax.scatter(np.full(len(values[k]), idx) + jitter, values[k], s=12, color=COLORS[k], alpha=0.78, edgecolor="white", linewidth=0.25, zorder=3)
    ax.axhline(10, color="#9aa6b2", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_xticks(positions, ["1-bay", "2-bay\nmem.", "3-bay\nmem.", "2-bay\nnorm.", "2-bay\ndesign"])
    ax.set_ylabel("Commitment drift (%)")
    ax.set_title("21-record zero-input commitment")
    ax.set_ylim(0, 11)
    ax.text(0.02, 0.04, "Points are records; boxes are IQR", transform=ax.transAxes, fontsize=7, color="#4f5963")

    ax = axs[1, 0]
    style_axes(ax)
    comp = ["2bay_free_member", "3bay_free_member", "2bay_free_normalized", "2bay_rigid_designinformed"]
    xx = np.arange(len(comp))
    for i, k in enumerate(comp):
        q = paired["paired_against_reference"][k]
        lo, hi = np.asarray(q["bootstrap_95_median_delta"]) * 100
        med = q["median_delta"] * 100
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o", color=COLORS[k], markersize=5, capsize=3.2, linewidth=1.2, zorder=3)
        ax.text(i, hi + 0.13, f"{q['positive_count']}/{q['negative_count']}", ha="center", va="bottom", fontsize=7, color="#4f5963")
    ax.axhline(0, color="#26323d", linewidth=0.8)
    ax.set_xticks(xx, ["2-bay\nmember", "3-bay\nmember", "2-bay\nnormalized", "2-bay\ndesign"])
    ax.set_ylabel("Paired Δ commitment drift (percentage points)")
    ax.set_title("Same-record change from 1-bay")
    ax.set_ylim(-2.5, 3.0)
    ax.text(0.02, 0.04, "Whiskers: bootstrap 95% CI for paired median", transform=ax.transAxes, fontsize=7, color="#4f5963")

    ax = axs[1, 1]
    style_axes(ax)
    for i, k in enumerate(comp):
        arr = ratios[k]
        jitter = rng.uniform(-0.13, 0.13, size=len(arr))
        ax.scatter(np.full(len(arr), i) + jitter, arr, s=12, color=COLORS[k], alpha=0.78, edgecolor="white", linewidth=0.25)
        ax.plot([i - 0.18, i + 0.18], [np.median(arr), np.median(arr)], color="#26323d", linewidth=1.2)
    ax.axhline(1, color="#9aa6b2", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_xticks(xx, ["2-bay\nmember", "3-bay\nmember", "2-bay\nnormalized", "2-bay\ndesign"])
    ax.set_ylabel("Max column / beam rotation")
    ax.set_title("Local hinge rotation at commitment")
    ax.set_ylim(0, 2.7)
    ax.text(0.02, 0.04, "Ratio > 1: column ends rotate farther", transform=ax.transAxes, fontsize=7, color="#4f5963")

    for ax, letter in zip(axs.flat, "abcd"):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    fig.suptitle("Multi-bay topology changes the recoverability bracket", fontsize=12, fontweight="bold", x=0.5, y=1.02)
    FIG.mkdir(parents=True, exist_ok=True)
    # The manuscript calls this synthesis Figure 8. Keep the generated
    # artifact name aligned with the upload package and the public release.
    fig.savefig(FIG / "fig8_multibay.svg", bbox_inches="tight")
    fig.savefig(FIG / "fig8_multibay.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig8_multibay.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "fig8_multibay.svg")


if __name__ == "__main__":
    main()
