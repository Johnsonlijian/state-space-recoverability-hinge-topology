"""Publication figures for the gravity-separatrix study. Python/matplotlib only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import run_study as rs

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
    }
)

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
GOLD = "#C47B00"
INK = "#272727"
GRAY = "#767676"


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def quant(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def family_stats(rows, tag):
    out = {}
    for kind in ("gabor", "kanai_tajimi"):
        sub = [r for r in rows if r["tag"] == tag and r["kind"] == kind]
        drifts = [r["drift_at_first_commitment"] for r in sub]
        fy = [r["fy_ratio_at_first_commitment"] for r in sub]
        ratios = []
        for r in sub:
            a, b = r["scale"]["tan"], r["scale"]["truth"]
            if a and b:
                ratios.append(a / b)
        re = sum(1 for r in sub if r["reentry_at_commitment_scale"] == 1)
        no10 = sum(1 for r in sub if r["scale"]["d10"] is None)
        out[kind] = {
            "drift": quant(drifts),
            "fy": quant(fy),
            "tan_ratio": quant(ratios),
            "reentry_n": re,
            "reentry_N": len(sub),
            "never_10pct": no10,
            "all_below_static": int(sum(d < (0.01 / 0.10) - 1e-12 for d in drifts)),
        }
    return out


def fig_separatrix():
    T, theta = 1.0, 0.10
    k, c, uy, Fy0 = rs.oscillator(T, theta)
    uc = Fy0 / (theta * k)
    u = np.linspace(0.08 * uc, 1.35 * uc, 72)
    v_scale = rs.analytical_v_crit(0.45 * uc, Fy0, k, theta)[0]
    v = np.linspace(-1.6 * v_scale, 1.8 * v_scale, 55)
    labels = rs.phase_labels(u, v, k, c, theta, Fy0, uy, rs.H)
    np.savetxt(RES / "fig1_phase_u.csv", u, delimiter=",")
    np.savetxt(RES / "fig1_phase_v.csv", v, delimiter=",")
    np.savetxt(RES / "fig1_phase_labels.csv", labels, delimiter=",", fmt="%d")

    uu = np.linspace(0.2 * uc, uc * 0.995, 200)
    vv = np.array([rs.analytical_v_crit(ui, Fy0, k, theta)[0] for ui in uu])

    fig, ax = plt.subplots(figsize=(3.46, 2.7))
    cmap = ListedColormap(["#D6E4F0", "#F4D2CF"])
    extent = [u[0] / rs.H * 100, u[-1] / rs.H * 100, v[0], v[-1]]
    ax.imshow(
        labels.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.plot(uu / rs.H * 100, vv, color=INK, lw=1.1)
    ax.annotate(
        "Inward velocity can\nstill cross the barrier",
        xy=(9.2, -0.45),
        xytext=(2.6, -0.52),
        fontsize=6,
        color=RED,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.5),
    )
    ax.axvline(10.0, color=GOLD, lw=0.9, ls="--")
    ax.axvline(1.0, color=TEAL, lw=0.8, ls=":", label="Yield drift, 1%")
    ax.set_xlabel("Drift on the yield cap (%)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    handles = [
        Line2D([0], [0], color="#8FB4D4", lw=4, label="Recovers under zero input"),
        Line2D([0], [0], color="#E7A8A4", lw=4, label="Runs away under zero input"),
        Line2D([0], [0], color=INK, lw=1.1, label="Undamped energy barrier"),
        Line2D([0], [0], color=GOLD, lw=0.9, ls="--", label="Static saddle (10% drift)"),
        Line2D([0], [0], color=TEAL, lw=0.8, ls=":", label="Yield drift (1%)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.02))
    ax.set_title("On-cap states, 5% damping", loc="left", fontsize=7, color=INK)
    save(fig, "fig1_separatrix")


def pick_traces(rows, motions, meta):
    b0 = [r for r in rows if r["tag"] == "main_b0"]
    gabor = [r for r in b0 if r["kind"] == "gabor" and r["scale"]["d10"] is None]
    gabor_pick = sorted(gabor, key=lambda r: r["drift_at_first_commitment"])[len(gabor) // 2]
    kt = [r for r in b0 if r["kind"] == "kanai_tajimi"]
    kt_med = float(np.median([r["drift_at_first_commitment"] for r in kt]))
    kt_pick = min(kt, key=lambda r: abs(r["drift_at_first_commitment"] - kt_med))
    chosen = []
    id_to_idx = {info["id"]: i for i, info in enumerate(meta)}
    for rec, label in ((gabor_pick, "pulse"), (kt_pick, "filtered")):
        idx = id_to_idx[rec["id"]]
        scale = rec["scale"]["truth"]
        nkeep, t, u, v, drift, truth, tan, en, sad, fyr = rs.trace_motion(
            np.ascontiguousarray(motions[idx]),
            float(scale),
            rs.DT,
            0.05,
            rs.H,
            *rs.oscillator(1.0, 0.10)[:2],
            0.10,
            0.0,
            rs.oscillator(1.0, 0.10)[3],
            rs.oscillator(1.0, 0.10)[2],
        )
        # oscillator returns k, c, uy, Fy0 — I unpacked badly. Fix below if needed.
        chosen.append((label, rec, nkeep, t, drift, truth, tan, en))
    return chosen


def trace_one(ag, scale):
    k, c, uy, Fy0 = rs.oscillator(1.0, 0.10)
    return rs.trace_motion(
        np.ascontiguousarray(ag), float(scale), rs.DT, 0.05, rs.H, k, c, 0.10, 0.0, Fy0, uy
    )


def fig_traces(rows):
    motions, meta = rs.make_suite(20, 20, 12.0, 25.0)
    b0 = [r for r in rows if r["tag"] == "main_b0"]
    gabor = [r for r in b0 if r["kind"] == "gabor" and r["scale"]["d10"] is None]
    g_pick = sorted(gabor, key=lambda r: r["drift_at_first_commitment"])[len(gabor) // 2]
    kt = [r for r in b0 if r["kind"] == "kanai_tajimi"]
    kt_med = float(np.median([r["drift_at_first_commitment"] for r in kt]))
    k_pick = min(kt, key=lambda r: abs(r["drift_at_first_commitment"] - kt_med))
    id_to_idx = {info["id"]: i for i, info in enumerate(meta)}
    picks = [("Pulse", g_pick), ("Filtered noise", k_pick)]

    fig, axes = plt.subplots(
        4, 1, figsize=(7.2, 5.4), sharex=False, gridspec_kw={"height_ratios": [3.4, 0.55, 3.4, 0.55], "hspace": 0.72}
    )
    catalog = []
    ylims = [(-3.5, 5.5), (-2.0, 16.0)]
    for i, (label, rec) in enumerate(picks):
        ax = axes[2 * i]
        strip = axes[2 * i + 1]
        idx = id_to_idx[rec["id"]]
        nkeep, t, u, v, drift, truth, tan, en, sad, fyr = trace_one(
            motions[idx], rec["scale"]["truth"]
        )
        t, drift, truth, tan, en = (a[:nkeep] for a in (t, drift, truth, tan, en))
        block = np.column_stack([t, drift, truth, tan, en])
        np.savetxt(
            RES / f"fig2_{rec['id']}.csv",
            block,
            delimiter=",",
            header="t_s,drift,truth,tangent_negative,energy_proxy",
            comments="",
        )
        ax.plot(t, drift * 100, color=BLUE, lw=1.0)
        if label == "Filtered noise":
            ax.axhline(10, color=GOLD, lw=0.7, ls="--")
        ax.axhline(1, color=TEAL, lw=0.6, ls=":")
        ax.set_ylim(*ylims[i])
        ax.set_ylabel("Drift (%)")
        drift0 = rec["drift_at_first_commitment"] * 100
        ax.set_title(
            f"{label} | first commitment: {drift0:.1f}% drift (PGA {rec['scale']['truth']:.2f} g)",
            loc="left",
            fontsize=7,
        )
        strip.fill_between(t, 0, truth, step="mid", color=RED, alpha=0.85, linewidth=0)
        strip.fill_between(t, 1, 1 + tan, step="mid", color=GRAY, alpha=0.9, linewidth=0)
        strip.set_ylim(0, 2)
        strip.set_yticks([0.5, 1.5])
        strip.set_yticklabels(["runaway", "tangent"])
        strip.spines["left"].set_visible(False)
        strip.tick_params(axis="y", length=0)
        catalog.append(
            {
                "id": rec["id"],
                "kind": rec["kind"],
                "scale_g": rec["scale"]["truth"],
                "drift_at_commitment": rec["drift_at_first_commitment"],
                "reentry": rec["reentry_at_commitment_scale"],
                "d10_scale": rec["scale"]["d10"],
                "tan_scale": rec["scale"]["tan"],
            }
        )
    axes[3].set_xlabel("Time (s)")
    save(fig, "fig2_trajectories")
    (RES / "fig2_cases.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def fig_drift(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    specs = [("main_b0", "Constant strength"), ("main_b004", r"Strength loss ($\beta=0.04$)")]
    for ax, (tag, title) in zip(axes, specs):
        for kind, color, name, dx in (
            ("gabor", RED, "Pulse", -0.12),
            ("kanai_tajimi", BLUE, "Filtered noise", 0.12),
        ):
            sub = [r for r in rows if r["tag"] == tag and r["kind"] == kind]
            y = np.array([r["drift_at_first_commitment"] * 100 for r in sub])
            x = np.full_like(y, 0.0) + dx
            x = x + np.linspace(-0.08, 0.08, y.size)
            ax.scatter(x, y, s=16, c=color, alpha=0.85, zorder=3)
            ax.hlines(np.median(y), dx - 0.18, dx + 0.18, color=color, lw=1.3)
        ax.axhline(10, color=GOLD, lw=0.8, ls="--", label="Static saddle (10%)")
        ax.axhline(1, color=TEAL, lw=0.7, ls=":", label="Yield drift (1%)")
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([-0.12, 0.12])
        ax.set_xticklabels(["Pulse", "Filtered noise"])
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_ylabel("Drift at first commitment (%)" if tag == "main_b0" else "")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    save(fig, "fig3_commitment_drift")


def fig_proxies(summary):
    # State-conditional error rates, nondeteriorating main cell.
    er = summary["main_b0"]["state_error_rates"]
    labels = ["Negative\ntangent", "0.20 s on\nthe cap", "Past static\nsaddle", "Cap-energy\nbarrier", "Drift\n> 10%", "Drift\n> 5%"]
    fp = [
        er["tangent_fp_given_tan"],
        er["sust_fp_given_sust"],
        er["saddle_fp_given_saddle"],
        er["energy_fp_given_energy"],
        er["d10_fp_given_d10"],
        er["d05_fp_given_d05"],
    ]
    fn = [
        er["tangent_fn_given_truth"],
        er["sust_fn_given_truth"],
        er["saddle_fn_given_truth"],
        er["energy_fn_given_truth"],
        er["d10_fn_given_truth"],
        er["d05_fn_given_truth"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(x - 0.16, np.array(fp) * 100, width=0.32, color=GOLD, label="Proxy false alarm")
    ax.bar(x + 0.16, np.array(fn) * 100, width=0.32, color=BLUE, label="Proxy miss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Conditional error (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.set_title("Conditional error rates of candidate markers", loc="left", fontsize=7)
    save(fig, "fig4_proxy_errors")


def main():
    rows = json.loads((RES / "rows.json").read_text(encoding="utf-8"))
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    stats = {
        "main_b0": family_stats(rows, "main_b0"),
        "main_b004": family_stats(rows, "main_b004"),
        "pooled_error_b0": summary["main_b0"]["state_error_rates"],
        "pooled_scale_b0": summary["main_b0"]["scale_ratio_vs_commitment"],
        "pooled_drift_b0": summary["main_b0"]["drift_at_first_commitment"],
        "pooled_reentry_b0": summary["main_b0"]["reentry_fraction_among_committed"],
        "pooled_drift_b004": summary["main_b004"]["drift_at_first_commitment"],
        "pooled_fy_b004": summary["main_b004"]["fy_ratio_at_first_commitment"],
        "pooled_error_b004": summary["main_b004"]["state_error_rates"],
    }
    (RES / "claim_numbers.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats["main_b0"], indent=2))
    print("phase grid", flush=True)
    fig_separatrix()
    print("traces", flush=True)
    fig_traces(rows)
    fig_drift(rows)
    fig_proxies(summary)
    print("figures done", flush=True)


if __name__ == "__main__":
    main()
