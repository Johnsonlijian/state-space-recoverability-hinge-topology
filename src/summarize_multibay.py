"""Summarize multi-bay release results with finite-sample diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"


def quantile_summary(values: list[float], seed: int = 4601) -> dict:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, size=(20_000, len(x)), replace=True), axis=1)
    loo = np.asarray([np.median(np.delete(x, i)) for i in range(len(x))])
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "iqr": [float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))],
        "bootstrap_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "loo_range": [float(np.min(loo)), float(np.max(loo))],
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    rows = json.loads((RES / "multibay_dynamic_full.json").read_text(encoding="utf-8"))
    static_rows = json.loads((RES / "multibay_static.json").read_text(encoding="utf-8"))
    # Frozen one-bay reference, stored with the derived results.
    baseline = json.loads((RES / "column_hinge_rows_recomputed.json").read_text(encoding="utf-8"))
    reference = {
        n: [x["drift_fine_recomputed"] for x in baseline if x["nstory"] == n and x.get("drift_fine_recomputed") is not None]
        for n in (4, 8)
    }

    summary: dict = {
        "provenance": {
            "dynamic_source": "multibay_dynamic_full.json",
            "static_source": "multibay_static.json",
            "reference_source": "column_hinge_rows_recomputed.json",
            "bootstrap_resamples": 20_000,
            "seed": 4601,
        },
        "reference_one_bay": {str(n): quantile_summary(v) for n, v in reference.items()},
        "models": {},
    }
    for label in sorted({x["label"] for x in rows}):
        x = [z for z in rows if z["label"] == label and z["status"] == "committed"]
        drifts = [z["drift_at_cut"] for z in x]
        col = [z["max_column_rotation_at_cut"] for z in x]
        beam = [z["max_beam_rotation_at_cut"] for z in x]
        ratios = [a / b if b > 0 else None for a, b in zip(col, beam)]
        ratios = [v for v in ratios if v is not None]
        max_story = Counter(
            max(range(len(z["story_drifts_at_cut"])), key=lambda i: z["story_drifts_at_cut"][i]) + 1
            for z in x
        )
        localization = [
            max(z["story_drifts_at_cut"]) / float(np.median(z["story_drifts_at_cut"]))
            for z in x
        ]
        static_match = [z for z in static_rows if z["nstory"] == 4 and z["pattern"] == "triangular"]
        if label == "2bay_free_member":
            st = [z for z in static_match if z["n_bays"] == 2 and not z["rigid_diaphragm"] and z["capacity_mode"] == "member" and z["column_ratio"] == 1.0 and z["first_story_factor"] == 1.0][0]
        elif label == "3bay_free_member":
            st = [z for z in static_match if z["n_bays"] == 3 and not z["rigid_diaphragm"] and z["capacity_mode"] == "member" and z["column_ratio"] == 1.0 and z["first_story_factor"] == 1.0][0]
        elif label == "2bay_free_normalized":
            st = [z for z in static_match if z["n_bays"] == 2 and not z["rigid_diaphragm"] and z["capacity_mode"] == "frame_normalized" and z["column_ratio"] == 1.0 and z["first_story_factor"] == 1.0][0]
        else:
            st = [z for z in static_match if z["n_bays"] == 2 and z["rigid_diaphragm"] and z["capacity_mode"] == "frame_normalized" and z["column_ratio"] == 1.5 and z["first_story_factor"] == 1.25][0]
        summary["models"][label] = {
            "dynamic_commitment_drift": quantile_summary(drifts),
            "max_column_rotation": quantile_summary(col),
            "max_beam_rotation": quantile_summary(beam),
            "column_to_beam_rotation_ratio": quantile_summary(ratios),
            "max_story_counts": {str(k): int(v) for k, v in sorted(max_story.items())},
            "story_localization_ratio": quantile_summary(localization),
            "static": {
                "status": st["status"],
                "converged_steps": st["steps"],
                "peak_story_drift": st["peak_story_drift"],
                "peak_base_shear": st["peak_base_shear"],
                "net_zero_story_drift": st["net_zero_story_drift"],
                "net_zero_reached_before_solver_stop": st["net_zero_story_drift"] is not None,
            },
            "counts": {
                "n_committed": len(x),
                "n_below_2pct": sum(z["drift_at_cut"] < 0.02 for z in x),
                "n_below_5pct": sum(z["drift_at_cut"] < 0.05 for z in x),
                "n_at_or_below_static_peak": sum(z["drift_at_cut"] <= st["peak_story_drift"] for z in x),
            },
        }
    (RES / "multibay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Multi-bay and design-informed topology summary",
        "",
        "This is a controlled simulation extension using the same 21 FEMA P695 far-field records and the same IMKBilin rotational backbone. It is not experimental validation and is not a code-designed building study.",
        "",
        "## Dynamic zero-input release census (4-story, 21 records)",
        "",
        "| model | median commitment drift | IQR | bootstrap 95% interval | max-column / max-beam rotation median | records at or below static peak |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, x in summary["models"].items():
        d = x["dynamic_commitment_drift"]
        r = x["column_to_beam_rotation_ratio"]
        lines.append(
            f"| {label} | {d['median']*100:.2f}% | {d['iqr'][0]*100:.2f}--{d['iqr'][1]*100:.2f}% | {d['bootstrap_95'][0]*100:.2f}--{d['bootstrap_95'][1]*100:.2f}% | {r['median']:.2f} | {x['counts']['n_at_or_below_static_peak']}/{x['counts']['n_committed']} |"
        )
    lines.extend(
        [
            "",
            "## Static path diagnostics",
            "",
            "Static curves use the same 0.12 roof-drift target and are reported up to the first nonconvergent step. A reported net-zero value is a positive-to-negative base-shear crossing reached before that solver stop; `not reached` is not a claim that no crossing exists.",
            "",
            "| model | static peak drift | peak base shear | net-zero drift | path status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for label, x in summary["models"].items():
        s = x["static"]
        nz = "" if s["net_zero_story_drift"] is None else f"{s['net_zero_story_drift']*100:.2f}%"
        lines.append(
            f"| {label} | {s['peak_story_drift']*100:.2f}% | {s['peak_base_shear']/1000:.1f} kN | {nz} | {s['status']} ({s['converged_steps']} steps) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The full 21-record runs support a topology/configuration sensitivity observation: adding bays changes the relationship between static peak strength and zero-input commitment, while normalizing the total hinge budget changes the local rotation partition. The result is an exploratory generalization of the one-bay model, not evidence for all multi-bay or code-designed frames. No experimental specimen or field record is included in this round.",
            "",
            "In the 4-story multi-bay runs, the largest story drift at commitment was the fourth (top) story for 19--20 of 21 records in the three analyzed configurations; the median maximum-to-story-median drift ratio was 4.0--6.0. This is a model-localization diagnostic, not a general rule for real buildings.",
        ]
    )
    (RES / "multibay_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
