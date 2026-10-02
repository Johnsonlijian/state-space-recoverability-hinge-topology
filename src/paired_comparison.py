"""Record-paired comparisons against the one-bay column-hinge reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
BASELINE = HERE.parent / "results"


def paired_delta(a: dict[str, float], b: dict[str, float], seed: int) -> dict:
    ids = sorted(set(a).intersection(b))
    x = np.asarray([b[k] - a[k] for k in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, size=(20_000, len(x)), replace=True), axis=1)
    return {
        "ids": ids,
        "n": len(ids),
        "median_delta": float(np.median(x)),
        "iqr_delta": [float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))],
        "bootstrap_95_median_delta": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "positive_count": int(np.sum(x > 0)),
        "negative_count": int(np.sum(x < 0)),
        "zero_count": int(np.sum(x == 0)),
        "deltas": {k: float(v) for k, v in zip(ids, x)},
    }


def main() -> None:
    full = json.loads((RES / "multibay_dynamic_full.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASELINE / "column_hinge_rows_recomputed.json").read_text(encoding="utf-8"))
    base = {x["id"]: x["drift_fine_recomputed"] for x in baseline if x["nstory"] == 4 and x.get("drift_fine_recomputed") is not None}
    models = {}
    for label in sorted({x["label"] for x in full}):
        models[label] = {x["id"]: x["drift_at_cut"] for x in full if x["label"] == label and x["status"] == "committed"}
    out = {
        "reference": "one-bay column-hinge 4-story baseline values",
        "paired_against_reference": {label: paired_delta(base, vals, 4610 + i) for i, (label, vals) in enumerate(models.items())},
        "paired_between_selected": {
            "2bay_free_member_minus_normalized": paired_delta(models["2bay_free_normalized"], models["2bay_free_member"], 4620),
            "2bay_free_member_minus_rigid_designinformed": paired_delta(models["2bay_rigid_designinformed"], models["2bay_free_member"], 4621),
        },
    }
    (RES / "multibay_paired_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# Multi-bay paired record comparisons",
        "",
        "Positive deltas mean the multi-bay census model commits at a larger drift than the one-bay reference for the same record. Bootstrap intervals are intervals for the paired median difference, not a probabilistic fragility estimate.",
        "",
        "| comparison | median paired delta | bootstrap 95% interval | positive / negative |",
        "|---|---:|---:|---:|",
    ]
    for label, x in out["paired_against_reference"].items():
        lines.append(f"| {label} - one-bay | {x['median_delta']*100:.2f} pp | {x['bootstrap_95_median_delta'][0]*100:.2f}--{x['bootstrap_95_median_delta'][1]*100:.2f} pp | {x['positive_count']} / {x['negative_count']} |")
    for label, x in out["paired_between_selected"].items():
        lines.append(f"| {label} | {x['median_delta']*100:.2f} pp | {x['bootstrap_95_median_delta'][0]*100:.2f}--{x['bootstrap_95_median_delta'][1]*100:.2f} pp | {x['positive_count']} / {x['negative_count']} |")
    (RES / "multibay_paired_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
