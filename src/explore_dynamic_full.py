"""Full 21-record multi-bay release census for the selected topology cells."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multibay_moment_frame as mf  # noqa: E402

OUT = HERE.parent / "results"
PATH = OUT / "multibay_dynamic_full.json"


CONFIGS = [
    {
        "label": "2bay_free_member",
        "nstory": 4,
        "n_bays": 2,
        "rigid_diaphragm": False,
        "capacity_mode": "member",
        "column_ratio": 1.0,
        "story_factors": [1.0] * 4,
    },
    {
        "label": "3bay_free_member",
        "nstory": 4,
        "n_bays": 3,
        "rigid_diaphragm": False,
        "capacity_mode": "member",
        "column_ratio": 1.0,
        "story_factors": [1.0] * 4,
    },
    {
        "label": "2bay_free_normalized",
        "nstory": 4,
        "n_bays": 2,
        "rigid_diaphragm": False,
        "capacity_mode": "frame_normalized",
        "column_ratio": 1.0,
        "story_factors": [1.0] * 4,
    },
    {
        "label": "2bay_rigid_designinformed",
        "nstory": 4,
        "n_bays": 2,
        "rigid_diaphragm": True,
        "capacity_mode": "frame_normalized",
        "column_ratio": 1.5,
        "story_factors": [1.25, 1.0, 1.0, 1.0],
    },
]


def run_one(cfg: dict, rid: str, ag: list[float]) -> dict:
    t0 = time.time()
    found = None
    for scale in mf.SCALES:
        hit = mf.first_commitment(
            cfg["nstory"], cfg["n_bays"], ag, scale,
            column_hinges=True,
            capacity_mode=cfg["capacity_mode"],
            column_ratio=cfg["column_ratio"],
            story_factors=cfg["story_factors"],
            rigid_diaphragm=cfg["rigid_diaphragm"],
        )
        if not hit.get("failed") and hit.get("committed"):
            found = {"scale": scale, **hit}
            break
    row = {**cfg, "id": rid, "status": "committed" if found else "not_committed_on_grid", "elapsed_s": time.time() - t0}
    if found:
        for key in (
            "scale", "t", "period", "drift_at_cut", "story_drifts_at_cut", "peak_free",
            "max_column_rotation_at_cut", "max_beam_rotation_at_cut",
            "column_rotation_median_at_cut", "beam_rotation_median_at_cut",
        ):
            if key in found:
                row[key] = found[key]
    return row


def main() -> None:
    records = dict(mf.load_records())
    rows = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else []
    done = {(x["label"], x["id"]) for x in rows}
    for cfg in CONFIGS:
        for rid, ag in records.items():
            if (cfg["label"], rid) in done:
                continue
            row = run_one(cfg, rid, ag)
            rows.append(row)
            PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(
                cfg["label"], rid, row["status"],
                "scale", row.get("scale"),
                "drift", None if row.get("drift_at_cut") is None else round(row["drift_at_cut"] * 100, 2),
                "sec", round(row["elapsed_s"], 1), flush=True,
            )
    print(f"wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
