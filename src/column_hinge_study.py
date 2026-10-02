"""Moment frames with IMK hinges at both ends of every column.

Beam hinges use the one-bay reference model. Each new column hinge uses
the same yield moment as the beam hinge of that story. Output is
results/column_hinge_rows.json and does not touch the reference census files.

Run with the Python 3.12 environment that can import openseespy.
"""

from __future__ import annotations

import argparse
import json

import opensees_moment_frame as mf
import refine_moment_bracket as rb
from refine_moment_bracket import bracket_one

mf.COLUMN_HINGES = True
# The pushover peak of this hinge map is near 1.4--1.6% drift, so the ladder
# starts below the 1.6% reference rung.
rb.LEVELS = (0.008, 0.010, 0.012, 0.014, 0.016, 0.020, 0.024, 0.028, 0.034, 0.044, 0.060, 0.080)


def pushover_pair():
    peaks = {}
    for nstory in (4, 8):
        curve = mf.pushover(nstory)
        if not curve:
            raise SystemExit(f"pushover failed for {nstory}")
        peak = max(curve, key=lambda p: p[1])
        z = mf.net_zero_drift(curve)
        peaks[nstory] = {"peak_drift": peak[0], "peak_shear": peak[1], "net_zero_drift": z}
        (mf.OUT / f"column_hinge_pushover_{nstory}.json").write_text(
            json.dumps({**peaks[nstory], "curve": curve}),
            encoding="utf-8",
        )
        print(
            f"pushover {nstory} peak_drift {peak[0]:.4f} V {peak[1]:.3e} net_zero {z}",
            flush=True,
        )
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pushover-only", action="store_true")
    args = parser.parse_args()
    mf.OUT.mkdir(parents=True, exist_ok=True)
    pushover_pair()
    if args.pushover_only:
        return
    out_path = mf.OUT / "column_hinge_rows.json"
    motions = dict(mf.load_records())
    rows = []
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8"))
    done = {(r["nstory"], r["id"]) for r in rows}
    for nstory in (4, 8):
        for stem, ag in motions.items():
            if (nstory, stem) in done:
                continue
            found = None
            status = "uncommitted"
            for scale in mf.SCALES:
                hit = bracket_one(nstory, ag, scale)
                print(
                    f"  try {nstory}s {stem} scale={scale} status={hit.get('status')} "
                    f"level={hit.get('committing_level')}",
                    flush=True,
                )
                if hit.get("status") == "indeterminate":
                    status = "indeterminate"
                    found = {"scale": scale, **hit}
                    break
                if hit.get("status") == "committed":
                    status = "committed"
                    found = {"scale": scale, **hit}
                    break
            row = {
                "id": stem,
                "nstory": nstory,
                "model": "column_hinges",
                "status": status,
                "scale": None if found is None else found.get("scale"),
                "committing_level": None if found is None else found.get("committing_level"),
                "drift_at_cut": None if found is None else found.get("drift_at_cut"),
                "last_recover_level": None if found is None else found.get("last_recover_level"),
                "t_s": None if found is None else found.get("t_s"),
                "period": None if found is None else found.get("period"),
            }
            rows.append(row)
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(
                f"{nstory}s {stem} status={status} drift={row['drift_at_cut']} scale={row['scale']}",
                flush=True,
            )
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
