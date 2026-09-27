"""Reference oscillator on the same FEMA P695 components used for the frames."""

import json
from pathlib import Path

import numpy as np

import run_study as rs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SCALES = np.geomspace(0.25, 3.0, 8)


def main():
    k, c, uy, Fy0 = rs.oscillator(1.0, 0.10)
    rows = []
    for f in sorted((ROOT / "data" / "fema_p695").glob("FF*.txt")):
        ag = np.ascontiguousarray(np.loadtxt(f), dtype=np.float64)
        hit = None
        for scale in SCALES:
            rec = rs.score_motion(ag, float(scale), 0.01, rs.H, k, c, 0.10, 0.0, Fy0, uy)
            if rec[20] < 1.0e8:
                hit = {
                    "id": f.stem,
                    "scale": float(scale),
                    "drift": float(rec[27]),
                    "t_s": float(rec[20]),
                }
                break
        if hit is None:
            hit = {"id": f.stem, "scale": None, "drift": None, "t_s": None}
        rows.append(hit)
        print(hit, flush=True)
    drifts = [r["drift"] for r in rows if r["drift"] is not None]
    summary = {
        "n": len(rows),
        "n_committed": len(drifts),
        "median": float(np.median(drifts)) if drifts else None,
        "q25": float(np.quantile(drifts, 0.25)) if drifts else None,
        "q75": float(np.quantile(drifts, 0.75)) if drifts else None,
        "below_0.05": int(sum(d < 0.05 for d in drifts)),
        "below_0.10": int(sum(d < 0.10 for d in drifts)),
    }
    (OUT / "sdof_recorded.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
