"""Tighten the moment-frame commitment drift at the already identified scale.

The coarse census samples story drift every 0.25 s and tests passages at
0.8, 1.2, 1.8, 2.5, 3.5, 5.0, and 7.0 percent. That grid steps over the
pushover peak at 3.2 percent. This script keeps the committing scale from
the reference census and tests a finer passage ladder, sampled every 0.05 s.
It writes results/moment_bracket.json and does not touch moment_rows.json.

Run with the Python 3.12 environment that can import openseespy.
"""

from __future__ import annotations

import argparse
import json

import openseespy.opensees as ops

from opensees_moment_frame import DT, OUT, ROOT, load_records, run_cut

LEVELS = (0.016, 0.020, 0.024, 0.028, 0.031, 0.034, 0.038, 0.044, 0.052, 0.062, 0.080)
CENSUS = ROOT / "results" / "moment_rows_census_20260924.json"
OUT_PATH = OUT / "moment_bracket.json"


def history_fine(nstory, ag, scale, sample=0.05):
    duration = (len(ag) - 1) * DT
    # Local import keeps the coarse drift_history unchanged.
    from opensees_moment_frame import build, max_story_drift

    built = build(nstory, ag, scale, duration, 0.0)
    if built is None:
        return None, None
    n_forced, period = built
    hist = []
    step = max(1, int(round(sample / DT)))
    done = 0
    truncated = False
    while done < n_forced:
        n = min(step, n_forced - done)
        ok = ops.analyze(n, DT)
        done += n
        if ok != 0:
            truncated = True
            break
        hist.append((done * DT, max_story_drift(nstory)))
    return hist, period, truncated


def bracket_one(nstory, ag, scale):
    packed = history_fine(nstory, ag, scale)
    if packed[0] is None:
        return {"status": "build_failed", "period": None}
    hist, period, truncated = packed
    last_recover = None
    for level in LEVELS:
        t_hit = None
        for t, drift in hist:
            if drift >= level:
                t_hit = t
                break
        if t_hit is None:
            continue
        out = run_cut(nstory, ag, scale, t_hit)
        if not out.get("ok"):
            return {
                "status": "indeterminate",
                "level": level,
                "t_s": t_hit,
                "period": period,
                "last_recover_level": last_recover,
            }
        if out["committed"]:
            return {
                "status": "committed",
                "committing_level": level,
                "drift_at_cut": out["drift_at_cut"],
                "t_s": t_hit,
                "period": period,
                "last_recover_level": last_recover,
                "peak_free": out.get("peak_free"),
            }
        last_recover = level
    if truncated:
        return {
            "status": "indeterminate",
            "period": period,
            "last_recover_level": last_recover,
            "reason": "history_truncated_before_commitment",
        }
    return {
        "status": "uncommitted_on_ladder",
        "period": period,
        "last_recover_level": last_recover,
        "max_drift_seen": hist[-1][1] if hist else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--record", default="")
    parser.add_argument("--nstory", type=int, default=0)
    args = parser.parse_args()
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    motions = dict(load_records())
    rows = []
    if OUT_PATH.exists() and not args.record:
        # Resume without repeating completed keys.
        rows = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    done = {(r["nstory"], r["id"]) for r in rows}
    n_run = 0
    for src in census:
        key = (src["nstory"], src["id"])
        if key in done:
            continue
        if args.record and src["id"] != args.record:
            continue
        if args.nstory and src["nstory"] != args.nstory:
            continue
        if src["scale"] is None:
            row = {
                "id": src["id"],
                "nstory": src["nstory"],
                "scale": None,
                "status": "uncommitted_in_census",
                "coarse_drift": src["drift_at_commitment"],
            }
        else:
            hit = bracket_one(src["nstory"], motions[src["id"]], src["scale"])
            row = {
                "id": src["id"],
                "nstory": src["nstory"],
                "scale": src["scale"],
                "coarse_drift": src["drift_at_commitment"],
                **hit,
            }
        rows.append(row)
        OUT.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(
            f"{row['nstory']}s {row['id']} status={row['status']} "
            f"level={row.get('committing_level')} drift={row.get('drift_at_cut')} "
            f"recover={row.get('last_recover_level')}",
            flush=True,
        )
        n_run += 1
        if args.limit and n_run >= args.limit:
            break
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
