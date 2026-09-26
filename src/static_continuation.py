"""Adaptive static continuation for selected multi-bay topology cases.

The first static census deliberately used a short, fixed DisplacementControl
run.  This companion script uses smaller increments and an algorithm fallback
sequence so that post-peak descending branches can be followed farther when the
degrading IMK springs make a single Newton step fail.  It remains an exploratory
model diagnostic: the continuation is not a replacement for a validated
nonlinear static solver protocol.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multibay_moment_frame as mf  # noqa: E402

OUT = HERE.parent / "results"


def _setup_pushover(
    nstory: int,
    n_bays: int,
    *,
    rigid: bool,
    capacity_mode: str,
    column_ratio: float,
    story_factor: float,
    pattern: str,
) -> dict | None:
    factors = [story_factor] + [1.0] * (nstory - 1)
    built = mf.build(
        nstory,
        n_bays,
        [0.0, 0.0],
        0.0,
        0.02,
        0.0,
        column_hinges=True,
        capacity_mode=capacity_mode,
        column_ratio=column_ratio,
        story_factors=factors,
        rigid_diaphragm=rigid,
    )
    if built is None:
        return None
    mf.ops.wipeAnalysis()
    mf.ops.remove("loadPattern", 1)
    mf.ops.timeSeries("Linear", 3)
    mf.ops.pattern("Plain", 3, 3)
    for floor in range(1, nstory + 1):
        coef = float(floor) if pattern == "triangular" else 1.0
        mf.ops.load(mf._master_floor(floor), coef, 0.0, 0.0)
    mf.ops.constraints("Transformation")
    mf.ops.numberer("RCM")
    mf.ops.system("BandGeneral")
    mf.ops.test("NormDispIncr", 1.0e-6, 80, 0)
    return built


def _try_increment() -> tuple[int, str]:
    """Try a conservative fallback sequence without changing the load step."""
    attempts = (
        ("Newton", lambda: mf.ops.algorithm("Newton")),
        ("NewtonLineSearch", lambda: mf.ops.algorithm("NewtonLineSearch", "-type", "Bisection")),
        ("KrylovNewton", lambda: mf.ops.algorithm("KrylovNewton")),
        ("ModifiedNewton", lambda: mf.ops.algorithm("ModifiedNewton")),
    )
    last = -1
    for name, set_algorithm in attempts:
        try:
            set_algorithm()
            code = int(mf.ops.analyze(1))
        except Exception:
            code = -99
        if code == 0:
            return 0, name
        last = code
    return last, "all_fallbacks_failed"


def _row(nstory: int, built: dict, n_bays: int) -> dict:
    n_columns = n_bays + 1
    shear = -sum(float(mf.ops.eleForce(mf.col_ele_id(0, line))[0]) for line in range(n_columns))
    drifts = mf.max_story_drifts(nstory)
    all_hinges = mf.hinge_rotations(built["column_meta"] + built["beam_meta"])
    col_hinges = mf.hinge_rotations(built["column_meta"])
    return {
        "roof_drift": abs(float(mf.ops.nodeDisp(mf._master_floor(nstory), 1))) / (nstory * mf.HST),
        "max_story_drift": max(drifts),
        "story_drifts": drifts,
        "base_shear": shear,
        "max_hinge_rotation": max((abs(x["rotation"]) for x in all_hinges if math.isfinite(x["rotation"])), default=0.0),
        "column_hinge_rotation": max((abs(x["rotation"]) for x in col_hinges if math.isfinite(x["rotation"])), default=0.0),
    }


def _net_zero(curve: list[dict]) -> float | None:
    if not curve:
        return None
    peak_v = max(row["base_shear"] for row in curve)
    seen_peak = False
    prev_v = curve[0]["base_shear"]
    for row in curve:
        if row["base_shear"] >= 0.98 * peak_v:
            seen_peak = True
        if seen_peak and prev_v > 0.0 and row["base_shear"] <= 0.0:
            return row["max_story_drift"]
        prev_v = row["base_shear"]
    return None


def run_case(
    nstory: int,
    n_bays: int,
    *,
    rigid: bool,
    capacity_mode: str,
    column_ratio: float,
    story_factor: float,
    pattern: str,
    roof_drift: float = 0.24,
    nsteps: int = 320,
) -> dict:
    built = _setup_pushover(
        nstory,
        n_bays,
        rigid=rigid,
        capacity_mode=capacity_mode,
        column_ratio=column_ratio,
        story_factor=story_factor,
        pattern=pattern,
    )
    base = {
        "nstory": nstory,
        "n_bays": n_bays,
        "rigid_diaphragm": rigid,
        "capacity_mode": capacity_mode,
        "column_ratio": column_ratio,
        "first_story_factor": story_factor,
        "pattern": pattern,
        "roof_drift_target": roof_drift,
        "requested_steps": nsteps,
    }
    if built is None:
        return {**base, "status": "failed_to_build", "curve": []}

    increment = roof_drift * nstory * mf.HST / nsteps
    mf.ops.integrator("DisplacementControl", mf._master_floor(nstory), 1, increment)
    mf.ops.analysis("Static")
    curve: list[dict] = []
    algorithms: list[str] = []
    failure_code = None
    for _ in range(nsteps):
        code, algorithm = _try_increment()
        if code != 0:
            failure_code = code
            break
        algorithms.append(algorithm)
        curve.append(_row(nstory, built, n_bays))

    peak = max(curve, key=lambda x: x["base_shear"]) if curve else None
    return {
        **base,
        "status": "complete_target" if len(curve) == nsteps else "stopped_after_solver_failure",
        "steps": len(curve),
        "failure_code": failure_code,
        "fallback_counts": {name: algorithms.count(name) for name in sorted(set(algorithms))},
        "peak": peak,
        "net_zero_story_drift": _net_zero(curve),
        "last": curve[-1] if curve else None,
        "curve": curve,
    }


def main() -> None:
    cases = [
        # One-bay replay is a frozen-reference check for the continuation.
        dict(nstory=4, n_bays=1, rigid=False, capacity_mode="member", column_ratio=1.0, story_factor=1.0, pattern="triangular"),
        dict(nstory=4, n_bays=2, rigid=False, capacity_mode="member", column_ratio=1.0, story_factor=1.0, pattern="triangular"),
        dict(nstory=4, n_bays=3, rigid=False, capacity_mode="member", column_ratio=1.0, story_factor=1.0, pattern="triangular"),
        dict(nstory=4, n_bays=2, rigid=False, capacity_mode="frame_normalized", column_ratio=1.0, story_factor=1.0, pattern="triangular"),
        dict(nstory=4, n_bays=2, rigid=True, capacity_mode="frame_normalized", column_ratio=1.5, story_factor=1.25, pattern="triangular"),
    ]
    rows = []
    for case in cases:
        print(
            f"continuation {case['nstory']}s {case['n_bays']}bay rigid={case['rigid']} "
            f"{case['capacity_mode']} col={case['column_ratio']} f1={case['story_factor']}",
            flush=True,
        )
        result = run_case(**case)
        rows.append(result)
        (OUT / "multibay_static_continuation_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        peak_value = None if result.get("peak") is None else result["peak"]["max_story_drift"]
        print(
            f"  {result['status']} steps={result.get('steps')} "
            f"peak={peak_value} "
            f"netzero={result.get('net_zero_story_drift')}",
            flush=True,
        )
    (OUT / "multibay_static_continuation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} continuation rows", flush=True)


if __name__ == "__main__":
    main()
