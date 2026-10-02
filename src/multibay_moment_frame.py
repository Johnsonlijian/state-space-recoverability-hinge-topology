"""Controlled multi-bay IMK moment-frame model.

This is an analysis companion to the one-bay reference model.  It keeps
the same story height, mass, IMKBilin backbone and 21 local FEMA P695 records,
but adds a rigid-diaphragm floor constraint and an arbitrary number of bays.
The code is deliberately explicit about what is exploratory:

* ``member`` capacity keeps the one-bay hinge moment at every member.
* ``frame_normalized`` divides the hinge moments so the total first-story
  hinge budget is comparable across bay counts.
* ``strong_column`` is a design-informed ratio sweep, not a code-designed
  building or a real project design.

No experimental data are read by this module.  Results are simulation
extensions and must not be called experimental validation.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import openseespy.opensees as ops


ROUND = Path(__file__).resolve().parents[1]  # .../reproducibility/
GM_DIR = ROUND / "data" / "fema_p695"

G = 9.80665
HST = 3.5
BAY = 6.0
M_FLOOR = 80_000.0
E = 2.0e11
A_MEM = 0.05
I_MEM = 0.02
A_LEAN = 0.2
I_LEAN = 1.0e-6
THETA_Y = 0.006
FMAX = 1.15
DP = 0.025
DPC = 0.08
FRES = 0.20
LAMDA = 25.0 * DP
ZETA = 0.02
DT = 0.01
T_FREE = 8.0
ESCAPE = 0.10
SCALES = [0.40, 0.60, 0.90, 1.30, 1.90, 2.80, 4.00]
# Fine first-passage ladder used by the baseline rebracketing census.  A coarse
# 0.25-s probe can jump over a short recoverable interval and is therefore not
# suitable for record-level comparisons.
COMMITMENT_LEVELS = (
    0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.031, 0.034,
    0.038, 0.044, 0.046, 0.050, 0.052, 0.054, 0.058, 0.062,
    0.064, 0.070, 0.076, 0.080,
)


def weights(nstory: int) -> list[float]:
    return [(nstory - i) * M_FLOOR * G for i in range(nstory)]


def node_id(floor: int, line: int) -> int:
    """Main frame joint at a floor and column line."""
    return 10_000 + floor * 100 + line


def support_id(line: int) -> int:
    return 20_000 + line


def lean_id(floor: int) -> int:
    return 30_000 + floor


def spring_node_id(kind: str, story: int, line: int) -> int:
    base = {"bottom": 40_000, "top": 50_000}[kind]
    return base + story * 100 + line


def spring_ele_id(kind: str, story: int, line: int) -> int:
    base = {"bottom": 60_000, "top": 70_000, "base": 80_000, "beam": 90_000}[kind]
    return base + story * 100 + line


def beam_end_node_id(story: int, bay: int, side: int) -> int:
    return 100_000 + story * 1000 + bay * 10 + side


def beam_ele_id(story: int, bay: int) -> int:
    return 110_000 + story * 100 + bay


def col_ele_id(story: int, line: int) -> int:
    return 120_000 + story * 100 + line


def add_rot_spring(ele: int, n_i: int, n_j: int, my: float) -> None:
    ke = my / THETA_Y
    du = DP + DPC + 0.05
    ops.uniaxialMaterial(
        "IMKBilin",
        ele,
        ke,
        DP,
        DPC,
        du,
        my,
        FMAX,
        FRES,
        DP,
        DPC,
        du,
        my,
        FMAX,
        FRES,
        LAMDA,
        LAMDA,
        LAMDA,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    ops.element("zeroLength", ele, n_i, n_j, "-mat", ele, "-dir", 3)
    ops.equalDOF(n_i, n_j, 1, 2)


def _hinge_moment(
    w_story: float,
    *,
    n_bays: int,
    n_columns: int,
    capacity_mode: str,
    kind: str,
    column_ratio: float,
    story_factor: float,
) -> float:
    """Return a transparent hinge moment parameterization.

    The one-bay model uses 0.10 W h / 4 at each first-story base/beam end.
    ``frame_normalized`` preserves the corresponding total member budget as
    bay count changes; ``member`` keeps the one-bay member value.  Column
    hinges may be multiplied by a strong-column ratio.
    """
    base = 0.10 * w_story * HST / 4.0 * story_factor
    if capacity_mode == "member":
        scale = 1.0
    elif capacity_mode == "frame_normalized":
        # Two one-bay base hinges and two one-bay beam-end hinges define the
        # reference.  Preserve each of those total budgets across lines.
        denom = n_columns if kind == "column" else n_bays
        # Two one-bay base hinges and two one-bay beam ends define the
        # reference budget.  Beam-end budget therefore scales as 1/nbay;
        # column-base budget scales as 2/ncolumn.
        scale = (2.0 / denom) if kind == "column" else (1.0 / denom)
    else:
        raise ValueError(f"unknown capacity_mode={capacity_mode}")
    if kind == "column":
        scale *= column_ratio
    return base * scale


def _master_floor(floor: int) -> int:
    return node_id(floor, 0)


def build(
    nstory: int,
    n_bays: int,
    ag: list[float],
    scale: float,
    t_forced: float,
    t_free: float,
    *,
    column_hinges: bool = True,
    capacity_mode: str = "member",
    column_ratio: float = 1.0,
    story_factors: list[float] | None = None,
    rigid_diaphragm: bool = True,
    dt_gm: float = DT,
) -> dict | None:
    """Build a multi-bay model and return dynamic bookkeeping."""
    if n_bays < 1:
        raise ValueError("n_bays must be >= 1")
    n_columns = n_bays + 1
    if story_factors is None:
        story_factors = [1.0] * nstory
    if len(story_factors) != nstory:
        raise ValueError("story_factors must have one value per story")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)

    # Main joints, with total floor mass split across the column lines.  A
    # horizontal equalDOF at every floor acts as a rigid diaphragm.
    for floor in range(nstory + 1):
        y = floor * HST
        for line in range(n_columns):
            nd = node_id(floor, line)
            ops.node(nd, line * BAY, y)
            if floor > 0:
                ops.mass(nd, M_FLOOR / n_columns, 0.0, 0.0)
        if floor > 0 and rigid_diaphragm:
            master = _master_floor(floor)
            for line in range(1, n_columns):
                ops.equalDOF(master, node_id(floor, line), 1)

    # Separate base nodes allow a rotational spring at every column base.
    for line in range(n_columns):
        sup = support_id(line)
        ops.node(sup, line * BAY, 0.0)
        ops.fix(sup, 1, 1, 1)
    # Leaning column carries the total gravity load, as in the one-bay reference.
    for floor in range(nstory + 1):
        ops.node(lean_id(floor), n_bays * BAY + 2.0, floor * HST)
    ops.fix(lean_id(0), 1, 1, 0)

    w_story = weights(nstory)
    column_meta: list[dict] = []
    # Base hinges are always present.  This is the beam-end/column-base
    # counterfactual when column_hinges=False.
    for line in range(n_columns):
        my = _hinge_moment(
            w_story[0],
            n_bays=n_bays,
            n_columns=n_columns,
            capacity_mode=capacity_mode,
            kind="column",
            column_ratio=column_ratio,
            story_factor=story_factors[0],
        )
        eid = spring_ele_id("base", 0, line)
        add_rot_spring(eid, support_id(line), node_id(0, line), my)
        column_meta.append({"eid": eid, "story": 0, "line": line, "end": "base"})

    # Columns and optional IMK hinges at both ends of every column.
    for story in range(nstory):
        for line in range(n_columns):
            n_bot = node_id(story, line)
            n_top = node_id(story + 1, line)
            if column_hinges and story > 0:
                n_bot = spring_node_id("bottom", story, line)
                ops.node(n_bot, line * BAY, story * HST)
                my = _hinge_moment(
                    w_story[story],
                    n_bays=n_bays,
                    n_columns=n_columns,
                    capacity_mode=capacity_mode,
                    kind="column",
                    column_ratio=column_ratio,
                    story_factor=story_factors[story],
                )
                eid = spring_ele_id("bottom", story, line)
                add_rot_spring(eid, node_id(story, line), n_bot, my)
                column_meta.append({"eid": eid, "story": story, "line": line, "end": "bottom"})
            if column_hinges:
                n_top = spring_node_id("top", story, line)
                ops.node(n_top, line * BAY, (story + 1) * HST)
                my = _hinge_moment(
                    w_story[story],
                    n_bays=n_bays,
                    n_columns=n_columns,
                    capacity_mode=capacity_mode,
                    kind="column",
                    column_ratio=column_ratio,
                    story_factor=story_factors[story],
                )
                eid = spring_ele_id("top", story, line)
                add_rot_spring(eid, node_id(story + 1, line), n_top, my)
                column_meta.append({"eid": eid, "story": story, "line": line, "end": "top"})
            ops.element("elasticBeamColumn", col_ele_id(story, line), n_bot, n_top, A_MEM, E, I_MEM, 1)

    # Beam-end hinges and elastic spans at every floor and bay.
    beam_meta: list[dict] = []
    for floor in range(1, nstory + 1):
        for bay in range(n_bays):
            left = beam_end_node_id(floor, bay, 0)
            right = beam_end_node_id(floor, bay, 1)
            y = floor * HST
            ops.node(left, bay * BAY, y)
            ops.node(right, (bay + 1) * BAY, y)
            my = _hinge_moment(
                w_story[floor - 1],
                n_bays=n_bays,
                n_columns=n_columns,
                capacity_mode=capacity_mode,
                kind="beam",
                column_ratio=column_ratio,
                story_factor=story_factors[floor - 1],
            )
            eleft = spring_ele_id("beam", floor, 2 * bay)
            eright = spring_ele_id("beam", floor, 2 * bay + 1)
            add_rot_spring(eleft, node_id(floor, bay), left, my)
            add_rot_spring(eright, node_id(floor, bay + 1), right, my)
            beam_meta.extend(
                [
                    {"eid": eleft, "story": floor - 1, "bay": bay, "end": "left"},
                    {"eid": eright, "story": floor - 1, "bay": bay, "end": "right"},
                ]
            )
            ops.element("elasticBeamColumn", beam_ele_id(floor, bay), left, right, A_MEM, E, I_MEM, 2)

    # Pinned leaning column and total gravity load.
    for story in range(nstory):
        ops.element("elasticBeamColumn", 130_000 + story, lean_id(story), lean_id(story + 1), A_LEAN, E, I_LEAN, 1)
        ops.equalDOF(_master_floor(story + 1), lean_id(story + 1), 1)
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for floor in range(1, nstory + 1):
        ops.load(lean_id(floor), 0.0, -M_FLOOR * G, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-7, 40, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    ok = ops.analyze(10)
    if ok != 0:
        return None
    ops.loadConst("-time", 0.0)

    n_forced = min(len(ag) - 1, max(1, int(round(t_forced / dt_gm))))
    n_free = int(round(t_free / dt_gm))
    vals = [float(scale) * float(ag[k]) * G for k in range(n_forced)] + [0.0] * n_free
    ops.timeSeries("Path", 1, "-dt", dt_gm, "-values", *vals)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 25, 0)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    lam = ops.eigen(1)[0]
    period = 2.0 * math.pi / (lam**0.5)
    ops.rayleigh(2.0 * ZETA * (lam**0.5), 0.0, 0.0, 0.0)
    return {
        "n_forced": n_forced,
        "period": period,
        "column_meta": column_meta,
        "beam_meta": beam_meta,
        "n_columns": n_columns,
        "n_bays": n_bays,
    }


def max_story_drifts(nstory: int) -> list[float]:
    out = []
    for story in range(nstory):
        u0 = ops.nodeDisp(_master_floor(story), 1)
        u1 = ops.nodeDisp(_master_floor(story + 1), 1)
        out.append(abs(u1 - u0) / HST)
    return out


def hinge_rotations(meta: list[dict]) -> list[dict]:
    rows = []
    for item in meta:
        try:
            deformation = ops.eleResponse(item["eid"], "deformation")
            theta = float(deformation[0]) if deformation else float("nan")
        except Exception:
            theta = float("nan")
        rows.append({**item, "rotation": theta})
    return rows


def pushover(
    nstory: int,
    n_bays: int,
    *,
    column_hinges: bool = True,
    capacity_mode: str = "member",
    column_ratio: float = 1.0,
    story_factors: list[float] | None = None,
    rigid_diaphragm: bool = True,
    pattern: str = "triangular",
    roof_drift: float = 0.12,
    nsteps: int = 80,
) -> dict | None:
    built = build(
        nstory,
        n_bays,
        [0.0, 0.0],
        0.0,
        0.02,
        0.0,
        column_hinges=column_hinges,
        capacity_mode=capacity_mode,
        column_ratio=column_ratio,
        story_factors=story_factors,
        rigid_diaphragm=rigid_diaphragm,
    )
    if built is None:
        return None
    ops.wipeAnalysis()
    ops.remove("loadPattern", 1)
    ops.timeSeries("Linear", 3)
    ops.pattern("Plain", 3, 3)
    n_columns = n_bays + 1
    for floor in range(1, nstory + 1):
        coef = float(floor) if pattern == "triangular" else 1.0
        ops.load(_master_floor(floor), coef, 0.0, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 40, 0)
    ops.algorithm("Newton")
    d_roof = roof_drift * nstory * HST
    ops.integrator("DisplacementControl", _master_floor(nstory), 1, d_roof / nsteps)
    ops.analysis("Static")
    curve: list[dict] = []
    for step in range(nsteps):
        ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("ModifiedNewton")
            ok = ops.analyze(1)
            ops.algorithm("Newton")
        if ok != 0:
            break
        # The transformation constraint does not expose the zeroLength
        # support reactions consistently through nodeReaction on Windows.
        # First-story elastic-column end forces are the same global shear
        # measure used by the one-bay reference model.
        shear = -sum(float(ops.eleForce(col_ele_id(0, line))[0]) for line in range(n_columns))
        drifts = max_story_drifts(nstory)
        h = hinge_rotations(built["column_meta"] + built["beam_meta"])
        hcol = hinge_rotations(built["column_meta"])
        curve.append(
            {
                "roof_drift": abs(float(ops.nodeDisp(_master_floor(nstory), 1))) / (nstory * HST),
                "max_story_drift": max(drifts),
                "story_drifts": drifts,
                "base_shear": shear,
                "max_hinge_rotation": max((abs(x["rotation"]) for x in h if math.isfinite(x["rotation"])), default=0.0),
                "column_hinge_rotation": max((abs(x["rotation"]) for x in hcol if math.isfinite(x["rotation"])), default=0.0),
            }
        )
    return {
        "curve": curve,
        "peak": max(curve, key=lambda x: x["base_shear"]) if curve else None,
        "n_bays": n_bays,
        "nstory": nstory,
        "column_hinges": column_hinges,
        "capacity_mode": capacity_mode,
        "column_ratio": column_ratio,
        "pattern": pattern,
    }


def net_zero_drift(result: dict | None) -> float | None:
    if not result or not result.get("curve"):
        return None
    curve = result["curve"]
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


def load_records() -> list[tuple[str, list[float]]]:
    out = []
    for f in sorted(GM_DIR.glob("FF*.txt")):
        vals = [float(line.split()[0]) for line in f.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if len(vals) > 100:
            out.append((f.stem, vals))
    return out


def run_cut(
    nstory: int,
    n_bays: int,
    ag: list[float],
    scale: float,
    t_cut: float,
    *,
    column_hinges: bool = True,
    capacity_mode: str = "member",
    column_ratio: float = 1.0,
    story_factors: list[float] | None = None,
    rigid_diaphragm: bool = True,
    t_free: float = T_FREE,
    dt: float = DT,
) -> dict:
    built = build(
        nstory,
        n_bays,
        ag,
        scale,
        t_cut,
        t_free,
        column_hinges=column_hinges,
        capacity_mode=capacity_mode,
        column_ratio=column_ratio,
        story_factors=story_factors,
        rigid_diaphragm=rigid_diaphragm,
        dt_gm=dt,
    )
    if built is None:
        return {"ok": False}
    ok = ops.analyze(built["n_forced"], dt)
    if ok != 0:
        return {"ok": False, "period": built["period"]}
    drift_cut = max(max_story_drifts(nstory))
    drift_by_story = max_story_drifts(nstory)
    peak = drift_cut
    committed = False
    done = 0
    steps = int(round(t_free / dt))
    while done < steps:
        n = min(5, steps - done)
        ok = ops.analyze(n, dt)
        done += n
        if ok != 0:
            return {"ok": False, "period": built["period"], "drift_at_cut": drift_cut}
        drifts = max_story_drifts(nstory)
        d = max(drifts)
        peak = max(peak, d)
        if d >= ESCAPE:
            committed = True
            break
    h = hinge_rotations(built["column_meta"] + built["beam_meta"])
    col = [abs(x["rotation"]) for x in h if x.get("end") in ("bottom", "top") and math.isfinite(x["rotation"])]
    beam = [abs(x["rotation"]) for x in h if x.get("end") in ("left", "right") and math.isfinite(x["rotation"])]
    return {
        "ok": True,
        "period": built["period"],
        "drift_at_cut": drift_cut,
        "story_drifts_at_cut": drift_by_story,
        "committed": committed,
        "peak_free": peak,
        "max_column_rotation_at_cut": max(col, default=0.0),
        "max_beam_rotation_at_cut": max(beam, default=0.0),
        "column_rotation_median_at_cut": float(sorted(col)[len(col) // 2]) if col else 0.0,
        "beam_rotation_median_at_cut": float(sorted(beam)[len(beam) // 2]) if beam else 0.0,
    }


def first_commitment(
    nstory: int,
    n_bays: int,
    ag: list[float],
    scale: float,
    *,
    column_hinges: bool = True,
    capacity_mode: str = "member",
    column_ratio: float = 1.0,
    story_factors: list[float] | None = None,
    rigid_diaphragm: bool = True,
    dt: float = DT,
) -> dict:
    """Coarse commitment search using first-passage drift levels."""
    duration = (len(ag) - 1) * dt
    built = build(
        nstory,
        n_bays,
        ag,
        scale,
        duration,
        0.0,
        column_hinges=column_hinges,
        capacity_mode=capacity_mode,
        column_ratio=column_ratio,
        story_factors=story_factors,
        rigid_diaphragm=rigid_diaphragm,
        dt_gm=dt,
    )
    if built is None:
        return {"failed": True, "t": None}
    hist: list[tuple[float, float]] = []
    step = max(1, int(round(0.05 / dt)))
    done = 0
    while done < built["n_forced"]:
        n = min(step, built["n_forced"] - done)
        ok = ops.analyze(n, dt)
        done += n
        if ok != 0:
            # Keep the successful history before a later convergence failure;
        # this mirrors the reference census, which only needs the first
            # zero-input commitment passage and does not label a later tail
            # failure as the commitment itself.
            break
        hist.append((done * dt, max(max_story_drifts(nstory))))
    if not hist:
        return {"failed": True, "t": done * dt}
    for level in COMMITMENT_LEVELS:
        for t, drift in hist:
            if drift >= level:
                out = run_cut(
                    nstory,
                    n_bays,
                    ag,
                    scale,
                    t,
                    column_hinges=column_hinges,
                    capacity_mode=capacity_mode,
                    column_ratio=column_ratio,
                    story_factors=story_factors,
                    rigid_diaphragm=rigid_diaphragm,
                    dt=dt,
                )
                if not out.get("ok"):
                    return {"failed": True, "t": t}
                if out["committed"]:
                    return {"failed": False, "t": t, **out}
                break
    return {"failed": False, "t": None}


if __name__ == "__main__":
    print("records", len(load_records()), "OpenSees", ops.version())
