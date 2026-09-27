"""
One-bay moment frames in OpenSees 3.7.

IMKBilin rotational springs at both beam ends and at both column bases.
Columns and beams are elastic and stiff relative to the hinges.
A pinned leaning column with a PDelta transformation carries the floor weight.
Run with the Python 3.12 environment that can import openseespy.
"""

from __future__ import annotations

import json
from pathlib import Path

import openseespy.opensees as ops

ROOT = Path(__file__).resolve().parents[1]
GM_DIR = ROOT / "data" / "fema_p695"
OUT = ROOT / "results"

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
T_QUERY = 0.25
T_FREE = 8.0
ESCAPE = 0.10
SCALES = [0.40, 0.60, 0.90, 1.30, 1.90, 2.80, 4.00]
# When true, every column has an IMK hinge at each end. Story 0 keeps the
# existing base hinges and gains hinges at the column tops. Upper stories
# gain hinges at both ends. Each of those hinges uses the same yield moment
# as the beam hinge of that story. Default false preserves the reference census.
COLUMN_HINGES = False


def weights(nstory):
    return [(nstory - i) * M_FLOOR * G for i in range(nstory)]


def add_rot_spring(ele, n_i, n_j, My):
    Ke = My / THETA_Y
    du = DP + DPC + 0.05
    ops.uniaxialMaterial(
        "IMKBilin", ele,
        Ke, DP, DPC, du, My, FMAX, FRES,
        DP, DPC, du, My, FMAX, FRES,
        LAMDA, LAMDA, LAMDA, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    ops.element("zeroLength", ele, n_i, n_j, "-mat", ele, "-dir", 3)
    ops.equalDOF(n_i, n_j, 1, 2)


def build(nstory, ag, scale, t_forced, t_free, dt_gm=DT):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)
    # floor i = 0..nstory. Left column line x=0, right x=BAY, lean x=BAY+2.
    for i in range(nstory + 1):
        y = i * HST
        ops.node(10 * i + 1, 0.0, y)
        ops.node(10 * i + 2, BAY, y)
        ops.node(10 * i + 3, BAY + 2.0, y)
        if i > 0:
            ops.mass(10 * i + 1, 0.5 * M_FLOOR, 0.0, 0.0)
            ops.mass(10 * i + 2, 0.5 * M_FLOOR, 0.0, 0.0)
    # fixed supports, column-base hinge nodes coincide with the supports
    ops.node(1001, 0.0, 0.0)
    ops.node(1002, BAY, 0.0)
    ops.fix(1001, 1, 1, 1)
    ops.fix(1002, 1, 1, 1)
    ops.fix(3, 1, 1, 0)  # leaning base, pinned
    Wstory = weights(nstory)
    # base hinges sized to the first-story shear together with the first beam
    # V*h = 2*M_base + 2*M_beam => each My = V*h/4, V = 0.10*W
    My_base = 0.10 * Wstory[0] * HST / 4.0
    add_rot_spring(501, 1001, 1, My_base)
    add_rot_spring(502, 1002, 2, My_base)
    # columns. With COLUMN_HINGES, rotational springs are added at the top of
    # every column and at the bottom of every column above the base.
    for i in range(nstory):
        for side, x in ((1, 0.0), (2, BAY)):
            joint_bot = 10 * i + side
            joint_top = 10 * (i + 1) + side
            n_bot = joint_bot
            n_top = joint_top
            My_col = 0.10 * Wstory[i] * HST / 4.0
            if COLUMN_HINGES and i > 0:
                n_bot = 4000 + 10 * i + side
                ops.node(n_bot, x, i * HST)
                add_rot_spring(800 + 10 * i + side, joint_bot, n_bot, My_col)
            if COLUMN_HINGES:
                n_top = 5000 + 10 * i + side
                ops.node(n_top, x, (i + 1) * HST)
                add_rot_spring(900 + 10 * i + side, joint_top, n_top, My_col)
            etag = side if i == 0 else 10 + 2 * i + (side - 1)
            ops.element("elasticBeamColumn", etag, n_bot, n_top, A_MEM, E, I_MEM, 1)
    # beams with end hinges. Internal beam nodes.
    for i in range(1, nstory + 1):
        bl = 10 * i + 6
        br = 10 * i + 7
        y = i * HST
        ops.node(bl, 0.0, y)
        ops.node(br, BAY, y)
        My = 0.10 * Wstory[i - 1] * HST / 4.0
        add_rot_spring(600 + 2 * i, 10 * i + 1, bl, My)
        add_rot_spring(601 + 2 * i, 10 * i + 2, br, My)
        ops.element("elasticBeamColumn", 200 + i, bl, br, A_MEM, E, I_MEM, 2)
    # leaning column
    for i in range(nstory):
        ops.element(
            "elasticBeamColumn", 300 + i,
            10 * i + 3, 10 * (i + 1) + 3, A_LEAN, E, I_LEAN, 1,
        )
        ops.equalDOF(10 * (i + 1) + 1, 10 * (i + 1) + 3, 1)
    # gravity
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for i in range(1, nstory + 1):
        ops.load(10 * i + 3, 0.0, -M_FLOOR * G, 0.0)
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
    period = 2.0 * 3.1415926535 / (lam ** 0.5)
    ops.rayleigh(2.0 * ZETA * (lam ** 0.5), 0.0, 0.0, 0.0)
    return n_forced, period


def max_story_drift(nstory):
    peak = 0.0
    for i in range(nstory):
        u0 = ops.nodeDisp(10 * i + 1, 1)
        u1 = ops.nodeDisp(10 * (i + 1) + 1, 1)
        peak = max(peak, abs(u1 - u0) / HST)
    return peak


def pushover(nstory, roof_drift=0.12, nsteps=80):
    ag = [0.0, 0.0]
    built = build(nstory, ag, 0.0, 0.02, 0.0)
    if built is None:
        return None
    # replace transient with a roof displacement push after gravity
    ops.wipeAnalysis()
    ops.remove("loadPattern", 1)
    ops.timeSeries("Linear", 3)
    ops.pattern("Plain", 3, 3)
    # triangular lateral load proportional to floor height
    for i in range(1, nstory + 1):
        ops.load(10 * i + 1, float(i), 0.0, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 40, 0)
    ops.algorithm("Newton")
    d_roof = roof_drift * nstory * HST
    ops.integrator("DisplacementControl", 10 * nstory + 1, 1, d_roof / nsteps)
    ops.analysis("Static")
    curve = []
    for _ in range(nsteps):
        ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm("ModifiedNewton")
            ok = ops.analyze(1)
            ops.algorithm("Newton")
        if ok != 0:
            break
        # Global horizontal force in the two first-story columns.
        shear = -(ops.eleForce(1)[0] + ops.eleForce(2)[0])
        curve.append((max_story_drift(nstory), shear))
    return curve


def run_cut(nstory, ag, scale, t_cut):
    built = build(nstory, ag, scale, t_cut, T_FREE)
    if built is None:
        return {"ok": False}
    n_forced, period = built
    ok = ops.analyze(n_forced, DT)
    if ok != 0:
        return {"ok": False, "period": period}
    drift_cut = max_story_drift(nstory)
    peak = drift_cut
    committed = False
    done = 0
    steps = int(round(T_FREE / DT))
    sub = 5
    while done < steps:
        n = min(sub, steps - done)
        ok = ops.analyze(n, DT)
        done += n
        if ok != 0:
            return {"ok": False, "period": period, "drift_at_cut": drift_cut}
        d = max_story_drift(nstory)
        peak = max(peak, d)
        if d >= ESCAPE:
            committed = True
            break
    return {"ok": True, "period": period, "drift_at_cut": drift_cut, "committed": committed, "peak_free": peak}


def drift_history(nstory, ag, scale):
    duration = (len(ag) - 1) * DT
    built = build(nstory, ag, scale, duration, 0.0)
    if built is None:
        return None, None
    n_forced, period = built
    hist = []
    step = max(1, int(round(0.25 / DT)))
    done = 0
    while done < n_forced:
        n = min(step, n_forced - done)
        ok = ops.analyze(n, DT)
        done += n
        if ok != 0:
            break
        hist.append((done * DT, max_story_drift(nstory)))
    return hist, period


def first_commitment(nstory, ag, scale):
    hist, period = drift_history(nstory, ag, scale)
    if hist is None:
        return {"failed": True, "t": None}
    # First passage of each drift level. This brackets the commitment drift.
    levels = (0.008, 0.012, 0.018, 0.025, 0.035, 0.050, 0.070)
    candidates = []
    for level in levels:
        for t, drift in hist:
            if drift >= level:
                candidates.append(t)
                break
    for t in candidates:
        out = run_cut(nstory, ag, scale, t)
        if not out.get("ok"):
            return {"failed": True, "t": t}
        if out["committed"]:
            return {"failed": False, "t": t, "drift": out["drift_at_cut"], "period": period}
    return {"failed": False, "t": None, "drift": None, "period": period}


def load_records():
    out = []
    for f in sorted(GM_DIR.glob("FF*.txt")):
        vals = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                vals.append(float(line.split()[0]))
        if len(vals) > 100:
            out.append((f.stem, vals))
    return out


def net_zero_drift(curve):
    """First story drift after the peak where base shear has fallen through zero."""
    if not curve:
        return None
    peak_v = max(v for _, v in curve)
    seen_peak = False
    prev_v = curve[0][1]
    for drift, shear in curve:
        if shear >= 0.98 * peak_v:
            seen_peak = True
        if seen_peak and prev_v > 0.0 and shear <= 0.0:
            return drift
        prev_v = shear
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for nstory in (4, 8):
        curve = pushover(nstory)
        if not curve:
            raise SystemExit(f"pushover failed for {nstory}")
        z = net_zero_drift(curve)
        peak = max(curve, key=lambda p: p[1])
        print(
            f"pushover {nstory} points {len(curve)} peak_drift {peak[0]:.4f} V {peak[1]:.3e} net_zero {z}",
            flush=True,
        )
        (OUT / f"moment_pushover_{nstory}.json").write_text(
            json.dumps({"peak_drift": peak[0], "peak_shear": peak[1], "net_zero_drift": z, "curve": curve}),
            encoding="utf-8",
        )
    records = load_records()
    rows = []
    for nstory in (4, 8):
        for stem, ag in records:
            found = None
            for scale in SCALES:
                hit = first_commitment(nstory, ag, scale)
                if hit.get("failed"):
                    print(f"  fail {nstory} {stem} {scale}", flush=True)
                    continue
                if hit.get("drift") is not None:
                    found = hit
                    found["scale"] = scale
                    break
            row = {
                "id": stem,
                "nstory": nstory,
                "scale": None if found is None else found["scale"],
                "drift_at_commitment": None if found is None else found["drift"],
                "t_s": None if found is None else found["t"],
                "period": None if found is None else found.get("period"),
            }
            rows.append(row)
            print(f"  {nstory}s {stem} drift={row['drift_at_commitment']} scale={row['scale']}", flush=True)
            # Partial runs must not replace the reference census moment_rows.json.
            (OUT / "moment_rows_partial.json").write_text(json.dumps(rows), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
