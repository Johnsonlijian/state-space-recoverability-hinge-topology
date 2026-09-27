"""
OpenSeesPy 3.7 shear frames with IMKBilin story springs.

The installed openseespywin wheel is a Python 3.12 binary whose wrapper
rejects every interpreter except 3.11. This script is run with the local
Python 3.12 environment at E:\\Users\\Monster\\.cursor\\venvs\\opensees312,
after that wrapper check is opened to 3.12.

Story springs match the R43 IMK shear frame: yield drift 0.8%, stability
coefficient 0.08, pre-cap plastic drift 1.5%, post-cap 5%, residual 0.15.
P-delta is a parallel elastic spring of stiffness -W/h.
"""

from __future__ import annotations

import json
from pathlib import Path

import openseespy.opensees as ops

ROOT = Path(__file__).resolve().parents[1]
GM_DIR = ROOT / "data" / "fema_p695"
OUT = ROOT / "results"

G = 9.80665
H = 3.5
M_FLOOR = 80_000.0
GAMMA_Y = 0.008
THETA = 0.08
FMAX_FY = 1.30
DP = 0.015
DPC = 0.05
FRES_FY = 0.15
# OpenSees documents Ref_Energy = Lamda * Fy. Match Et = 25*Fy*(dp*h).
LAMDA = 25.0 * DP * H
C_DET = 1.0
ZETA = 0.02
DT = 0.01
T_QUERY = 0.20
T_FREE = 8.0
ESCAPE = 0.12
SCALES = [0.35, 0.50, 0.70, 0.95, 1.30, 1.80, 2.50, 3.40]


def story_weights(nstory):
    return [(nstory - i) * M_FLOOR * G for i in range(nstory)]


def build(nstory, ag, dt_gm, scale, t_forced, t_free):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(0, 0.0, 0.0)
    ops.fix(0, 1, 1)
    for i in range(1, nstory + 1):
        ops.node(i, 0.0, 0.0)
        ops.mass(i, M_FLOOR, 0.0)
        ops.fix(i, 0, 1)
    weights = story_weights(nstory)
    for i, W in enumerate(weights, start=1):
        Fy = W * GAMMA_Y / THETA
        Ke_disp = (Fy / GAMMA_Y) / H
        dp = DP * H
        dpc = DPC * H
        du = (DP + DPC + 0.05) * H
        ops.uniaxialMaterial(
            "IMKBilin", 100 + i,
            Ke_disp, dp, dpc, du, Fy, FMAX_FY, FRES_FY,
            dp, dpc, du, Fy, FMAX_FY, FRES_FY,
            LAMDA, LAMDA, LAMDA, C_DET, C_DET, C_DET, 1.0, 1.0,
        )
        ops.uniaxialMaterial("Elastic", 200 + i, -W / H)
        ops.uniaxialMaterial("Parallel", i, 100 + i, 200 + i)
        ops.element("zeroLength", i, i - 1, i, "-mat", i, "-dir", 1)
    n_forced = max(1, int(round(t_forced / dt_gm)))
    n_free = int(round(t_free / dt_gm))
    n_forced = min(n_forced, len(ag) - 1)
    vals = [float(scale) * float(ag[k]) * G for k in range(n_forced)]
    vals.extend([0.0] * n_free)
    ops.timeSeries("Path", 1, "-dt", dt_gm, "-values", *vals)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    # mass-proportional damping from the elastic period, set after eigen
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 20, 0)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    w2 = ops.eigen(1)[0]
    w = w2 ** 0.5
    ops.rayleigh(2.0 * ZETA * w, 0.0, 0.0, 0.0)
    return n_forced, n_free, 2.0 * 3.1415926535 / w


def story_drifts(nstory):
    drifts = []
    u_prev = 0.0
    for i in range(1, nstory + 1):
        u = ops.nodeDisp(i, 1)
        drifts.append((u - u_prev) / H)
        u_prev = u
    return drifts


def max_drift(nstory):
    return max(abs(g) for g in story_drifts(nstory))


def run_cut(nstory, ag, scale, t_cut):
    n_forced, n_free, period = build(nstory, ag, DT, scale, t_cut, T_FREE)
    ok = ops.analyze(n_forced, DT)
    if ok != 0:
        return {"ok": False, "period": period}
    drift_at_cut = max_drift(nstory)
    peak_free = drift_at_cut
    steps = n_free
    committed = False
    # sample free vibration; stop early on escape
    sub = max(1, int(0.05 / DT))
    done = 0
    while done < steps:
        n = min(sub, steps - done)
        ok = ops.analyze(n, DT)
        done += n
        if ok != 0:
            return {"ok": False, "period": period, "drift_at_cut": drift_at_cut}
        d = max_drift(nstory)
        if d > peak_free:
            peak_free = d
        if d >= ESCAPE:
            committed = True
            break
    return {
        "ok": True,
        "period": period,
        "drift_at_cut": drift_at_cut,
        "peak_free": peak_free,
        "committed": committed,
    }


def first_commitment(nstory, ag, scale):
    duration = (len(ag) - 1) * DT
    t = T_QUERY
    while t < duration - 0.05:
        out = run_cut(nstory, ag, scale, t)
        if not out["ok"]:
            return {"failed": True, "t": t}
        if out["committed"]:
            return {
                "failed": False,
                "t": t,
                "drift": out["drift_at_cut"],
                "period": out["period"],
            }
        # skip ahead while the frame is still quiet
        if out["drift_at_cut"] < 0.004:
            t += 1.0
        else:
            t += T_QUERY
    return {"failed": False, "t": None, "drift": None, "period": None}


def load_records():
    out = []
    for f in sorted(GM_DIR.glob("FF*.txt")):
        vals = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            vals.append(float(line.split()[0]))
        if len(vals) > 100:
            out.append((f.stem, vals))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # period check on a one-step model
    ag = [0.0, 0.0, 0.0]
    _, _, period = build(4, ag, DT, 0.0, DT, 0.0)
    print("OpenSees", ops.version(), "4-story T1", round(period, 3), flush=True)
    records = load_records()
    print("records", len(records), flush=True)
    rows = []
    for nstory in (4, 8):
        for stem, ag in records:
            found = None
            for scale in SCALES:
                hit = first_commitment(nstory, ag, scale)
                if hit.get("failed"):
                    print(f"  {nstory} {stem} scale {scale} failed", flush=True)
                    continue
                if hit["drift"] is not None:
                    found = {"scale": scale, **hit}
                    break
            row = {
                "id": stem,
                "nstory": nstory,
                "scale": None if found is None else found["scale"],
                "drift_at_commitment": None if found is None else found["drift"],
                "t_s": None if found is None else found["t"],
                "period": None if found is None else found["period"],
            }
            rows.append(row)
            print(f"  {nstory}s {stem} drift={row['drift_at_commitment']} scale={row['scale']}", flush=True)
            (OUT / "opensees_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
