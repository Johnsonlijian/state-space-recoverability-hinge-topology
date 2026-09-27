"""
Multi-story shear frames with Ibarra-Medina-Krawinkler story springs,
driven by FEMA P695 far-field components.

The spring follows the modified IMK backbone (elastic, pre-capping hardening,
post-capping descent, residual) and deteriorates yield strength from hysteretic
energy at each reversal:
    E_t = Lambda * F_y0 * d_p
    beta = (E_half / (E_t - E_sum))^c
P-delta is a linear geometric story force -W*gamma, outside the spring.
OpenSees was not used: the Windows build failed to load.

Frozen before looking at aggregate commitment drifts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

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
LAMBDA_E = 25.0
C_DET = 1.0
ZETA = 0.02
DT = 0.005
T_SAMPLE = 0.20
H_CONT = 10.0
ESCAPE = 0.12
SCALES = np.geomspace(0.35, 4.0, 9)


@njit(cache=True)
def backbone(gamma, Fy, Ke, dp, dpc, fmax_fy, fres_fy):
    uy = Fy / Ke
    ucap = uy + dp
    Fcap = fmax_fy * Fy
    ures = ucap + dpc
    Fres = fres_fy * Fy
    au = abs(gamma)
    sgn = 1.0 if gamma >= 0.0 else -1.0
    if au <= uy:
        return sgn * Ke * au
    if au <= ucap:
        return sgn * (Fy + (Fcap - Fy) * (au - uy) / dp)
    if au <= ures:
        return sgn * (Fcap + (Fres - Fcap) * (au - ucap) / dpc)
    return sgn * Fres


@njit(cache=True)
def story_update(gamma, F, Fy, Ke0, Esum, Ehalf, sign_prev, Fy0, dp0, dpc, fmax_fy, fres_fy, Et, c_det, dgamma):
    """One IMK story step. Returns new gamma, F, Fy, Esum, Ehalf, sign, plastic flag."""
    gamma_new = gamma + dgamma
    sign_now = 0.0
    if dgamma > 1.0e-14:
        sign_now = 1.0
    elif dgamma < -1.0e-14:
        sign_now = -1.0
    else:
        sign_now = sign_prev
    # Deteriorate at reversal, using the half-cycle energy just completed.
    if sign_prev != 0.0 and sign_now != 0.0 and sign_now != sign_prev and Ehalf > 0.0:
        room = Et - Esum
        if room < 1.0e-9:
            beta = 0.95
        else:
            beta = (Ehalf / room) ** c_det
        if beta > 0.95:
            beta = 0.95
        if beta < 0.0:
            beta = 0.0
        Fy = Fy * (1.0 - beta)
        if Fy < 0.25 * Fy0:
            Fy = 0.25 * Fy0
        Esum = Esum + Ehalf
        Ehalf = 0.0
    # Current stiffness scales with remaining yield force so gamma_y stays put
    # only through Fy/Ke; Ke is reduced in proportion to Fy (unloading stiffness deterioration).
    Ke = Ke0 * (Fy / Fy0)
    # Shrink plastic plateau with strength so the cap drift moves inward.
    dp = dp0 * (Fy / Fy0)
    if dp < 0.25 * dp0:
        dp = 0.25 * dp0
    F_env = backbone(gamma_new, Fy, Ke, dp, dpc, fmax_fy, fres_fy)
    F_trial = F + Ke * dgamma
    # Stay inside the backbone envelope.
    if abs(F_trial) > abs(F_env) and F_trial * F_env >= 0.0:
        F_new = F_env
    elif F_trial * F_env < 0.0 and abs(F_trial) > abs(F_env):
        F_new = F_env
    else:
        F_new = F_trial
    # Plastic energy if the step was pulled onto the envelope.
    dF_elastic = Ke * dgamma
    if abs(dF_elastic) > 1.0e-12:
        scale = (F_new - F) / dF_elastic
        if scale < 0.0:
            scale = 0.0
        if scale > 1.0:
            scale = 1.0
        du_p = dgamma * (1.0 - scale)
    else:
        du_p = 0.0
    Ehalf = Ehalf + abs(0.5 * (F + F_new) * du_p)
    return gamma_new, F_new, Fy, Esum, Ehalf, sign_now


@njit(cache=True)
def net_tangent_negative(gamma, F, Fy, Ke0, Fy0, dp0, dpc, fmax_fy, fres_fy, W):
    Ke = Ke0 * (Fy / Fy0)
    dp = dp0 * (Fy / Fy0)
    if dp < 0.25 * dp0:
        dp = 0.25 * dp0
    # One-sided probe of the envelope slope.
    dg = 1.0e-4
    fp = backbone(gamma + dg, Fy, Ke, dp, dpc, fmax_fy, fres_fy)
    fm = backbone(gamma - dg, Fy, Ke, dp, dpc, fmax_fy, fres_fy)
    slope = (fp - fm) / (2.0 * dg)
    return 1 if (slope - W) < 0.0 else 0


@njit(cache=True)
def zero_input_frame(u, v, gamma, F, Fy, Esum, Ehalf, sign_s, Ke0, Fy0, W, m, c_damp, h, nstory):
    nstep = int(H_CONT / DT)
    quiet = 0
    for _ in range(nstep):
        acc = np.empty(nstory)
        # story drift rates
        for i in range(nstory):
            if i == 0:
                dgamma = (v[0] / h) * DT
            else:
                dgamma = ((v[i] - v[i - 1]) / h) * DT
            g2, f2, fy2, es2, eh2, sg2 = story_update(
                gamma[i], F[i], Fy[i], Ke0[i], Esum[i], Ehalf[i], sign_s[i],
                Fy0[i], DP, DPC, FMAX_FY, FRES_FY, LAMBDA_E * Fy0[i] * DP, C_DET, dgamma,
            )
            gamma[i] = g2
            F[i] = f2
            Fy[i] = fy2
            Esum[i] = es2
            Ehalf[i] = eh2
            sign_s[i] = sg2
            if abs(gamma[i]) >= ESCAPE:
                return 1
        for i in range(nstory):
            V = F[i] - W[i] * gamma[i]
            acc[i] = -V / m
            if i + 1 < nstory:
                Vn = F[i + 1] - W[i + 1] * gamma[i + 1]
                acc[i] = acc[i] + Vn / m
            acc[i] = acc[i] - c_damp * v[i]
        for i in range(nstory):
            v[i] = v[i] + acc[i] * DT
            u[i] = u[i] + v[i] * DT
        small = True
        for i in range(nstory):
            if abs(gamma[i]) > 0.45 * GAMMA_Y or abs(v[i]) > 0.05:
                small = False
        if small:
            quiet += 1
            if quiet * DT > 1.5:
                return 0
        else:
            quiet = 0
    return 0


@njit(cache=True)
def run_record(ag, dt_gm, scale, nstory, Ke0, Fy0, W, m, c_damp, h):
    """Return counts and first-event drifts/scales are handled outside; this returns a fixed vector.

    out[0] committed (0/1)
    out[1] drift at first commitment (max |story drift|)
    out[2] time of first commitment
    out[3] 1 if 2% drift occurred
    out[4] 1 if 5% drift occurred
    out[5] 1 if 10% occurred
    out[6] 1 if net tangent went negative
    out[7] drift when net tangent first went negative
    out[8] minimum Fy/Fy0 at commitment
    out[9] reentry
    """
    out = np.zeros(10)
    out[2] = 1.0e9
    out[7] = 1.0e9
    n = ag.shape[0]
    u = np.zeros(nstory)
    v = np.zeros(nstory)
    gamma = np.zeros(nstory)
    F = np.zeros(nstory)
    Fy = Fy0.copy()
    Esum = np.zeros(nstory)
    Ehalf = np.zeros(nstory)
    sign_s = np.zeros(nstory)
    next_sample = 0.0
    seen = 0
    reentry = 0
    # subsample the record onto DT by linear hold of the nearest gm sample
    t = 0.0
    nstep = int((n * dt_gm) / DT)
    for step in range(nstep):
        tg = t
        idx = int(tg / dt_gm)
        if idx >= n - 1:
            break
        ag_i = scale * G * ag[idx]
        # predictor using previous spring forces
        acc = np.empty(nstory)
        for i in range(nstory):
            V = F[i] - W[i] * gamma[i]
            acc[i] = -ag_i - V / m - c_damp * v[i]
            if i + 1 < nstory:
                Vn = F[i + 1] - W[i + 1] * gamma[i + 1]
                acc[i] = acc[i] + Vn / m
        for i in range(nstory):
            v[i] = v[i] + acc[i] * DT
            u[i] = u[i] + v[i] * DT
        for i in range(nstory):
            if i == 0:
                g_target = u[0] / h
            else:
                g_target = (u[i] - u[i - 1]) / h
            dgamma = g_target - gamma[i]
            g2, f2, fy2, es2, eh2, sg2 = story_update(
                gamma[i], F[i], Fy[i], Ke0[i], Esum[i], Ehalf[i], sign_s[i],
                Fy0[i], DP, DPC, FMAX_FY, FRES_FY, LAMBDA_E * Fy0[i] * DP, C_DET, dgamma,
            )
            gamma[i] = g2
            F[i] = f2
            Fy[i] = fy2
            Esum[i] = es2
            Ehalf[i] = eh2
            sign_s[i] = sg2
        t = t + DT
        max_abs = 0.0
        for i in range(nstory):
            if abs(gamma[i]) > max_abs:
                max_abs = abs(gamma[i])
        if max_abs >= 0.02 and out[3] == 0.0:
            out[3] = 1.0
        if max_abs >= 0.05 and out[4] == 0.0:
            out[4] = 1.0
        if max_abs >= 0.10 and out[5] == 0.0:
            out[5] = 1.0
        if out[6] == 0.0:
            for i in range(nstory):
                if net_tangent_negative(gamma[i], F[i], Fy[i], Ke0[i], Fy0[i], DP, DPC, FMAX_FY, FRES_FY, W[i]) == 1:
                    out[6] = 1.0
                    out[7] = max_abs
                    break
        if max_abs >= ESCAPE:
            break
        if t + 1.0e-12 >= next_sample:
            next_sample += T_SAMPLE
            # copy state for the continuation
            u2 = u.copy()
            v2 = v.copy()
            g2 = gamma.copy()
            f2s = F.copy()
            fy2 = Fy.copy()
            es2 = Esum.copy()
            eh2 = Ehalf.copy()
            sg2 = sign_s.copy()
            truth = 0
            if max_abs < 0.5 * GAMMA_Y:
                truth = 0
            else:
                truth = zero_input_frame(u2, v2, g2, f2s, fy2, es2, eh2, sg2, Ke0, Fy0, W, m, c_damp, h, nstory)
            if truth == 1 and out[0] == 0.0:
                out[0] = 1.0
                out[1] = max_abs
                out[2] = t
                ratio = 1.0
                for i in range(nstory):
                    r = Fy[i] / Fy0[i]
                    if r < ratio:
                        ratio = r
                out[8] = ratio
            if truth == 1:
                seen = 1
            elif seen == 1:
                reentry = 1
    out[9] = reentry
    return out


def frame_properties(nstory):
    Fy0 = np.zeros(nstory)
    Ke0 = np.zeros(nstory)
    W = np.zeros(nstory)
    for i in range(nstory):
        W[i] = (nstory - i) * M_FLOOR * G
        Fy0[i] = W[i] * GAMMA_Y / THETA
        Ke0[i] = Fy0[i] / GAMMA_Y
    # elastic period from the shear matrix, geometric stiffness included
    K = np.zeros((nstory, nstory))
    for i in range(nstory):
        k = Ke0[i] / H - W[i] / H
        K[i, i] += k
        if i > 0:
            K[i, i - 1] -= k
            K[i - 1, i] -= k
            K[i - 1, i - 1] += k
        if i + 1 < nstory:
            kn = Ke0[i + 1] / H - W[i + 1] / H
            K[i, i] += kn
    M = np.eye(nstory) * M_FLOOR
    evals = np.linalg.eigvals(np.linalg.solve(M, K))
    evals = np.sort(np.real(evals[np.real(evals) > 0]))
    w1 = float(np.sqrt(evals[0]))
    T1 = 2.0 * np.pi / w1
    c_damp = 2.0 * ZETA * w1
    return Fy0, Ke0, W, T1, c_damp


def self_test():
    Fy0, Ke0, W, T1, c_damp = frame_properties(1)
    # Monotonic push of one story, no dynamics: backbone then P-delta zero.
    gamma = 0.0
    F = 0.0
    Fy = Fy0[0]
    Esum = 0.0
    Ehalf = 0.0
    sgn = 0.0
    zeros = []
    prev = 0.0
    for k in range(800):
        dgamma = 0.0002
        gamma, F, Fy, Esum, Ehalf, sgn = story_update(
            gamma, F, Fy, Ke0[0], Esum, Ehalf, sgn, Fy0[0], DP, DPC, FMAX_FY, FRES_FY,
            LAMBDA_E * Fy0[0] * DP, C_DET, dgamma,
        )
        net = F - W[0] * gamma
        if prev > 0.0 and net <= 0.0:
            zeros.append(gamma)
            break
        prev = net
    # elastic stiffness check at small drift
    g2, f2, *_ = story_update(0.0, 0.0, Fy0[0], Ke0[0], 0.0, 0.0, 0.0, Fy0[0], DP, DPC, FMAX_FY, FRES_FY, 1.0e18, 1.0, 0.002)
    # story_update returns 6 values; the call above is wrong if I unpack wrong.
    return {
        "T1_4_will_be_computed_separately": T1,
        "monotonic_net_zero_drift": None if not zeros else float(zeros[0]),
        "small_force_over_Ke_gamma": float(f2 / (Ke0[0] * 0.002)),
    }


def load_records():
    files = sorted(GM_DIR.glob("FF*.txt"))
    records = []
    for f in files:
        ag = np.loadtxt(f)
        if ag.ndim != 1 or ag.size < 100:
            continue
        records.append((f.stem, np.ascontiguousarray(ag, dtype=np.float64)))
    return records


def first_scale(flags, scales):
    for s, flag in zip(scales, flags):
        if flag:
            return float(s)
    return None


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print("records", len(records), flush=True)
    rows = []
    summaries = {}
    for nstory in (4, 8):
        Fy0, Ke0, W, T1, c_damp = frame_properties(nstory)
        print(f"stories {nstory} T1 {T1:.3f}s", flush=True)
        # one-story monotonic zero from the same properties rule
        for stem, ag in records:
            flags_truth = []
            flags_d02 = []
            flags_d05 = []
            flags_d10 = []
            flags_tan = []
            detail = None
            for scale in SCALES:
                rec = run_record(
                    ag, 0.01, float(scale), nstory,
                    np.ascontiguousarray(Ke0), np.ascontiguousarray(Fy0), np.ascontiguousarray(W),
                    M_FLOOR, c_damp, H,
                )
                flags_truth.append(rec[0] > 0.5)
                flags_d02.append(rec[3] > 0.5)
                flags_d05.append(rec[4] > 0.5)
                flags_d10.append(rec[5] > 0.5)
                flags_tan.append(rec[6] > 0.5)
                if detail is None and rec[0] > 0.5:
                    detail = rec
            row = {
                "id": stem,
                "nstory": nstory,
                "T1_s": T1,
                "scale_truth": first_scale(flags_truth, SCALES),
                "scale_d02": first_scale(flags_d02, SCALES),
                "scale_d05": first_scale(flags_d05, SCALES),
                "scale_d10": first_scale(flags_d10, SCALES),
                "scale_tan": first_scale(flags_tan, SCALES),
                "drift_at_commitment": None if detail is None else float(detail[1]),
                "fy_ratio": None if detail is None else float(detail[8]),
                "reentry": None if detail is None else int(detail[9]),
                "drift_at_first_negative_tangent": None if detail is None else float(detail[7]),
            }
            rows.append(row)
            print(f"  {nstory}s {stem} drift={row['drift_at_commitment']} scale={row['scale_truth']}", flush=True)
        sub = [r for r in rows if r["nstory"] == nstory and r["drift_at_commitment"] is not None]
        drifts = np.array([r["drift_at_commitment"] for r in sub])
        summaries[str(nstory)] = {
            "n": len(records),
            "n_committed": int(drifts.size),
            "T1_s": T1,
            "drift_median": None if drifts.size == 0 else float(np.median(drifts)),
            "drift_q25": None if drifts.size == 0 else float(np.quantile(drifts, 0.25)),
            "drift_q75": None if drifts.size == 0 else float(np.quantile(drifts, 0.75)),
            "below_0.02": int(np.sum(drifts < 0.02)) if drifts.size else 0,
            "below_0.05": int(np.sum(drifts < 0.05)) if drifts.size else 0,
            "below_0.10": int(np.sum(drifts < 0.10)) if drifts.size else 0,
            "never_reached_0.10": int(sum(1 for r in rows if r["nstory"] == nstory and r["scale_d10"] is None and r["scale_truth"] is not None)),
        }
        (OUT / "frame_summary.json").write_text(json.dumps({"rows_so_far": len(rows), "summary": summaries}, indent=2), encoding="utf-8")
    (OUT / "frame_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (OUT / "frame_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("DONE", flush=True)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    # quick monotonic check before the census
    Fy0, Ke0, W, T1, c_damp = frame_properties(4)
    print("4-story T1", round(T1, 3), "c", round(c_damp, 4))
    gamma = 0.0
    F = 0.0
    Fy = float(Fy0[0])
    Esum = 0.0
    Ehalf = 0.0
    sgn = 0.0
    prev = 1.0
    hit = None
    for _ in range(1200):
        gamma, F, Fy, Esum, Ehalf, sgn = story_update(
            gamma, F, Fy, float(Ke0[0]), Esum, Ehalf, sgn, float(Fy0[0]),
            DP, DPC, FMAX_FY, FRES_FY, LAMBDA_E * float(Fy0[0]) * DP, C_DET, 0.00015,
        )
        net = F - float(W[0]) * gamma
        if prev > 0.0 and net <= 0.0 and gamma > 0.01:
            hit = gamma
            break
        prev = net
    print("monotonic net-zero drift", hit, "Fy_ratio", Fy / float(Fy0[0]))
    if hit is None or not (0.02 < hit < 0.09):
        raise SystemExit("backbone/P-delta check failed")
    run()
