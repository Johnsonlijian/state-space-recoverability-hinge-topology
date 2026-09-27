"""
Gravity-commitment separatrix study.

Pre-specified analysis protocol:
- Reference outcome is finite-horizon zero-input continuation.
- Proxies: 5% drift, 10% drift, negative net tangent, sustained cap
  sojourn of 0.20 s, current static saddle, monotonic cap-energy barrier.
- Oscillator: unit mass, elastic-perfectly plastic spring, linear P-Delta.
- theta is an imposed stability coefficient, not P = mg of a specific building.
- Main cell: T=1 s, theta=0.10, uy/h=0.01, h=3.5 m, zeta=0.05.
- beta=0 and beta=0.04 strength deterioration.
- Motions: 20 Gabor pulses + 20 Kanai-Tajimi records, PGA-normalized,
  scaled on a pre-specified geometric grid. Synthetic by design.
- Escape drift 0.40; continuation horizon 8 s; sample 0.10 s; dt=0.001 s.
- A self-test against the analytical cap separatrix must pass before the census.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

G = 9.80665
H = 3.5
UY_DRIFT = 0.01
ZETA = 0.05
DT = 0.001
T_SAMPLE = 0.10
H_CONT = 8.0
ESCAPE_DRIFT = 0.40
SUSTAIN_S = 0.20
FY_FLOOR = 0.25
SEED = 20260924

SCALES = np.geomspace(0.08, 2.5, 12)


@njit(cache=True)
def uc_of(Fy, k, theta):
    return Fy / (theta * k)


@njit(cache=True)
def step(u, v, S, Fy, Lam, ag, dt, k, c, theta, beta, Fy0, uy):
    R = S - theta * k * u
    a = -ag - c * v - R
    v_new = v + a * dt
    u_new = u + v_new * dt
    du = u_new - u
    Strial = S + k * du
    plastic = 0.0
    if Strial > Fy:
        plastic = (Strial - Fy) / k
        S_new = Fy
    elif Strial < -Fy:
        plastic = (Strial + Fy) / k
        S_new = -Fy
    else:
        S_new = Strial
    dlam = abs(plastic) / uy
    Lam_new = Lam + dlam
    Fy_new = Fy
    if beta > 0.0 and dlam > 0.0:
        Fy_new = Fy0 * np.exp(-beta * Lam_new)
        if Fy_new < FY_FLOOR * Fy0:
            Fy_new = FY_FLOOR * Fy0
        if S_new > Fy_new:
            S_new = Fy_new
        elif S_new < -Fy_new:
            S_new = -Fy_new
    return u_new, v_new, S_new, Fy_new, Lam_new


@njit(cache=True)
def on_cap(S, Fy):
    return abs(S) >= Fy * (1.0 - 1.0e-8)


@njit(cache=True)
def energy_proxy(u, v, S, Fy, k, theta):
    """Monotonic cap-barrier proxy. Returns 1 (committed) or 0.

    Valid derivation assumes the spring stays on the current cap until the
    saddle. Inward velocity leaves the cap immediately, so those states are
    scored only by the past-saddle outward rule below; otherwise 0.
    """
    uc = uc_of(Fy, k, theta)
    tol = 1.0e-8 * max(Fy, 1.0e-12)
    if S >= Fy - tol:
        if u >= uc and v >= 0.0:
            return 1
        if u < uc and v > 0.0:
            barrier = Fy * (uc - u) - 0.5 * theta * k * (uc * uc - u * u)
            if barrier < 0.0:
                barrier = 0.0
            if 0.5 * v * v >= barrier:
                return 1
        return 0
    if S <= -Fy + tol:
        if u <= -uc and v <= 0.0:
            return 1
        if u > -uc and v < 0.0:
            barrier = Fy * (uc + u) - 0.5 * theta * k * (uc * uc - u * u)
            if barrier < 0.0:
                barrier = 0.0
            if 0.5 * v * v >= barrier:
                return 1
        return 0
    return 0


@njit(cache=True)
def past_saddle(u, S, Fy, k, theta):
    uc = uc_of(Fy, k, theta)
    tol = 1.0e-8 * max(Fy, 1.0e-12)
    if S >= Fy - tol and u >= uc:
        return 1
    if S <= -Fy + tol and u <= -uc:
        return 1
    return 0


@njit(cache=True)
def zero_input_committed(u, v, S, Fy, Lam, k, c, theta, beta, Fy0, uy, h):
    uc = uc_of(Fy, k, theta)
    # Obvious runaway: on cap, beyond the current saddle, velocity outward or zero.
    tol = 1.0e-8 * max(Fy, 1.0e-12)
    if S >= Fy - tol and u >= 1.01 * uc and v >= 0.0:
        return 1
    if S <= -Fy + tol and u <= -1.01 * uc and v <= 0.0:
        return 1
    # Obviously inside the elastic well.
    if abs(u) < 0.25 * uy and abs(v) < 0.02 * uy * np.sqrt(k) and abs(S) < 0.5 * Fy:
        return 0
    n = int(H_CONT / DT)
    quiet = 0
    for _ in range(n):
        u, v, S, Fy, Lam = step(u, v, S, Fy, Lam, 0.0, DT, k, c, theta, beta, Fy0, uy)
        if abs(u) / h >= ESCAPE_DRIFT:
            return 1
        if abs(u) < 0.35 * uy and abs(v) < 0.03 * uy * np.sqrt(k) and abs(S) < 0.55 * Fy:
            quiet += 1
            if quiet * DT >= 1.5:
                return 0
        else:
            quiet = 0
    return 0


@njit(cache=True)
def score_motion(ag, scale, dt, h, k, c, theta, beta, Fy0, uy):
    """Return fixed summary for one scaled motion.

    out[0] n_states
    out[1] n_truth
    out[2] n_tan_neg
    out[3] n_tan_fp          tangent neg, truth safe
    out[4] n_tan_fn          tangent pos, truth committed
    out[5] n_d10
    out[6] n_d10_fp          drift>=10%, truth safe
    out[7] n_d10_fn          drift<10%, truth committed
    out[8] n_d05
    out[9] n_d05_fp
    out[10] n_d05_fn
    out[11] n_saddle
    out[12] n_saddle_fp
    out[13] n_saddle_fn
    out[14] n_energy
    out[15] n_energy_fp
    out[16] n_energy_fn
    out[17] n_sust
    out[18] n_sust_fp
    out[19] n_sust_fn
    out[20] t_truth
    out[21] t_d05
    out[22] t_d10
    out[23] t_tan
    out[24] t_sust
    out[25] t_saddle
    out[26] t_energy
    out[27] drift_at_truth
    out[28] fy_ratio_at_truth
    out[29] reentry   1 if truth goes 1 then later 0
    out[30] drift_c_original   uy/h/theta, constant
    """
    out = np.zeros(31)
    out[20:27] = 1.0e9
    n = ag.shape[0]
    u = 0.0
    v = 0.0
    S = 0.0
    Fy = Fy0
    Lam = 0.0
    next_sample = 0.0
    cap_run = 0.0
    seen_commit = 0
    reentry = 0
    t = 0.0
    for i in range(n):
        ag_i = scale * G * ag[i]
        u, v, S, Fy, Lam = step(u, v, S, Fy, Lam, ag_i, dt, k, c, theta, beta, Fy0, uy)
        t = (i + 1) * dt
        if abs(u) / h >= ESCAPE_DRIFT:
            break
        capped = 1 if on_cap(S, Fy) else 0
        if capped == 1:
            cap_run += dt
        else:
            cap_run = 0.0
        # Proxy first-passage is recorded every step. Zero-input truth stays
        # on the 0.10 s grid because each test is a continuation.
        drift_now = abs(u) / h
        if drift_now >= 0.05 and out[21] > 1.0e8:
            out[21] = t
        if drift_now >= 0.10 and out[22] > 1.0e8:
            out[22] = t
        if capped == 1 and out[23] > 1.0e8:
            out[23] = t
        if cap_run >= SUSTAIN_S and out[24] > 1.0e8:
            out[24] = t
        if past_saddle(u, S, Fy, k, theta) == 1 and out[25] > 1.0e8:
            out[25] = t
        if energy_proxy(u, v, S, Fy, k, theta) == 1 and out[26] > 1.0e8:
            out[26] = t
        if t + 1.0e-12 >= next_sample:
            next_sample += T_SAMPLE
            truth = zero_input_committed(u, v, S, Fy, Lam, k, c, theta, beta, Fy0, uy, h)
            drift = abs(u) / h
            tan = capped
            d10 = 1 if drift >= 0.10 else 0
            d05 = 1 if drift >= 0.05 else 0
            sad = past_saddle(u, S, Fy, k, theta)
            en = energy_proxy(u, v, S, Fy, k, theta)
            sust = 1 if cap_run >= SUSTAIN_S else 0
            out[0] += 1
            out[1] += truth
            out[2] += tan
            if tan == 1 and truth == 0:
                out[3] += 1
            if tan == 0 and truth == 1:
                out[4] += 1
            out[5] += d10
            if d10 == 1 and truth == 0:
                out[6] += 1
            if d10 == 0 and truth == 1:
                out[7] += 1
            out[8] += d05
            if d05 == 1 and truth == 0:
                out[9] += 1
            if d05 == 0 and truth == 1:
                out[10] += 1
            out[11] += sad
            if sad == 1 and truth == 0:
                out[12] += 1
            if sad == 0 and truth == 1:
                out[13] += 1
            out[14] += en
            if en == 1 and truth == 0:
                out[15] += 1
            if en == 0 and truth == 1:
                out[16] += 1
            out[17] += sust
            if sust == 1 and truth == 0:
                out[18] += 1
            if sust == 0 and truth == 1:
                out[19] += 1
            if truth == 1 and out[20] > 1.0e8:
                out[20] = t
                out[27] = drift
                out[28] = Fy / Fy0
            if truth == 1:
                seen_commit = 1
            elif seen_commit == 1:
                reentry = 1
    out[29] = reentry
    out[30] = UY_DRIFT / theta
    return out


@njit(cache=True)
def simulate_free(u, v, S, Fy, n, dt, k, c, theta, uy):
    beta = 0.0
    Fy0 = Fy
    Lam = 0.0
    peak = abs(u)
    for _ in range(n):
        u, v, S, Fy, Lam = step(u, v, S, Fy, Lam, 0.0, dt, k, c, theta, beta, Fy0, uy)
        if abs(u) > peak:
            peak = abs(u)
    return u, v, peak


def oscillator(T, theta):
    k = (2.0 * np.pi / T) ** 2
    c = 2.0 * ZETA * np.sqrt(k)
    uy = UY_DRIFT * H
    Fy0 = k * uy
    return k, c, uy, Fy0


def analytical_v_crit(u, Fy, k, theta):
    uc = Fy / (theta * k)
    barrier = Fy * (uc - u) - 0.5 * theta * k * (uc * uc - u * u)
    return float(np.sqrt(max(0.0, 2.0 * barrier))), float(uc)


def self_test():
    T = 1.0
    theta = 0.10
    k, c, uy, Fy0 = oscillator(T, theta)
    # Elastic period lengthens by 1/sqrt(1-theta).
    u, v, _ = simulate_free(1.0e-4, 0.0, k * 1.0e-4, Fy0, 20000, DT, k, c, theta, uy)
    # Count zero crossings of a dedicated elastic run.
    uu = 1.0e-3
    vv = 0.0
    SS = k * uu
    sign_changes = 0
    prev = uu
    t_cross = []
    for i in range(30000):
        uu, vv, SS, _, _ = step(uu, vv, SS, Fy0, 0.0, 0.0, DT, k, c, theta, 0.0, Fy0, uy)
        if prev > 0.0 and uu <= 0.0:
            sign_changes += 1
            t_cross.append((i + 1) * DT)
            if sign_changes == 3:
                break
        prev = uu
    # Positive-to-negative crossings are one per cycle.
    periods = np.diff(t_cross)
    T_eff = float(np.mean(periods))
    T_target = T / np.sqrt(1.0 - theta)
    period_err = abs(T_eff - T_target) / T_target

    uc = Fy0 / (theta * k)
    # Separatrix checks use c=0 so they match the undamped energy barrier.
    safe = zero_input_committed(0.55 * uc, 0.0, Fy0, Fy0, 0.0, k, 0.0, theta, 0.0, Fy0, uy, H)
    gone = zero_input_committed(1.15 * uc, 0.0, Fy0, Fy0, 0.0, k, 0.0, theta, 0.0, Fy0, uy, H)

    u_test = 0.72 * uc
    v_crit, _ = analytical_v_crit(u_test, Fy0, k, theta)
    below = zero_input_committed(u_test, 0.85 * v_crit, Fy0, Fy0, 0.0, k, 0.0, theta, 0.0, Fy0, uy, H)
    above = zero_input_committed(u_test, 1.15 * v_crit, Fy0, Fy0, 0.0, k, 0.0, theta, 0.0, Fy0, uy, H)
    report = {
        "period_s": T_eff,
        "period_target_s": T_target,
        "period_rel_err": period_err,
        "safe_below_saddle": int(safe),
        "escape_above_saddle": int(gone),
        "v_crit_m_s": v_crit,
        "recover_at_0.85_vcrit": int(below),
        "escape_at_1.15_vcrit": int(above),
        "uc_m": uc,
        "drift_c": uc / H,
    }
    ok = (
        period_err < 0.03
        and safe == 0
        and gone == 1
        and below == 0
        and above == 1
    )
    report["pass"] = bool(ok)
    return report


def gabor(rng, n, dt):
    fp = rng.uniform(0.5, 3.0)
    sig = rng.uniform(0.35, 1.4)
    t0 = 3.2 * sig
    pol = rng.choice(np.array([-1.0, 1.0]))
    t = np.arange(n) * dt
    ag = pol * np.exp(-((t - t0) / sig) ** 2) * np.cos(2.0 * np.pi * fp * (t - t0))
    pga = np.max(np.abs(ag))
    if pga < 1.0e-8:
        ag = np.zeros(n)
        ag[n // 5] = 1.0
        pga = 1.0
    return ag / pga, {"kind": "gabor", "fp_Hz": fp, "sigma_s": sig, "t0_s": t0, "polarity": pol}


def kanai_tajimi(rng, n, dt):
    fg = rng.uniform(1.0, 4.0)
    wg = 2.0 * np.pi * fg
    zg = 0.55
    x = 0.0
    xd = 0.0
    ag = np.zeros(n)
    td = rng.uniform(4.0, 10.0)
    for i in range(n):
        t = i * dt
        env = (t / td) * np.exp(1.0 - t / td) if t < 4.0 * td else 0.0
        w = rng.normal() * np.sqrt(dt)
        xdd = -2.0 * zg * wg * xd - wg * wg * x + env * w
        xd = xd + xdd * dt
        x = x + xd * dt
        ag[i] = -(2.0 * zg * wg * xd + wg * wg * x)
    pga = np.max(np.abs(ag))
    if pga < 1.0e-12:
        ag[n // 3] = 1.0
        pga = 1.0
    return ag / pga, {"kind": "kanai_tajimi", "fg_Hz": fg, "td_s": td}


def make_suite(n_gabor, n_kt, dur_gabor, dur_kt):
    rng = np.random.default_rng(SEED)
    motions = []
    meta = []
    nA = int(round(dur_gabor / DT))
    nB = int(round(dur_kt / DT))
    for i in range(n_gabor):
        ag, info = gabor(rng, nA, DT)
        motions.append(ag)
        info["id"] = f"gabor_{i:02d}"
        meta.append(info)
    for i in range(n_kt):
        ag, info = kanai_tajimi(rng, nB, DT)
        motions.append(ag)
        info["id"] = f"kt_{i:02d}"
        meta.append(info)
    return motions, meta


def first_scale(times_by_scale, scales):
    """times_by_scale: list of event times, one per scale. Return scale or None."""
    for s, t in zip(scales, times_by_scale):
        if t < 1.0e8:
            return float(s)
    return None


def run_cell(T, theta, beta, motions, meta, tag):
    k, c, uy, Fy0 = oscillator(T, theta)
    rows = []
    # column sums of the count block, plus motion-level arrays
    for im, ag in enumerate(motions):
        per_scale = []
        for scale in SCALES:
            rec = score_motion(np.ascontiguousarray(ag), float(scale), DT, H, k, c, theta, beta, Fy0, uy)
            per_scale.append(rec)
        stack = np.vstack(per_scale)
        counts = stack[:, 0:20].sum(axis=0)
        # first scale at which each event time is finite
        keys = ["truth", "d05", "d10", "tan", "sust", "saddle", "energy"]
        scales_hit = {}
        for j, key in enumerate(keys):
            scales_hit[key] = first_scale(stack[:, 20 + j], SCALES)
        # drift at the first committing scale
        drift_at = None
        fy_at = None
        reentry_at = None
        if scales_hit["truth"] is not None:
            idx = int(np.where(np.isclose(SCALES, scales_hit["truth"]))[0][0])
            drift_at = float(stack[idx, 27])
            fy_at = float(stack[idx, 28])
            reentry_at = int(stack[idx, 29])
        rows.append(
            {
                "id": meta[im]["id"],
                "kind": meta[im]["kind"],
                "tag": tag,
                "T_s": T,
                "theta": theta,
                "beta": beta,
                "counts": counts.tolist(),
                "scale": scales_hit,
                "drift_at_first_commitment": drift_at,
                "fy_ratio_at_first_commitment": fy_at,
                "reentry_at_commitment_scale": reentry_at,
                "drift_c": float(UY_DRIFT / theta),
            }
        )
        print(f"  {tag} {meta[im]['id']} truth_scale={scales_hit['truth']}", flush=True)
    return rows


def summarize(rows):
    def ratios(num_key, den_key):
        vals = []
        for r in rows:
            a = r["scale"][num_key]
            b = r["scale"][den_key]
            if a is not None and b is not None and b > 0:
                vals.append(a / b)
        return vals

    count_names = [
        "n_states",
        "n_truth",
        "n_tan",
        "n_tan_fp",
        "n_tan_fn",
        "n_d10",
        "n_d10_fp",
        "n_d10_fn",
        "n_d05",
        "n_d05_fp",
        "n_d05_fn",
        "n_saddle",
        "n_saddle_fp",
        "n_saddle_fn",
        "n_energy",
        "n_energy_fp",
        "n_energy_fn",
        "n_sust",
        "n_sust_fp",
        "n_sust_fn",
    ]
    tot = np.zeros(20)
    for r in rows:
        tot += np.array(r["counts"], dtype=float)

    def safe_div(a, b):
        return None if b == 0 else float(a / b)

    drifts = [r["drift_at_first_commitment"] for r in rows if r["drift_at_first_commitment"] is not None]
    fy = [r["fy_ratio_at_first_commitment"] for r in rows if r["fy_ratio_at_first_commitment"] is not None]
    re_den = [r for r in rows if r["scale"]["truth"] is not None]
    re_num = [r for r in re_den if r["reentry_at_commitment_scale"] == 1]
    n_truth_motions = sum(1 for r in rows if r["scale"]["truth"] is not None)
    n_d10_motions = sum(1 for r in rows if r["scale"]["d10"] is not None)

    def med_iqr(xs):
        if not xs:
            return None
        arr = np.array(xs, dtype=float)
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        }

    summary = {
        "n_motions": len(rows),
        "n_motions_with_commitment": n_truth_motions,
        "n_motions_reaching_10pct": n_d10_motions,
        "counts": {name: int(tot[i]) for i, name in enumerate(count_names)},
        "state_error_rates": {
            "tangent_fp_given_tan": safe_div(tot[3], tot[2]),
            "tangent_fn_given_truth": safe_div(tot[4], tot[1]),
            "d10_fp_given_d10": safe_div(tot[6], tot[5]),
            "d10_fn_given_truth": safe_div(tot[7], tot[1]),
            "d05_fp_given_d05": safe_div(tot[9], tot[8]),
            "d05_fn_given_truth": safe_div(tot[10], tot[1]),
            "saddle_fp_given_saddle": safe_div(tot[12], tot[11]),
            "saddle_fn_given_truth": safe_div(tot[13], tot[1]),
            "energy_fp_given_energy": safe_div(tot[15], tot[14]),
            "energy_fn_given_truth": safe_div(tot[16], tot[1]),
            "sust_fp_given_sust": safe_div(tot[18], tot[17]),
            "sust_fn_given_truth": safe_div(tot[19], tot[1]),
        },
        "scale_ratio_vs_commitment": {
            "d10": med_iqr(ratios("d10", "truth")),
            "d05": med_iqr(ratios("d05", "truth")),
            "tan": med_iqr(ratios("tan", "truth")),
            "sust": med_iqr(ratios("sust", "truth")),
            "saddle": med_iqr(ratios("saddle", "truth")),
            "energy": med_iqr(ratios("energy", "truth")),
        },
        "drift_at_first_commitment": med_iqr(drifts),
        "fy_ratio_at_first_commitment": med_iqr(fy),
        "reentry_fraction_among_committed": None
        if not re_den
        else {"n": len(re_num), "N": len(re_den), "p": len(re_num) / len(re_den)},
    }
    return summary


@njit(cache=True)
def phase_labels(u_vals, v_vals, k, c, theta, Fy0, uy, h):
    nu = u_vals.shape[0]
    nv = v_vals.shape[0]
    lab = np.zeros((nu, nv), dtype=np.int8)
    for i in range(nu):
        for j in range(nv):
            lab[i, j] = zero_input_committed(
                u_vals[i], v_vals[j], Fy0, Fy0, 0.0, k, c, theta, 0.0, Fy0, uy, h
            )
    return lab


@njit(cache=True)
def trace_motion(ag, scale, dt, sample_dt, h, k, c, theta, beta, Fy0, uy):
    maxn = 5000
    t_s = np.zeros(maxn)
    uu = np.zeros(maxn)
    vv = np.zeros(maxn)
    drift = np.zeros(maxn)
    truth = np.zeros(maxn)
    tan = np.zeros(maxn)
    en = np.zeros(maxn)
    sad = np.zeros(maxn)
    fyr = np.zeros(maxn)
    u = 0.0
    v = 0.0
    S = 0.0
    Fy = Fy0
    Lam = 0.0
    next_sample = 0.0
    nkeep = 0
    n = ag.shape[0]
    for i in range(n):
        ag_i = scale * G * ag[i]
        u, v, S, Fy, Lam = step(u, v, S, Fy, Lam, ag_i, dt, k, c, theta, beta, Fy0, uy)
        t = (i + 1) * dt
        if abs(u) / h >= ESCAPE_DRIFT:
            break
        if t + 1.0e-12 >= next_sample and nkeep < maxn:
            t_s[nkeep] = t
            uu[nkeep] = u
            vv[nkeep] = v
            drift[nkeep] = u / h
            truth[nkeep] = zero_input_committed(u, v, S, Fy, Lam, k, c, theta, beta, Fy0, uy, h)
            tan[nkeep] = 1.0 if on_cap(S, Fy) else 0.0
            en[nkeep] = energy_proxy(u, v, S, Fy, k, theta)
            sad[nkeep] = past_saddle(u, S, Fy, k, theta)
            fyr[nkeep] = Fy / Fy0
            nkeep += 1
            next_sample += sample_dt
    return nkeep, t_s, uu, vv, drift, truth, tan, en, sad, fyr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("self-test", flush=True)
    gate = self_test()
    print(json.dumps(gate, indent=2), flush=True)
    (OUT / "self_test.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["pass"]:
        raise SystemExit("self-test failed; census not run")

    motions, meta = make_suite(20, 20, 12.0, 25.0)
    cells = [
        ("main_b0", 1.0, 0.10, 0.0, motions, meta),
        ("main_b004", 1.0, 0.10, 0.04, motions, meta),
    ]
    # Smaller sensitivity suite, frozen seed offset by rebuilding with fewer motions
    # from the same generator call would change the suite. Slice the first 8 of each kind.
    sens_motions = [m for m, info in zip(motions, meta) if info["id"].endswith(("00", "01", "02", "03", "04", "05", "06", "07"))]
    sens_meta = [info for info in meta if info["id"].endswith(("00", "01", "02", "03", "04", "05", "06", "07"))]
    for theta in (0.06, 0.14):
        cells.append((f"th{theta:.2f}_b0", 1.0, theta, 0.0, sens_motions, sens_meta))
        cells.append((f"th{theta:.2f}_b004", 1.0, theta, 0.04, sens_motions, sens_meta))
    for T in (0.5, 2.0):
        cells.append((f"T{T:.1f}_b0", T, 0.10, 0.0, sens_motions, sens_meta))
        cells.append((f"T{T:.1f}_b004", T, 0.10, 0.04, sens_motions, sens_meta))

    all_rows = []
    summaries = {}
    for tag, T, theta, beta, mots, metas in cells:
        print(f"CELL {tag} n={len(mots)}", flush=True)
        rows = run_cell(T, theta, beta, mots, metas, tag)
        all_rows.extend(rows)
        summaries[tag] = summarize(rows)
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        (OUT / "rows.json").write_text(json.dumps(all_rows), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
