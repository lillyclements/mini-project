"""
reliability_extensions.py
=========================
Extensions to the ASTE 404 Spacecraft Reliability Monte Carlo tool:

  1. Vectorized Monte Carlo (fast, chunked, reproducible via a seeded RNG)
  2. Statistically sound convergence study (many repeats per N, RMS error,
     fitted log-log slope, comparison to theoretical binomial std error)
  3. Common-cause failure model (beta-factor) for redundant subsystems,
     with an analytical check and a redundancy-vs-beta study

Self-contained: needs only numpy and matplotlib.

Usage:
    python reliability_extensions.py --run convergence
    python reliability_extensions.py --run commoncause
    python reliability_extensions.py --run all
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FAILURES = [0.01, 0.02, 0.015, 0.01, 0.025]


# ---------------------------------------------------------------------------
# Analytical solutions (validation oracles)
# ---------------------------------------------------------------------------
def analytical_reliability(failure_probs, redundancy=None):
    """Series system of independent subsystems; subsystem i has k_i active
    parallel units. R = prod(1 - p_i^k_i)."""
    p = np.asarray(failure_probs, dtype=float)
    k = np.ones(len(p), dtype=int) if redundancy is None else np.asarray(redundancy)
    return float(np.prod(1.0 - p ** k))


def group_failure_prob_beta(p, k, beta):
    """Failure probability of one k-unit redundant group under the beta-factor
    common-cause model.

    Each unit's failure probability p is split into
        independent part  (1 - beta) * p   (units fail independently)
        common-cause part  beta * p        (one event takes out ALL k units)
    The group fails if the common-cause event happens OR all k units fail
    independently. For k = 1 there is nothing to be common with, so the
    group failure probability is simply p.
    """
    if k == 1:
        return p
    return 1.0 - (1.0 - beta * p) * (1.0 - ((1.0 - beta) * p) ** k)


def analytical_reliability_beta(failure_probs, redundancy, beta):
    """Exact reliability of the series system under the beta-factor model."""
    r = 1.0
    for p, k in zip(failure_probs, redundancy):
        r *= 1.0 - group_failure_prob_beta(p, k, beta)
    return r


# ---------------------------------------------------------------------------
# Vectorized Monte Carlo
# ---------------------------------------------------------------------------
def mc_reliability(failure_probs, n_samples, redundancy=None, beta=0.0,
                   rng=None, chunk=200_000):
    """Vectorized MC estimate of mission reliability.

    beta = 0 reproduces the independent model. beta > 0 adds common-cause
    failures within each redundant group (k > 1).
    Samples are processed in chunks to bound memory use.
    """
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(failure_probs, dtype=float)
    k = np.ones(len(p), dtype=int) if redundancy is None else np.asarray(redundancy)

    successes = 0
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        mission_ok = np.ones(m, dtype=bool)
        for p_i, k_i in zip(p, k):
            if k_i == 1:
                group_failed = rng.random(m) < p_i
            else:
                independent_all_fail = (rng.random((m, k_i)) < (1 - beta) * p_i).all(axis=1)
                common_cause = rng.random(m) < beta * p_i
                group_failed = independent_all_fail | common_cause
            mission_ok &= ~group_failed
        successes += int(mission_ok.sum())
        done += m
    return successes / n_samples


# ---------------------------------------------------------------------------
# 1. Improved convergence study
# ---------------------------------------------------------------------------
def convergence_study(failure_probs, n_values, n_repeats=200, seed=42):
    """For each N, run n_repeats independent MC estimates and report the
    RMS error vs the analytical value.

    A single run's error is itself random, so one run per N gives a noisy,
    misleading slope. The RMS over many repeats estimates the true standard
    error, which should follow sqrt(R(1-R)/N).
    """
    rng = np.random.default_rng(seed)
    exact = analytical_reliability(failure_probs)

    rms_errors, mean_estimates = [], []
    for n in n_values:
        est = np.array([mc_reliability(failure_probs, n, rng=rng)
                        for _ in range(n_repeats)])
        rms_errors.append(np.sqrt(np.mean((est - exact) ** 2)))
        mean_estimates.append(est.mean())
        print(f"  N = {n:>9,}: RMS error = {rms_errors[-1]:.6f}   "
              f"theory = {np.sqrt(exact * (1 - exact) / n):.6f}")

    rms_errors = np.array(rms_errors)
    theory = np.sqrt(exact * (1 - exact) / np.asarray(n_values))

    # Fit log(err) = slope * log(N) + intercept
    slope, intercept = np.polyfit(np.log(n_values), np.log(rms_errors), 1)

    # Uncertainty on the slope from the fit itself (standard error)
    x = np.log(n_values)
    resid = np.log(rms_errors) - (slope * x + intercept)
    dof = len(x) - 2
    slope_se = np.sqrt(np.sum(resid ** 2) / dof / np.sum((x - x.mean()) ** 2)) if dof > 0 else np.nan

    return {
        "exact": exact, "n_values": np.asarray(n_values), "rms": rms_errors,
        "theory": theory, "slope": slope, "slope_se": slope_se,
        "n_repeats": n_repeats,
    }


def plot_convergence(res, save_path="convergence_rms.png"):
    n, rms, theory = res["n_values"], res["rms"], res["theory"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n, rms, "o-", lw=2, ms=7,
              label=f"Measured RMS error ({res['n_repeats']} runs per N)")
    ax.loglog(n, theory, "r--", lw=2, label=r"Theory: $\sqrt{R(1-R)/N}$")
    ax.set_xlabel("Number of samples, N", fontsize=13)
    ax.set_ylabel("RMS error vs analytical reliability", fontsize=13)
    ax.set_title(f"Monte Carlo error scales as N$^{{{res['slope']:.2f}}}$ "
                 f"(theory: N$^{{-0.50}}$)", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"  saved {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Common-cause failure study
# ---------------------------------------------------------------------------
def common_cause_study(failure_probs, betas, n_samples=400_000, seed=7):
    """For each subsystem, reliability gain from adding ONE backup
    (k: 1 -> 2) as a function of beta. Also tracks the 'all subsystems
    duplicated' system. Returns MC and analytical values."""
    rng = np.random.default_rng(seed)
    n_sub = len(failure_probs)
    base = [1] * n_sub
    base_exact = analytical_reliability(failure_probs)

    gains_mc = np.zeros((len(betas), n_sub))
    gains_exact = np.zeros((len(betas), n_sub))
    full_mc, full_exact = [], []

    for bi, beta in enumerate(betas):
        base_mc = mc_reliability(failure_probs, n_samples, base, beta, rng)
        for i in range(n_sub):
            cfg = base.copy()
            cfg[i] = 2
            gains_mc[bi, i] = mc_reliability(failure_probs, n_samples, cfg, beta, rng) - base_mc
            gains_exact[bi, i] = analytical_reliability_beta(failure_probs, cfg, beta) - base_exact
        full = [2] * n_sub
        full_mc.append(mc_reliability(failure_probs, n_samples, full, beta, rng))
        full_exact.append(analytical_reliability_beta(failure_probs, full, beta))

    return {"betas": np.asarray(betas), "gains_mc": gains_mc,
            "gains_exact": gains_exact, "full_mc": np.array(full_mc),
            "full_exact": np.array(full_exact), "failure_probs": failure_probs,
            "base_exact": base_exact}


def plot_common_cause(res, save_path="common_cause_study.png"):
    betas = res["betas"]
    fp = res["failure_probs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, p in enumerate(fp):
        ax1.plot(betas, res["gains_exact"][:, i] * 100, "-", lw=2,
                 label=f"Sub {i+1} (p={p:.3f})")
        ax1.plot(betas, res["gains_mc"][:, i] * 100, "k.", ms=4, alpha=0.5)
    ax1.set_xlabel(r"Common-cause fraction $\beta$", fontsize=13)
    ax1.set_ylabel("Reliability gain from one backup (pct. points)", fontsize=12)
    ax1.set_title("Common-cause failures erode the value of redundancy",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(betas, res["full_exact"] * 100, "r-", lw=2, label="Analytical")
    ax2.plot(betas, res["full_mc"] * 100, "bo", ms=5, label="Monte Carlo")
    ax2.axhline(res["base_exact"] * 100, color="gray", ls="--",
                label=f"No redundancy ({res['base_exact']*100:.1f}%)")
    ax2.set_xlabel(r"Common-cause fraction $\beta$", fontsize=13)
    ax2.set_ylabel("Mission reliability, all subsystems duplicated (%)", fontsize=12)
    ax2.set_title("Full duplication: diminishing returns", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"  saved {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------
def verify_common_cause(failure_probs=None, n_samples=1_000_000, seed=3):
    """Sanity checks: beta=0 matches independent model; beta=1 makes a backup
    worthless; MC matches analytical within 4 standard errors."""
    fp = DEFAULT_FAILURES if failure_probs is None else failure_probs
    rng = np.random.default_rng(seed)
    cfg = [1, 1, 1, 1, 2][:len(fp)] if len(fp) == 5 else [2] + [1] * (len(fp) - 1)
    ok = True
    print("\nVerification of common-cause model")

    for beta in (0.0, 0.1, 0.3, 1.0):
        exact = analytical_reliability_beta(fp, cfg, beta)
        est = mc_reliability(fp, n_samples, cfg, beta, rng)
        se = np.sqrt(exact * (1 - exact) / n_samples)
        passed = abs(est - exact) < 4 * se + 1e-12
        ok &= passed
        print(f"  beta={beta:.1f}: MC={est:.6f} analytical={exact:.6f} "
              f"|err|/SE={abs(est-exact)/se if se else 0:.2f} "
              f"{'PASS' if passed else 'FAIL'}")

    no_backup = analytical_reliability(fp)
    useless = np.isclose(analytical_reliability_beta(fp, cfg, 1.0), no_backup)
    ok &= useless
    print(f"  beta=1 backup equals no backup: {'PASS' if useless else 'FAIL'}")
    print(f"  ALL PASSED: {ok}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reliability tool extensions")
    ap.add_argument("--run", choices=["convergence", "commoncause", "all"], default="all")
    ap.add_argument("--failures", type=float, nargs="+", default=DEFAULT_FAILURES)
    ap.add_argument("--repeats", type=int, default=200,
                    help="independent MC runs per N in the convergence study")
    args = ap.parse_args()

    if args.run in ("convergence", "all"):
        print("\nCONVERGENCE STUDY (RMS error over repeated runs)")
        n_values = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
        t0 = time.time()
        res = convergence_study(args.failures, n_values, n_repeats=args.repeats)
        print(f"  fitted slope = {res['slope']:.3f} +/- {res['slope_se']:.3f} "
              f"(theory -0.500)   [{time.time()-t0:.1f}s]")
        plot_convergence(res)

    if args.run in ("commoncause", "all"):
        print("\nCOMMON-CAUSE STUDY (beta-factor model)")
        verify_common_cause(args.failures)
        betas = np.linspace(0, 0.5, 11)
        res = common_cause_study(args.failures, betas)
        i5 = int(np.argmax(args.failures))
        for b in (0.0, 0.1, 0.2):
            j = int(np.argmin(np.abs(betas - b)))
            print(f"  beta={betas[j]:.2f}: backup on sub {i5+1} gains "
                  f"{res['gains_exact'][j, i5]*100:.2f} pct pts; "
                  f"full duplication R = {res['full_exact'][j]*100:.2f}%")
        plot_common_cause(res)


if __name__ == "__main__":
    main()
