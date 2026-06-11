"""
manuscript_code.py
===================
Reference implementation accompanying the manuscript
"Resistance-aware precision dosing as posterior inference".

SCOPE AND HONESTY NOTE
----------------------
This is a runnable *scaffold*, not a finished pipeline. It implements the
closed-form machinery of Sections 3-4 and the Monte-Carlo cross-check, plus a
NumPyro model skeleton for Section 2. It DOES NOT contain real data and DOES
NOT produce the manuscript's (to-be-computed) results. Functions marked
`TODO` are where you must plug in real susceptibility/fitness data and a fitted
zoliflodacin PK model. No numbers in this file should be reported as findings.

Dependencies: numpy, scipy; (optional) numpyro + jax for the Bayesian fit.
"""

from __future__ import annotations
import itertools
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# =============================================================================
# 1. GENOTYPE ENCODING (Section 2.1)
# =============================================================================
# Each tracked substitution is one binary slot. We keep the *codon* of each
# slot so we can (a) enforce mutual exclusivity within a codon and (b) attach
# C-alpha distances for the structural prior on epistasis (Section 2.4).
SUBS  = ["gyrA_S91F", "gyrA_A92T", "parC_D86N",
         "gyrB_D429N", "gyrB_K450T", "gyrB_S467N"]      # extend as needed
CODON = ["gyrA91", "gyrA92", "parC86", "gyrB429", "gyrB450", "gyrB467"]
P = len(SUBS)


def encode(present_subs) -> np.ndarray:
    """Binary feature vector for an allele given the substitutions it carries."""
    return np.array([1.0 if s in present_subs else 0.0 for s in SUBS])


def hamming1_neighbours(x: np.ndarray) -> list[np.ndarray]:
    """Single-substitution neighbours on the Hamming graph.

    We only toggle a slot ON if no other slot at the same codon is already ON
    (mutual exclusivity), and we allow toggling any ON slot OFF (reversion).
    This defines the mutational moves used to build the MPC maximum.
    """
    neigh = []
    for j in range(P):
        if x[j] == 0:
            # forbid a second substitution at an already-mutated codon
            same_codon_on = any(x[k] == 1 and CODON[k] == CODON[j] for k in range(P))
            if same_codon_on:
                continue
        xn = x.copy()
        xn[j] = 1.0 - xn[j]
        neigh.append(xn)
    return neigh


# =============================================================================
# 2. INTERACTION DESIGN MATRIX (Section 2.3)
# =============================================================================
PAIRS = list(itertools.combinations(range(P), 2))


def design(X: np.ndarray) -> np.ndarray:
    """Main effects + pairwise interaction columns for a (n, P) genotype matrix."""
    X = np.atleast_2d(X)
    inter = np.stack([X[:, j] * X[:, k] for (j, k) in PAIRS], axis=1)
    return np.hstack([X, inter])                            # (n, P + len(PAIRS))


def pair_prior_scale(dist_ang: dict, contact=8.0, tight=0.05, loose=0.5):
    """Structure-informed prior SD on epistasis terms: large only for residue
    pairs in spatial contact (Section 2.4). `dist_ang[(j,k)]` in angstroms."""
    return np.array([loose if dist_ang.get((j, k), np.inf) < contact else tight
                     for (j, k) in PAIRS])


# =============================================================================
# 3. BAYESIAN FIT SKELETON (Section 2; optional NumPyro)
# =============================================================================
def numpyro_model(Xdesign, drug_idx, mic_lo, mic_hi,
                  fit_X, fit_obs, n_drug, pair_sd_main=0.3):
    """Censored-MIC + fitness model with sparse epistasis and cross-drug pooling.

    Cross-drug sharing is shown here as a simple hierarchical normal; swap in a
    low-rank U V^T (Section 2.5) to encode shared-pocket structure explicitly.
    """
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    n_feat = Xdesign.shape[1]
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_feat)))
    mu_beta = numpyro.sample("mu_beta", dist.Normal(0.0, tau * lam))
    with numpyro.plate("drugs", n_drug):
        beta = numpyro.sample("beta", dist.Normal(mu_beta, pair_sd_main))
    beta0 = numpyro.sample("beta0", dist.Normal(jnp.zeros(n_drug), 2.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(jnp.ones(n_drug)))

    yhat = beta0[drug_idx] + jnp.sum(Xdesign * beta[drug_idx], axis=1)
    # interval-censored likelihood on log2 scale: P(lo < y <= hi)
    cdf_hi = dist.Normal(yhat, sigma[drug_idx]).cdf(mic_hi)
    cdf_lo = dist.Normal(yhat, sigma[drug_idx]).cdf(mic_lo)
    numpyro.factor("mic", jnp.sum(jnp.log(cdf_hi - cdf_lo + 1e-12)))

    gamma = numpyro.sample("gamma", dist.Normal(jnp.zeros(fit_X.shape[1]), 0.5))
    s_fit = numpyro.sample("s_fit", dist.HalfNormal(0.5))
    numpyro.sample("fit", dist.StudentT(4.0, fit_X @ gamma, s_fit), obs=fit_obs)


# =============================================================================
# 4. PHARMACODYNAMICS: lab-MIC -> in-host threshold (Section 3)
# =============================================================================
def lambda_factor(w, psi_max, delta_vitro, delta_host, Emax, h):
    """Fitness-and-environment correction Lambda_d(g') so that C*(g') = MIC * Lambda.
    Returns 1.0 when in-host == in-vitro. Mutants non-viable in host (psi_host<=0)
    are flagged with NaN so they can be screened out of the MPC maximum."""
    psi_vitro = psi_max * w - delta_vitro
    psi_host  = psi_max * w - delta_host
    if psi_host <= 0:                       # cannot grow in host -> not a threat
        return np.nan
    if not (0 < psi_vitro < Emax and 0 < psi_host < Emax):
        return np.nan
    num = (Emax - psi_vitro) * psi_host
    den = psi_vitro * (Emax - psi_host)
    return (num / den) ** (1.0 / h)


# =============================================================================
# 5. CLOSED-FORM MPC / MSW POSTERIOR (Section 4.2)
# =============================================================================
def neighbour_logthreshold_moments(mu_beta, Sigma_beta, X_nb, logLambda_nb):
    """z_g' = log2 C*(g') is LINEAR in beta, so the neighbour vector is jointly
    Gaussian in closed form. logLambda_nb is 0 when Lambda==1."""
    mu_z = X_nb @ mu_beta + logLambda_nb
    Sig_z = X_nb @ Sigma_beta @ X_nb.T
    return mu_z, Sig_z


def clark_max_moments(mu, Sig):
    """Clark (1961) pairwise recursion for E[max] and Var[max] of correlated
    normals, folding in one neighbour at a time. Also returns cov(max, each z_j),
    needed for Var(W) when g0 is among the components."""
    mu = np.asarray(mu, float); Sig = np.asarray(Sig, float)
    m, v = mu[0], Sig[0, 0]
    cov_row = Sig[0].copy()
    for j in range(1, len(mu)):
        m2, v2 = mu[j], Sig[j, j]
        theta = np.sqrt(max(v + v2 - 2 * cov_row[j], 1e-12))
        a = (m - m2) / theta
        Phi, phi = norm.cdf(a), norm.pdf(a)
        m_new = m * Phi + m2 * (1 - Phi) + theta * phi
        e2 = (m**2 + v) * Phi + (m2**2 + v2) * (1 - Phi) + (m + m2) * theta * phi
        v = max(e2 - m_new**2, 1e-12)
        cov_row = Phi * cov_row + (1 - Phi) * Sig[j]
        m = m_new
    return m, v, cov_row


def mpc_msw_posterior(mu_beta, Sigma_beta, X_nb, x_g0,
                      logLambda_nb, logLambda_g0, g0_index_in_nb=None):
    """Closed-form posterior moments for log2 MPC and the window width W.

    Cov(max_z, z_g0) is computed analytically:
      alpha = lstsq(Sig_z, cov_max)  =>  Cov(max_z, z_g0) = alpha @ (X_nb @ Sigma_beta @ x_g0)
    This is correct even when g0 is not among the neighbours, because max_z and
    z_g0 share uncertainty in beta via their joint covariance through Sigma_beta.
    The previous approach (cov_max[g0_index_in_nb]) was an approximation valid
    only when g0 appeared as a neighbour entry; it has been replaced here.
    """
    mu_z, Sig_z = neighbour_logthreshold_moments(mu_beta, Sigma_beta, X_nb, logLambda_nb)
    m_mpc, v_mpc, cov_max = clark_max_moments(mu_z, Sig_z)
    mu_g0  = float(x_g0 @ mu_beta + logLambda_g0)
    var_g0 = float(x_g0 @ Sigma_beta @ x_g0)
    cov_z_nb_g0 = X_nb @ Sigma_beta @ x_g0
    alpha = np.linalg.lstsq(Sig_z, cov_max, rcond=None)[0]
    cov_mg0 = float(alpha @ cov_z_nb_g0)
    return dict(E_log2MPC=m_mpc, Var_log2MPC=v_mpc,
                E_W=m_mpc - mu_g0,
                Var_W=max(v_mpc + var_g0 - 2 * cov_mg0, 1e-12))


def mpc_msw_montecarlo(beta_draws, X_nb, x_g0, logLambda_nb, logLambda_g0):
    """Exact pushforward cross-check: run the deterministic map over posterior
    draws. Returns empirical posteriors of log2 MPC and W (exact up to MC error).
    Use this to validate the Clark closed form (Section 6.2)."""
    Z = beta_draws @ X_nb.T + logLambda_nb              # (n_draws, n_nb)
    z0 = beta_draws @ x_g0 + logLambda_g0               # (n_draws,)
    log2_mpc = Z.max(axis=1)
    return log2_mpc, log2_mpc - z0


# =============================================================================
# 6. PK COUPLING: time in the selection window (Section 4.3)
# =============================================================================
def Tmsw_iv(W, t_half):
    """IV-bolus limit, peak clears MPC: T_MSW = t_half * W (W in doublings)."""
    return t_half * W


def Tmsw_oral(mic, mpc, F, D, Vd, ka, ke):
    """Oral single-dose (Bateman) time inside [MIC, MPC]. Solves threshold
    crossings numerically because the IV cancellation does NOT hold once an
    absorption phase is present (manuscript Section 4.3 correction).

    Returns time-in-window; 0 if the peak never reaches MIC.
    """
    A = F * D * ka / (Vd * (ka - ke))

    def C(t):
        return A * (np.exp(-ke * t) - np.exp(-ka * t))

    tmax = np.log(ka / ke) / (ka - ke)          # time of peak concentration
    Cmax = C(tmax)
    if Cmax < mic:
        return 0.0

    def crossings(theta):
        if theta >= Cmax:
            return []                            # never reached
        up = brentq(lambda t: C(t) - theta, 1e-9, tmax)        # ascending limb
        # descending limb: search beyond tmax out to a long horizon
        hi = tmax
        while C(hi) > theta and hi < tmax + 50 / ke:
            hi *= 2 if hi > 0 else 1
            hi += 1.0 / ke
        down = brentq(lambda t: C(t) - theta, tmax, hi)
        return [up, down]

    mic_cr = crossings(mic)                       # [enter-above-MIC, fall-below-MIC]
    mpc_cr = crossings(mpc)                        # may be [] if peak < MPC
    if not mic_cr:
        return 0.0
    t_in_mic = mic_cr[1] - mic_cr[0]               # total time above MIC
    if not mpc_cr:                                 # never exceeds MPC: whole MIC-time in window
        return t_in_mic
    t_in_mpc = mpc_cr[1] - mpc_cr[0]               # time above MPC (outside window)
    return t_in_mic - t_in_mpc


def Tmsw_posterior_iv(E_W, Var_W, E_thalf, Var_thalf):
    """Posterior mean/var of T_MSW = t_half * W (independent factors)."""
    E_T = E_thalf * E_W
    Var_T = (E_thalf**2) * Var_W + (E_W**2) * Var_thalf + Var_thalf * Var_W
    return E_T, Var_T


def prob_selection_time_below(tau, E_T, Var_T):
    """Decision-grade posterior probability P(T_MSW < tau | data)."""
    return float(norm.cdf((tau - E_T) / np.sqrt(Var_T)))


# =============================================================================
# 7. STOCHASTIC SOFT-MPC (Section 4.4) -- constant-environment placeholder
# =============================================================================
def p_establish_constant(r_net, psi_max, w):
    """Birth-death establishment probability in a CONSTANT environment.
    TODO: replace with the time-varying-environment integral over C(t) before
    using as a reported result (manuscript Section 4.4 caveat)."""
    return max(0.0, r_net / (psi_max * w))


# =============================================================================
# 8. ENTRY POINTS TO COMPLETE WITH REAL INPUTS
# =============================================================================
def TODO_load_genotype_mic_fitness():
    """Assemble real genotype/MIC/fitness records (PathogenWatch + isogenic
    panels). Must return censored MIC intervals on the log2 scale per drug and
    fitness observations. See manuscript Section 5.1."""
    raise NotImplementedError("Plug in real data; do not fabricate.")


def TODO_load_zoliflodacin_pk():
    """Return a posterior (or point + uncertainty) over oral PK parameters
    F, D, Vd, ka, ke / t_half for zoliflodacin from a fitted/literature model."""
    raise NotImplementedError("Plug in a real PK model.")


if __name__ == "__main__":
    print("Scaffold loaded. No results are produced without real data "
          "(see TODO_* functions and manuscript Sections 5-6).")
