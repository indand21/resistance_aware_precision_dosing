"""
model_core.py
=============
Shared modelling machinery for the manuscript
"Resistance-aware precision dosing as posterior inference".

This module centralises every numerical routine so that the empirical analysis
(section5_analysis.py), the validation suite (section6_validation.py), and the
robustness check (section64_robustness.py) all exercise *the same* code. In
particular, the parameter-recovery validation (Section 6.1) calibrates the very
estimator that is deployed on real data (Section 5), rather than a proxy.

Contents
--------
Genotype->phenotype fitting
  censored_ridge_posterior  interval-censored (doubling-dilution) MAP + Laplace
                            posterior, with residual-SD uncertainty propagated
                            into the coefficient covariance.
  analytic_ridge_posterior  Gaussian-likelihood ridge posterior (continuous MIC);
                            retained for comparison and as a fast initialiser.
  design_diagnostics        condition number, VIF, and effective sample size for a
                            design matrix (collinearity / range-restriction audit).

MPC / mutant-selection window
  clark_max_moments         Clark (1961) moments of the maximum of correlated normals.
  mpc_msw_posterior         closed-form posterior moments of log2 MPC and window W.

Pharmacokinetics
  bateman_conc              oral first-order-absorption concentration profile.
  Tmsw_iv, Tmsw_oral        time inside the selection window (IV bolus / oral).

Stochastic refinement (soft MPC)
  net_growth_rate           Regoes/Hill per-capita net growth rate.
  establishment_probability time-varying-environment establishment probability of a
                            resistant lineage along a concentration profile C(t).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.integrate import quad

# =============================================================================
# Genotype -> phenotype fitting
# =============================================================================

def analytic_ridge_posterior(X, y, lambda_ridge=0.5, prior_sd_intercept=5.0):
    """Analytic ridge posterior beta | X,y ~ N(mu_post, Sigma_post).

    Gaussian likelihood on continuous log2 MIC. The intercept carries a weak
    prior; all other coefficients carry precision ``lambda_ridge``. Residual
    variance is estimated by a short fixed-point iteration. Retained for
    comparison with the censored fit and as a warm start.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    prior_prec = np.ones(p) * lambda_ridge
    prior_prec[0] = 1.0 / prior_sd_intercept**2
    sigma2 = np.var(y - np.mean(y)) + 1e-6
    for _ in range(5):
        A = X.T @ X + np.diag(prior_prec * sigma2)
        mu_post = np.linalg.solve(A, X.T @ y)
        resid = y - X @ mu_post
        sigma2 = max(float(resid @ resid) / n, 1e-6)
    A = X.T @ X + np.diag(prior_prec * sigma2)
    Sigma_post = sigma2 * np.linalg.inv(A)
    return mu_post, Sigma_post, sigma2


def _nll_censored(params, X, k_obs, half_width, prior_prec):
    """Negative log-posterior for the interval-censored model (beta, log_sigma)."""
    p = X.shape[1]
    beta = params[:p]
    sigma = np.exp(params[p])
    eta = X @ beta
    hi = (k_obs + half_width - eta) / sigma
    lo = (k_obs - half_width - eta) / sigma
    P = np.clip(norm.cdf(hi) - norm.cdf(lo), 1e-12, 1.0)
    nll = -np.sum(np.log(P))
    penalty = 0.5 * np.sum(prior_prec * beta**2)          # Gaussian prior on coeffs
    return nll + penalty


def _numerical_hessian(f, x, eps=1e-4):
    """Symmetric finite-difference Hessian of scalar f at x."""
    x = np.asarray(x, float); n = x.size
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy(); xj = x.copy(); xij = x.copy()
            xi[i] += eps; xj[j] += eps; xij[i] += eps; xij[j] += eps
            H[i, j] = H[j, i] = (f(xij) - f(xi) - f(xj) + fx) / eps**2
    return H


def censored_ridge_posterior(X, y_log2, lambda_ridge=0.5, prior_sd_intercept=5.0,
                             half_width=0.5, round_to_dilution=True, prior_prec=None):
    """Interval-censored (doubling-dilution) ridge posterior via MAP + Laplace.

    A reported log2 MIC is interval-censored on the doubling-dilution scale: a
    reading at step ``k`` implies the latent log2 MIC lies in
    ``(k - half_width, k + half_width]``. The likelihood contribution of one
    observation is the Gaussian interval probability
        Phi((k + hw - X beta)/sigma) - Phi((k - hw - X beta)/sigma),
    which is exactly the measurement model stated in manuscript Section 2.2.

    The residual SD ``sigma`` is a free parameter; the posterior covariance is
    the Laplace (inverse-Hessian) approximation over ``(beta, log sigma)``, so
    the returned coefficient covariance already propagates residual-variance
    uncertainty (manuscript Section 6.1 / reviewer point on sigma^2).

    Parameters
    ----------
    X : (n, p) design matrix (first column assumed intercept).
    y_log2 : (n,) reported log2 MIC values (continuous; rounded to the nearest
        dilution step internally when ``round_to_dilution`` is True).
    lambda_ridge : prior precision on non-intercept coefficients.
    prior_sd_intercept : prior SD on the intercept.
    half_width : half the dilution-step width on the log2 scale (0.5 = one doubling).
    prior_prec : optional (p,) array of per-coefficient prior precisions; when given
        it overrides ``lambda_ridge``/``prior_sd_intercept`` (used by the Section 6.1
        calibration so that validation and deployment share one estimator).

    Returns
    -------
    mu_post : (p,) MAP coefficient estimate.
    Sigma_post : (p, p) Laplace posterior covariance (sigma-uncertainty included).
    sigma : MAP residual SD (log2 doublings).
    """
    X = np.asarray(X, float)
    y = np.asarray(y_log2, float)
    n, p = X.shape
    k_obs = np.round(y) if round_to_dilution else y

    if prior_prec is None:
        prior_prec = np.ones(p) * lambda_ridge
        prior_prec[0] = 1.0 / prior_sd_intercept**2
    else:
        prior_prec = np.asarray(prior_prec, float)

    # Warm start from the Gaussian-likelihood ridge on the dilution midpoints.
    mu0, _, s20 = analytic_ridge_posterior(X, k_obs, lambda_ridge, prior_sd_intercept)
    init = np.concatenate([mu0, [0.5 * np.log(max(s20, 1e-3))]])

    obj = lambda th: _nll_censored(th, X, k_obs, half_width, prior_prec)
    res = minimize(obj, init, method="L-BFGS-B",
                   options=dict(maxiter=2000, ftol=1e-10))
    theta_hat = res.x
    mu_post = theta_hat[:p]
    sigma = float(np.exp(theta_hat[p]))

    # Laplace covariance over (beta, log_sigma); return the beta block.
    H = _numerical_hessian(obj, theta_hat)
    try:
        cov_full = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_full = np.linalg.pinv(H)
    Sigma_post = cov_full[:p, :p]
    # Symmetrise and guard against tiny negative eigenvalues from finite differences.
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
    w, V = np.linalg.eigh(Sigma_post)
    w = np.clip(w, 1e-10, None)
    Sigma_post = (V * w) @ V.T
    return mu_post, Sigma_post, sigma


def design_diagnostics(X, names=None):
    """Collinearity / range-restriction diagnostics for a design matrix.

    Returns a dict with:
      condition_number : sqrt of the ratio of largest/smallest eigenvalue of the
                         correlation matrix of the non-intercept columns.
      vif              : variance inflation factor per non-intercept column.
      effective_n      : sum of unique-row multiplicities normalised; reported as
                         the number of distinct genotype rows (an upper bound on the
                         information content when many isolates share a genotype).
    """
    X = np.asarray(X, float)
    n, p = X.shape
    cols = list(range(1, p)) if p > 1 else list(range(p))      # drop intercept
    names = names if names is not None else [f"x{j}" for j in range(p)]
    Z = X[:, cols]
    # Drop zero-variance columns from the conditioning analysis.
    sd = Z.std(axis=0)
    keep = sd > 1e-12
    Zk = Z[:, keep]
    kept_names = [names[cols[j]] for j in range(len(cols)) if keep[j]]

    vif = {}
    if Zk.shape[1] >= 2:
        # Correlation matrix condition number.
        C = np.corrcoef(Zk, rowvar=False)
        eig = np.linalg.eigvalsh(C)
        eig = np.clip(eig, 1e-12, None)
        cond = float(np.sqrt(eig.max() / eig.min()))
        # VIF: 1 / (1 - R^2) of each column regressed on the others.
        for j, nm in enumerate(kept_names):
            yj = Zk[:, j]
            Xj = np.delete(Zk, j, axis=1)
            Xj1 = np.hstack([np.ones((n, 1)), Xj])
            beta, *_ = np.linalg.lstsq(Xj1, yj, rcond=None)
            r2 = 1.0 - np.sum((yj - Xj1 @ beta)**2) / np.sum((yj - yj.mean())**2)
            vif[nm] = float(1.0 / max(1.0 - r2, 1e-6))
    else:
        cond = 1.0

    distinct_rows = len({tuple(np.round(r, 6)) for r in X})
    return dict(condition_number=cond, vif=vif,
                n_rows=n, distinct_genotypes=distinct_rows,
                zero_variance_cols=[names[cols[j]] for j in range(len(cols)) if not keep[j]])


# =============================================================================
# MPC / mutant-selection window  (Clark closed form)
# =============================================================================

def clark_max_moments(mu, Sig):
    """Clark (1961) pairwise recursion for E[max], Var[max], and cov(max, z_j)."""
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


def mpc_msw_posterior(mu_beta, Sigma_beta, X_nb, x_g0, logL_nb=None, logL_g0=0.0):
    """Closed-form posterior moments for log2 MPC and window width W.

    ``logL_nb``/``logL_g0`` carry the optional log2-Lambda (fitness/environment)
    shift per neighbour and for the founder; default zero (Lambda == 1).
    """
    mu_beta = np.asarray(mu_beta, float); Sigma_beta = np.asarray(Sigma_beta, float)
    X_nb = np.asarray(X_nb, float); x_g0 = np.asarray(x_g0, float)
    if logL_nb is None:
        logL_nb = np.zeros(len(X_nb))
    mu_z = X_nb @ mu_beta + logL_nb
    Sig_z = X_nb @ Sigma_beta @ X_nb.T
    m_mpc, v_mpc, cov_max = clark_max_moments(mu_z, Sig_z)
    mu_g0 = float(x_g0 @ mu_beta + logL_g0)
    var_g0 = float(x_g0 @ Sigma_beta @ x_g0)
    cov_z_nb_g0 = X_nb @ Sigma_beta @ x_g0
    alpha = np.linalg.lstsq(Sig_z, cov_max, rcond=None)[0]
    cov_mg0 = float(alpha @ cov_z_nb_g0)
    return dict(E_log2MPC=m_mpc, Var_log2MPC=v_mpc,
                E_W=m_mpc - mu_g0, Var_W=max(v_mpc + var_g0 - 2 * cov_mg0, 1e-12))


# =============================================================================
# Pharmacokinetics
# =============================================================================

def bateman_conc(t, F, D, Vd, ka, ke):
    """Oral first-order-absorption (Bateman) concentration at time t."""
    A = F * D * ka / (Vd * (ka - ke))
    return A * (np.exp(-ke * t) - np.exp(-ka * t))


def Tmsw_iv(W, t_half):
    """IV-bolus limit: time inside the window = half-life x window width (doublings)."""
    return t_half * W


def Tmsw_oral(mic, mpc, F, D, Vd, ka, ke):
    """Time inside [MIC, MPC] for an oral single dose (Bateman profile)."""
    def C(t):
        return bateman_conc(t, F, D, Vd, ka, ke)
    tmax = np.log(ka / ke) / (ka - ke)
    Cmax = C(tmax)
    if Cmax < mic:
        return 0.0

    def crossings(theta):
        if theta >= Cmax:
            return []
        up = brentq(lambda t: C(t) - theta, 1e-9, tmax)
        hi = tmax
        while C(hi) > theta and hi < tmax + 50 / ke:
            hi += 1.0 / ke
        down = brentq(lambda t: C(t) - theta, tmax, hi)
        return [up, down]

    mic_cr = crossings(mic)
    mpc_cr = crossings(mpc)
    if not mic_cr:
        return 0.0
    t_in_mic = mic_cr[1] - mic_cr[0]
    if not mpc_cr:
        return t_in_mic
    return t_in_mic - (mpc_cr[1] - mpc_cr[0])


# =============================================================================
# Stochastic refinement: soft MPC with a time-varying environment
# =============================================================================

def net_growth_rate(C, mic, psi_max, Emax, H):
    """Regoes/Hill per-capita net growth rate at concentration C.

    Calibrated so the net growth rate is zero at C = MIC (the MIC is the zero of
    the Hill-shaped curve), following Regoes et al. 2004. ``psi_max`` is the
    drug-free net growth rate, ``Emax`` the maximal kill rate, ``H`` the Hill
    coefficient. EC50 is solved from the MIC condition.
    """
    # At C = mic: psi_max - Emax * mic^H/(EC50^H + mic^H) = 0  ->  solve EC50.
    frac = psi_max / Emax
    if not (0.0 < frac < 1.0):
        raise ValueError("Require 0 < psi_max < Emax for a finite MIC.")
    ec50_H = mic**H * (1.0 - frac) / frac
    return psi_max - Emax * C**H / (ec50_H + C**H)


def establishment_probability(mic, psi_max, Emax, H, conc_profile, t_grid,
                              mutation_supply=1.0):
    """Time-varying-environment establishment probability of a resistant lineage.

    Replaces the constant-environment placeholder of manuscript Section 4.4. For a
    birth-death lineage experiencing the time-varying net growth rate r(t) =
    net_growth_rate(C(t), ...), the lineage-extinction theory for a time-
    inhomogeneous branching process gives a survival (establishment) probability
    governed by the integral of the instantaneous net growth rate while it is
    positive. We integrate r(t) along the supplied concentration profile and map
    the accumulated net growth to an establishment probability.

    Parameters
    ----------
    conc_profile : callable C(t) giving drug concentration at time t.
    t_grid : array of times spanning the dosing interval (defines the horizon).
    mutation_supply : scaling for the per-course mutational supply Theta_{g'}.

    Returns
    -------
    p_est : establishment probability in (0, 1).
    integral : the accumulated positive net growth (log-scale lineage growth).
    """
    t0, t1 = float(t_grid[0]), float(t_grid[-1])

    # Instantaneous single-lineage establishment probability (birth-death, Section
    # 4.4): p_inst(t) = max(0, r(C(t)) / psi_max). It is 1 when the drug is absent
    # (full growth advantage) and 0 at/above the mutant's suppression threshold.
    def p_inst(t):
        r = net_growth_rate(conc_profile(t), mic, psi_max, Emax, H)
        return max(r, 0.0) / psi_max

    # Per-course establishment hazard = mutational supply x time-integrated p_inst
    # along the *time-varying* concentration profile (replaces the constant-
    # environment placeholder). No-establishment prob = exp(-hazard).
    integral, _ = quad(p_inst, t0, t1, limit=200)
    hazard = mutation_supply * integral
    p_est = 1.0 - np.exp(-hazard)
    return float(np.clip(p_est, 0.0, 1.0)), float(integral)
