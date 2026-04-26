"""
Core Mandelbrot Ranking Distribution mathematics.

f(r) = C / (r + q)^s

where:
  r = rank (1-indexed)
  q = shift parameter (handles head of distribution)
  s = exponent (cost-entropy tradeoff parameter)
  C = normalization constant
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from dataclasses import dataclass
from typing import Optional


@dataclass
class MandelbrotParams:
    """Fitted Mandelbrot distribution parameters."""
    C: float
    q: float
    s: float
    log_likelihood: float
    n_tokens: int
    vocab_size: int


def mandelbrot_freq(ranks: np.ndarray, C: float, q: float, s: float) -> np.ndarray:
    """Compute Mandelbrot frequency for given ranks.

    f(r) = C / (r + q)^s
    """
    return C / (ranks + q) ** s


def mandelbrot_log_freq(ranks: np.ndarray, q: float, s: float) -> np.ndarray:
    """Compute log-frequency (unnormalized) for given ranks.

    log f(r) = -s * log(r + q)  [+ log C, absorbed into normalization]
    """
    return -s * np.log(ranks + q)


def mandelbrot_pmf(ranks: np.ndarray, q: float, s: float) -> np.ndarray:
    """Compute normalized Mandelbrot probability mass function."""
    log_f = mandelbrot_log_freq(ranks, q, s)
    log_f -= np.max(log_f)  # numerical stability
    f = np.exp(log_f)
    return f / f.sum()


def fit_mandelbrot_mle(
    ranks: np.ndarray,
    frequencies: np.ndarray,
    q_init: float = 2.7,
    s_init: float = 1.0,
) -> MandelbrotParams:
    """Fit Mandelbrot parameters (q, s) via maximum likelihood estimation.

    Given observed (rank, frequency) pairs, find q and s that maximize
    the likelihood of the observed frequency distribution under the
    Mandelbrot model.

    Args:
        ranks: 1-indexed ranks (1, 2, 3, ...)
        frequencies: observed token counts at each rank
        q_init: initial guess for shift parameter
        s_init: initial guess for exponent

    Returns:
        MandelbrotParams with fitted C, q, s and diagnostics
    """
    total_count = frequencies.sum()
    empirical_probs = frequencies / total_count

    def neg_log_likelihood(params):
        q, s = params
        if q < 0 or s <= 0:
            return 1e12
        log_pmf = mandelbrot_log_freq(ranks, q, s)
        log_Z = np.log(np.sum(np.exp(log_pmf - np.max(log_pmf)))) + np.max(log_pmf)
        log_probs = log_pmf - log_Z
        # Weighted by observed frequencies
        ll = np.sum(frequencies * log_probs)
        return -ll

    result = minimize(
        neg_log_likelihood,
        x0=[q_init, s_init],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10},
    )

    q_fit, s_fit = result.x
    # Compute C from normalization
    unnorm = 1.0 / (ranks + q_fit) ** s_fit
    C_fit = total_count / unnorm.sum()

    return MandelbrotParams(
        C=C_fit,
        q=q_fit,
        s=s_fit,
        log_likelihood=-result.fun,
        n_tokens=int(total_count),
        vocab_size=len(ranks),
    )


def fit_mandelbrot_ols_loglog(
    ranks: np.ndarray,
    frequencies: np.ndarray,
) -> MandelbrotParams:
    """Fit Mandelbrot parameters via OLS on log-log scale.

    This is a simpler alternative to MLE. Fits log(f) = log(C) - s*log(r+q)
    by grid search over q, then linear regression for each q.

    Biased toward the body of the distribution but useful as a comparison.
    """
    log_freq = np.log(frequencies.astype(float))
    mask = frequencies > 0
    log_freq_valid = log_freq[mask]
    ranks_valid = ranks[mask]

    best_r2 = -np.inf
    best_params = (0.0, 1.0, 1.0)

    for q_candidate in np.linspace(0.0, 10.0, 200):
        log_r_shifted = np.log(ranks_valid + q_candidate)
        # Linear regression: log(f) = log(C) - s * log(r + q)
        A = np.column_stack([np.ones_like(log_r_shifted), -log_r_shifted])
        coeffs, residuals, _, _ = np.linalg.lstsq(A, log_freq_valid, rcond=None)
        log_C, s = coeffs[0], coeffs[1]

        # R^2
        predicted = A @ coeffs
        ss_res = np.sum((log_freq_valid - predicted) ** 2)
        ss_tot = np.sum((log_freq_valid - log_freq_valid.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 > best_r2:
            best_r2 = r2
            best_params = (np.exp(log_C), q_candidate, s)

    C_fit, q_fit, s_fit = best_params
    return MandelbrotParams(
        C=C_fit,
        q=q_fit,
        s=s_fit,
        log_likelihood=best_r2,  # Store R^2 here for OLS
        n_tokens=int(frequencies.sum()),
        vocab_size=len(ranks),
    )


def goodness_of_fit(
    ranks: np.ndarray,
    observed_freq: np.ndarray,
    params: MandelbrotParams,
) -> dict:
    """Compute goodness-of-fit metrics for a Mandelbrot fit.

    Returns:
        dict with keys: r_squared, ks_statistic, ks_pvalue,
        residuals_by_region (head/body/tail)
    """
    predicted_freq = mandelbrot_freq(ranks, params.C, params.q, params.s)

    # R^2 on log-log scale
    mask = observed_freq > 0
    log_obs = np.log(observed_freq[mask].astype(float))
    log_pred = np.log(predicted_freq[mask])
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - log_obs.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # KS statistic on normalized distributions
    obs_pmf = observed_freq / observed_freq.sum()
    pred_pmf = predicted_freq / predicted_freq.sum()
    obs_cdf = np.cumsum(obs_pmf)
    pred_cdf = np.cumsum(pred_pmf)
    ks_statistic = np.max(np.abs(obs_cdf - pred_cdf))

    # Residuals by region (Section 9.5 predictions)
    log_residuals = log_obs - log_pred[mask]
    head_mask = ranks[mask] <= 10
    body_mask = (ranks[mask] > 10) & (ranks[mask] <= 50000)
    tail_mask = ranks[mask] > 50000

    residuals_by_region = {}
    for name, region_mask in [("head", head_mask), ("body", body_mask), ("tail", tail_mask)]:
        if region_mask.any():
            residuals_by_region[name] = {
                "mean_residual": float(np.mean(log_residuals[region_mask])),
                "std_residual": float(np.std(log_residuals[region_mask])),
                "max_abs_residual": float(np.max(np.abs(log_residuals[region_mask]))),
                "n_tokens": int(region_mask.sum()),
            }

    return {
        "r_squared": float(r_squared),
        "ks_statistic": float(ks_statistic),
        "residuals_by_region": residuals_by_region,
    }


def compare_distributions(
    ranks: np.ndarray,
    observed_freq: np.ndarray,
) -> dict:
    """Compare Mandelbrot fit against alternative distributions.

    Fits: Mandelbrot (3 params), pure Zipf (1 param, q=0), log-normal.
    Returns AIC/BIC for model comparison.
    """
    n = observed_freq.sum()

    # Mandelbrot fit (2 free params: q, s; C determined by normalization)
    mandelbrot_fit = fit_mandelbrot_mle(ranks, observed_freq)

    # Pure Zipf fit (1 free param: s; q=0, C determined)
    def zipf_nll(params):
        s = params[0]
        if s <= 0:
            return 1e12
        log_pmf = -s * np.log(ranks.astype(float))
        log_Z = np.log(np.sum(np.exp(log_pmf - np.max(log_pmf)))) + np.max(log_pmf)
        log_probs = log_pmf - log_Z
        return -np.sum(observed_freq * log_probs)

    zipf_result = minimize(zipf_nll, x0=[1.0], method="Nelder-Mead")
    zipf_ll = -zipf_result.fun

    # AIC and BIC
    k_mandelbrot = 2  # q, s
    k_zipf = 1  # s only

    results = {
        "mandelbrot": {
            "params": {"C": mandelbrot_fit.C, "q": mandelbrot_fit.q, "s": mandelbrot_fit.s},
            "log_likelihood": mandelbrot_fit.log_likelihood,
            "aic": 2 * k_mandelbrot - 2 * mandelbrot_fit.log_likelihood,
            "bic": k_mandelbrot * np.log(n) - 2 * mandelbrot_fit.log_likelihood,
        },
        "zipf": {
            "params": {"s": zipf_result.x[0]},
            "log_likelihood": zipf_ll,
            "aic": 2 * k_zipf - 2 * zipf_ll,
            "bic": k_zipf * np.log(n) - 2 * zipf_ll,
        },
    }

    return results
