"""Run the alternative-slab prior sensitivity analysis for the Mroz data."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import invgamma, rankdata

REPO_ROOT = Path(__file__).resolve().parents[3]


DATA_FILE = REPO_ROOT / "data" / "processed" / "mroz" / "mroz_zuehlke_design.csv"

OUTPUT_DIR = REPO_ROOT / "results" / "empirical" / "mroz" / "prior_sensitivity_tau1"

N_CHAINS = 4
N_MCMC = 50000
BURN_IN = 25000
CHAIN_SEEDS = [2026101901, 2026101902, 2026101903, 2026101904]


TAU0 = 0.10
TAU1 = 1.00
SIGMA12_PRIOR_VARIANCE = 0.25
SIGMA_V2_PRIOR_SHAPE = 3.0
SIGMA_V2_PRIOR_SCALE = 2.0
LAMBDA_LOWER = 0.01
LAMBDA_UPPER = 5.0


WAGE_GEOMETRIC_MEAN = 3.28765090410458


MARGINAL_EFFECT_GH_NODES = 40


W_COLUMNS = [
    "w_intercept",
    "w_age",
    "w_age2",
    "w_faminc",
    "w_child",
    "w_educ",
]

X_COLUMNS = [
    "x_intercept",
    "x_educ",
    "x_exper",
    "x_exper2",
    "x_city",
]


def bd_transform(y, lam):
    return (np.sign(y) * np.abs(y) ** lam - 1.0) / lam


def bd_inverse(t, lam):
    value = 1.0 + lam * t
    return np.sign(value) * np.abs(value) ** (1.0 / lam)


def load_mroz_data():
    data = pd.read_csv(DATA_FILE)

    required = ["row_id", "d", "y_observed", "y_scaled"] + W_COLUMNS + X_COLUMNS
    if list(data.columns) != required:
        raise ValueError(
            "设计文件的列名或顺序已经改变。\n"
            f"应为：{required}\n"
            f"实际：{list(data.columns)}"
        )

    if len(data) != 753:
        raise ValueError(f"样本量应为753，实际为{len(data)}。")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id 存在重复。")

    d = data["d"].to_numpy(dtype=int)
    if not np.isin(d, [0, 1]).all():
        raise ValueError("d 必须只包含0和1。")

    if int(np.sum(d == 1)) != 428 or int(np.sum(d == 0)) != 325:
        raise ValueError("就业/未就业人数与冻结数据不一致。")

    selected = d == 1
    missing = d == 0

    if data.loc[selected, "y_scaled"].isna().any():
        raise ValueError("d=1时 y_scaled 不应缺失。")
    if (data.loc[selected, "y_scaled"] <= 0).any():
        raise ValueError("d=1时 y_scaled 必须为正。")
    if data.loc[missing, "y_scaled"].notna().any():
        raise ValueError("d=0时 y_scaled 必须保持缺失，不能补成0。")
    if data.loc[missing, "y_observed"].notna().any():
        raise ValueError("d=0时 y_observed 必须保持缺失。")

    if data[W_COLUMNS + X_COLUMNS].isna().any().any():
        raise ValueError("设计矩阵中存在缺失值。")

    
    expected_scaled = (
        data.loc[selected, "y_observed"].to_numpy(dtype=float)
        / WAGE_GEOMETRIC_MEAN
    )
    actual_scaled = data.loc[selected, "y_scaled"].to_numpy(dtype=float)
    if not np.allclose(expected_scaled, actual_scaled, rtol=0.0, atol=1e-12):
        raise ValueError("y_scaled 与冻结的工资几何均值不一致。")

    x = data[X_COLUMNS].to_numpy(dtype=float)
    w = data[W_COLUMNS].to_numpy(dtype=float)
    y_scaled = data["y_scaled"].to_numpy(dtype=float)

    return data, x, w, d, y_scaled


def draw_truncated_normal(mean, sd, lower=None, upper=None, rng=None):
    mean = np.asarray(mean)

    if lower is not None:
        cdf_lower = ndtr((lower - mean) / sd)
        uniform = cdf_lower + (1.0 - cdf_lower) * rng.random(mean.shape)
    elif upper is not None:
        cdf_upper = ndtr((upper - mean) / sd)
        uniform = cdf_upper * rng.random(mean.shape)
    else:
        return mean + sd * rng.normal(size=mean.shape)

    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    return mean + sd * ndtri(uniform)


def lambda_log_target(
    lam,
    observed_y,
    log_abs_y_sum,
    conditional_mean,
    sigma_v2,
):
    if lam < LAMBDA_LOWER or lam > LAMBDA_UPPER:
        return -np.inf

    transformed_y = bd_transform(observed_y, lam)
    residual = transformed_y - conditional_mean

    log_likelihood = -0.5 * np.sum(residual ** 2) / sigma_v2
    log_jacobian = (lam - 1.0) * log_abs_y_sum

    return log_likelihood + log_jacobian


def draw_normal_from_precision(precision, linear_term, rng):
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, linear_term))
    random_part = np.linalg.solve(chol.T, rng.normal(size=precision.shape[0]))
    return mean + random_part


def update_gamma_vector(coefficients, coefficient_scales, inclusion_rate, rng):
    safe_p = np.clip(inclusion_rate, 1e-12, 1.0 - 1e-12)
    standardized_effects = coefficients * coefficient_scales

    log_slab = (
        np.log(safe_p)
        - np.log(TAU1)
        - 0.5 * (standardized_effects / TAU1) ** 2
    )
    log_spike = (
        np.log1p(-safe_p)
        - np.log(TAU0)
        - 0.5 * (standardized_effects / TAU0) ** 2
    )

    log_difference = np.clip(log_spike - log_slab, -700.0, 700.0)
    inclusion_probability = 1.0 / (1.0 + np.exp(log_difference))
    return (rng.random(coefficients.size) < inclusion_probability).astype(int)


def run_mcmc(x, w, d, y_scaled, rng):
    n = len(d)
    k_alpha = w.shape[1]
    k_beta = x.shape[1]
    p_alpha = k_alpha - 1
    p_beta = k_beta - 1

    selected = d == 1
    missing = d == 0

    x_selected = x[selected]
    w_selected = w[selected]
    x_missing = x[missing]
    w_missing = w[missing]
    observed_y_selected = y_scaled[selected]
    n_missing = int(np.sum(missing))

    xtx = x.T @ x
    wtw = w.T @ w
    log_abs_y_sum = np.sum(np.log(np.abs(observed_y_selected)))

    
    
    alpha_scales = np.std(w[:, 1:], axis=0, ddof=0)
    beta_scales = np.std(x[:, 1:], axis=0, ddof=0)
    if np.any(alpha_scales <= 0) or np.any(beta_scales <= 0):
        raise ValueError("用于变量选择的解释变量必须具有正标准差。")

    alpha = np.zeros(k_alpha)
    beta = np.zeros(k_beta)
    sigma12 = 0.0
    sigma_v2 = 1.0
    lam = 1.0

    gamma_alpha = np.ones(p_alpha, dtype=int)
    gamma_beta = np.ones(p_beta, dtype=int)
    inclusion_alpha = 0.5
    inclusion_beta = 0.5

    z_star = np.where(selected, 0.5, -0.5).astype(float)
    t_star = np.zeros(n)
    t_star[selected] = bd_transform(observed_y_selected, lam)

    n_keep = N_MCMC - BURN_IN
    alpha_save = np.zeros((n_keep, k_alpha))
    beta_save = np.zeros((n_keep, k_beta))
    gamma_alpha_save = np.zeros((n_keep, p_alpha), dtype=np.int8)
    gamma_beta_save = np.zeros((n_keep, p_beta), dtype=np.int8)
    sigma12_save = np.zeros(n_keep)
    sigma_v2_save = np.zeros(n_keep)
    lambda_save = np.zeros(n_keep)

    proposal_sd = 0.10
    accepted_total = 0
    accepted_window = 0

    for iteration in range(N_MCMC):

        
        sigma_e2 = sigma_v2 + sigma12 ** 2
        t_star[selected] = bd_transform(observed_y_selected, lam)

        z_mean_selected = (
            w_selected @ alpha
            + sigma12 / sigma_e2
            * (t_star[selected] - x_selected @ beta)
        )
        z_sd_selected = math.sqrt(sigma_v2 / sigma_e2)

        z_star[selected] = draw_truncated_normal(
            z_mean_selected,
            z_sd_selected,
            lower=0.0,
            rng=rng,
        )
        z_star[missing] = draw_truncated_normal(
            w_missing @ alpha,
            1.0,
            upper=0.0,
            rng=rng,
        )

        t_mean_missing = (
            x_missing @ beta
            + sigma12 * (z_star[missing] - w_missing @ alpha)
        )
        t_star[missing] = (
            t_mean_missing
            + rng.normal(0.0, math.sqrt(sigma_v2), size=n_missing)
        )

        
        inclusion_alpha = rng.beta(
            1 + np.sum(gamma_alpha),
            1 + p_alpha - np.sum(gamma_alpha),
        )
        inclusion_beta = rng.beta(
            1 + np.sum(gamma_beta),
            1 + p_beta - np.sum(gamma_beta),
        )

        
        gamma_alpha = update_gamma_vector(
            alpha[1:],
            alpha_scales,
            inclusion_alpha,
            rng,
        )
        gamma_beta = update_gamma_vector(
            beta[1:],
            beta_scales,
            inclusion_beta,
            rng,
        )

        
        alpha_prior_variance = np.r_[
            100.0,
            np.where(
                gamma_alpha == 1,
                (TAU1 / alpha_scales) ** 2,
                (TAU0 / alpha_scales) ** 2,
            ),
        ]

        q = t_star - x @ beta - sigma12 * z_star
        alpha_precision = (
            (1.0 + sigma12 ** 2 / sigma_v2) * wtw
            + np.diag(1.0 / alpha_prior_variance)
        )
        alpha_linear = (
            w.T @ z_star
            - sigma12 / sigma_v2 * (w.T @ q)
        )
        alpha = draw_normal_from_precision(
            alpha_precision,
            alpha_linear,
            rng,
        )

        
        omega = z_star - w @ alpha
        adjusted_t = t_star - sigma12 * omega

        beta_prior_variance = np.r_[
            100.0,
            np.where(
                gamma_beta == 1,
                (TAU1 / beta_scales) ** 2,
                (TAU0 / beta_scales) ** 2,
            ),
        ]
        beta_precision = (
            xtx / sigma_v2
            + np.diag(1.0 / beta_prior_variance)
        )
        beta_linear = x.T @ adjusted_t / sigma_v2
        beta = draw_normal_from_precision(
            beta_precision,
            beta_linear,
            rng,
        )

        
        omega = z_star - w @ alpha
        outcome_residual = t_star - x @ beta

        sigma12_variance = 1.0 / (
            np.sum(omega ** 2) / sigma_v2
            + 1.0 / SIGMA12_PRIOR_VARIANCE
        )
        sigma12_mean = sigma12_variance * (
            np.sum(omega * outcome_residual) / sigma_v2
        )
        sigma12 = rng.normal(
            sigma12_mean,
            math.sqrt(sigma12_variance),
        )

        
        v_residual = outcome_residual - sigma12 * omega
        posterior_shape = SIGMA_V2_PRIOR_SHAPE + n / 2.0
        posterior_scale = (
            SIGMA_V2_PRIOR_SCALE
            + 0.5 * np.sum(v_residual ** 2)
        )
        sigma_v2 = invgamma.rvs(
            a=posterior_shape,
            scale=posterior_scale,
            random_state=rng,
        )

        
        conditional_t_mean = (
            x_selected @ beta
            + sigma12 * (z_star[selected] - w_selected @ alpha)
        )

        proposal = float(draw_truncated_normal(
            lam,
            proposal_sd,
            lower=0.0,
            rng=rng,
        ))

        log_acceptance = (
            lambda_log_target(
                proposal,
                observed_y_selected,
                log_abs_y_sum,
                conditional_t_mean,
                sigma_v2,
            )
            - lambda_log_target(
                lam,
                observed_y_selected,
                log_abs_y_sum,
                conditional_t_mean,
                sigma_v2,
            )
        )

        
        log_acceptance += (
            log_ndtr(lam / proposal_sd)
            - log_ndtr(proposal / proposal_sd)
        )

        if np.log(rng.random()) < min(0.0, log_acceptance):
            lam = proposal
            accepted_total += 1
            accepted_window += 1

        if iteration < BURN_IN and (iteration + 1) % 100 == 0:
            window_acceptance = accepted_window / 100.0
            proposal_sd *= math.exp(0.10 * (window_acceptance - 0.44))
            proposal_sd = float(np.clip(proposal_sd, 0.005, 1.0))
            accepted_window = 0

        if iteration >= BURN_IN:
            j = iteration - BURN_IN
            alpha_save[j] = alpha
            beta_save[j] = beta
            gamma_alpha_save[j] = gamma_alpha
            gamma_beta_save[j] = gamma_beta
            sigma12_save[j] = sigma12
            sigma_v2_save[j] = sigma_v2
            lambda_save[j] = lam

    rho_save = sigma12_save / np.sqrt(
        sigma12_save ** 2 + sigma_v2_save
    )

    return {
        "alpha": alpha_save,
        "beta": beta_save,
        "gamma_alpha": gamma_alpha_save,
        "gamma_beta": gamma_beta_save,
        "sigma12": sigma12_save,
        "sigma_v2": sigma_v2_save,
        "lambda": lambda_save,
        "rho": rho_save,
        "acceptance_rate": accepted_total / N_MCMC,
        "proposal_sd": proposal_sd,
    }


def split_chains(draws):
    draws = np.asarray(draws, dtype=float)
    n = draws.shape[1]
    if n % 2 == 1:
        draws = draws[:, :-1]
        n -= 1
    half = n // 2
    return np.concatenate((draws[:, :half], draws[:, half:]), axis=0)


def basic_rhat(draws):
    m, n = draws.shape
    chain_means = np.mean(draws, axis=1)
    within = np.mean(np.var(draws, axis=1, ddof=1))
    if within < 1e-15:
        return 1.0
    between = n * np.var(chain_means, ddof=1)
    variance = (n - 1.0) / n * within + between / n
    return float(math.sqrt(variance / within))


def rank_normalize(draws):
    flat = np.asarray(draws, dtype=float).ravel()
    ranks = rankdata(flat, method="average")
    probabilities = (ranks - 3.0 / 8.0) / (len(flat) + 1.0 / 4.0)
    normalized = ndtri(probabilities)
    return normalized.reshape(np.asarray(draws).shape)


def rank_split_rhat(draws):
    split = split_chains(draws)
    rank_rhat = basic_rhat(rank_normalize(split))

    folded = np.abs(split - np.median(split))
    folded_rhat = basic_rhat(rank_normalize(folded))
    return max(rank_rhat, folded_rhat)


def autocovariance_fft(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    centered = values - np.mean(values)
    size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=size,
    )[:n]
    return covariance / n


def multichain_ess_core(draws):
    draws = split_chains(draws)
    m, n = draws.shape

    autocov = np.vstack([autocovariance_fft(chain) for chain in draws])
    within = float(np.mean(autocov[:, 0]))
    if within < 1e-15:
        return float(m * n)

    between = n * np.var(np.mean(draws, axis=1), ddof=1)
    variance_plus = (n - 1.0) / n * within + between / n

    rho = np.ones(n)
    for lag in range(1, n):
        rho[lag] = 1.0 - (
            within - float(np.mean(autocov[:, lag]))
        ) / variance_plus

    pair_sums = []
    for lag in range(0, n - 1, 2):
        pair_sum = rho[lag] + rho[lag + 1]
        if pair_sum < 0:
            break
        pair_sums.append(pair_sum)

    if not pair_sums:
        return float(m * n)

    for j in range(1, len(pair_sums)):
        pair_sums[j] = min(pair_sums[j], pair_sums[j - 1])

    autocorrelation_time = max(-1.0 + 2.0 * sum(pair_sums), 1.0)
    return float(min(m * n / autocorrelation_time, m * n))


def bulk_ess(draws):
    return multichain_ess_core(rank_normalize(draws))


def tail_ess(draws):
    values = np.asarray(draws, dtype=float)
    lower, upper = np.quantile(values, [0.05, 0.95])
    lower_indicator = (values <= lower).astype(float)
    upper_indicator = (values >= upper).astype(float)
    return min(
        multichain_ess_core(lower_indicator),
        multichain_ess_core(upper_indicator),
    )


def stack_parameter_draws(chain_results):
    parameter_draws = {}

    for j, column in enumerate(W_COLUMNS):
        parameter_draws[f"alpha_{column[2:]}"] = np.stack(
            [result["alpha"][:, j] for result in chain_results]
        )

    for j, column in enumerate(X_COLUMNS):
        parameter_draws[f"beta_{column[2:]}"] = np.stack(
            [result["beta"][:, j] for result in chain_results]
        )

    for name in ["lambda", "sigma12", "sigma_v2", "rho"]:
        parameter_draws[name] = np.stack(
            [result[name] for result in chain_results]
        )

    return parameter_draws


def posterior_summary_table(parameter_draws):
    rows = []
    for name, chains in parameter_draws.items():
        draws = chains.ravel()
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append({
            "parameter": name,
            "mean": np.mean(draws),
            "sd": np.std(draws, ddof=1),
            "median": np.median(draws),
            "lower_95": lower,
            "upper_95": upper,
        })
    return pd.DataFrame(rows)


def diagnostic_table(parameter_draws):
    rows = []
    for name, chains in parameter_draws.items():
        draws = chains.ravel()
        ess_bulk = bulk_ess(chains)
        ess_tail = tail_ess(chains)
        posterior_sd = np.std(draws, ddof=1)
        rows.append({
            "parameter": name,
            "rank_split_rhat": rank_split_rhat(chains),
            "bulk_ess": ess_bulk,
            "tail_ess": ess_tail,
            "mcse_mean": posterior_sd / math.sqrt(ess_bulk),
            "posterior_sd": posterior_sd,
        })
    return pd.DataFrame(rows)


def rhat_evolution_table(
    parameter_draws,
    method_label,
    burn_in,
    n_points=30,
    min_kept_draws=1000,
):
    """Calculate cumulative rank-normalized split-Rhat diagnostics."""
    if not parameter_draws:
        raise ValueError("parameter_draws is empty.")

    n_kept = min(
        np.asarray(chains).shape[1]
        for chains in parameter_draws.values()
    )
    if n_kept < min_kept_draws:
        raise ValueError(
            f"Only {n_kept} retained draws are available; "
            f"at least {min_kept_draws} are required."
        )

    cutoffs = np.unique(np.rint(np.linspace(
        min_kept_draws, n_kept, n_points
    )).astype(int))
    rows = []
    for cutoff in cutoffs:
        parameter_rhats = {
            parameter: rank_split_rhat(
                np.asarray(chains, dtype=float)[:, :cutoff]
            )
            for parameter, chains in parameter_draws.items()
        }
        finite_rhats = {
            parameter: value
            for parameter, value in parameter_rhats.items()
            if np.isfinite(value)
        }
        if not finite_rhats:
            raise RuntimeError(
                f"No finite Rhat values at cutoff {cutoff}."
            )
        worst_parameter = max(finite_rhats, key=finite_rhats.get)
        values = np.asarray(list(finite_rhats.values()), dtype=float)
        rows.append({
            "method": method_label,
            "retained_draws_per_chain": cutoff,
            "total_iteration": burn_in + cutoff,
            "max_rhat": finite_rhats[worst_parameter],
            "median_rhat": float(np.median(values)),
            "parameters_above_1_01": int(np.sum(values > 1.01)),
            "worst_parameter": worst_parameter,
        })
    return pd.DataFrame(rows)


def variable_selection_table(chain_results):
    gamma_alpha = np.concatenate(
        [result["gamma_alpha"] for result in chain_results],
        axis=0,
    )
    gamma_beta = np.concatenate(
        [result["gamma_beta"] for result in chain_results],
        axis=0,
    )
    alpha = np.concatenate(
        [result["alpha"] for result in chain_results],
        axis=0,
    )
    beta = np.concatenate(
        [result["beta"] for result in chain_results],
        axis=0,
    )

    rows = []

    for j, column in enumerate(W_COLUMNS[1:]):
        draws = alpha[:, j + 1]
        lower, upper = np.quantile(draws, [0.025, 0.975])
        pip = float(np.mean(gamma_alpha[:, j]))
        rows.append({
            "equation": "selection",
            "variable": column[2:],
            "pip": pip,
            "selected_pip_gt_0_5": int(pip > 0.5),
            "coefficient_mean": np.mean(draws),
            "coefficient_lower_95": lower,
            "coefficient_upper_95": upper,
        })

    for j, column in enumerate(X_COLUMNS[1:]):
        draws = beta[:, j + 1]
        lower, upper = np.quantile(draws, [0.025, 0.975])
        pip = float(np.mean(gamma_beta[:, j]))
        rows.append({
            "equation": "outcome",
            "variable": column[2:],
            "pip": pip,
            "selected_pip_gt_0_5": int(pip > 0.5),
            "coefficient_mean": np.mean(draws),
            "coefficient_lower_95": lower,
            "coefficient_upper_95": upper,
        })

    return pd.DataFrame(rows)


def inverse_mills(index):
    log_phi = -0.5 * index ** 2 - 0.5 * math.log(2.0 * math.pi)
    return np.exp(np.clip(log_phi - log_ndtr(index), -700.0, 700.0))


def expected_inverse_and_derivative_factor(mu, variance, lam, nodes, weights):
    latent = (
        mu[:, None]
        + math.sqrt(2.0 * variance) * nodes[None, :]
    )
    y_scaled = bd_inverse(latent, lam)
    expected_y = y_scaled @ weights / math.sqrt(math.pi)
    derivative_factor = (
        np.abs(y_scaled) ** (1.0 - lam)
    ) @ weights / math.sqrt(math.pi)
    return expected_y, derivative_factor


def _legacy_partial_association_table(data, x, w, d, chain_results):
    alpha = np.mean(np.concatenate(
        [result["alpha"] for result in chain_results],
        axis=0,
    ), axis=0)
    beta = np.mean(np.concatenate(
        [result["beta"] for result in chain_results],
        axis=0,
    ), axis=0)
    sigma12 = float(np.mean(np.concatenate(
        [result["sigma12"] for result in chain_results]
    )))
    sigma_v2 = float(np.mean(np.concatenate(
        [result["sigma_v2"] for result in chain_results]
    )))
    lam = float(np.mean(np.concatenate(
        [result["lambda"] for result in chain_results]
    )))

    selected = d == 1
    selection_index = w @ alpha
    phi = np.exp(-0.5 * selection_index ** 2) / math.sqrt(2.0 * math.pi)

    
    age = data["w_age"].to_numpy(dtype=float)
    selection_effects = {
        "age": np.mean(phi * (alpha[1] + 2.0 * age * alpha[2])),
        "faminc": 10000.0 * np.mean(phi * alpha[3]),
        "child": np.mean(phi * alpha[4]),
        "educ": np.mean(phi * alpha[5]),
    }

    rows = []
    for variable, estimate in selection_effects.items():
        rows.append({
            "equation": "selection",
            "variable": variable,
            "scale": "probability",
            "population": "all 753 observations",
            "estimand": "average partial association with Pr(d=1|w)",
            "estimate": estimate,
        })

    
    mills = inverse_mills(selection_index[selected])
    mills_derivative = -mills * (
        selection_index[selected] + mills
    )

    beta_educ = beta[1]
    beta_exper = beta[2]
    beta_exper2 = beta[3]
    beta_city = beta[4]
    alpha_educ = alpha[5]
    experience_selected = x[selected, 2]

    transformed_latent = {
        "educ": beta_educ,
        "exper": np.mean(
            beta_exper + 2.0 * experience_selected * beta_exper2
        ),
        "city": beta_city,
    }
    transformed_selected = {
        "educ": np.mean(
            beta_educ
            + sigma12 * mills_derivative * alpha_educ
        ),
        "exper": transformed_latent["exper"],
        "city": beta_city,
    }

    for variable in ["educ", "exper", "city"]:
        rows.append({
            "equation": "outcome",
            "variable": variable,
            "scale": "transformed",
            "population": "observed wage subsample",
            "estimand": "average partial association with E[t(y)|d=1,w,x]",
            "estimate": transformed_selected[variable],
        })
        rows.append({
            "equation": "outcome",
            "variable": variable,
            "scale": "transformed",
            "population": "latent outcome",
            "estimand": "partial association with E[t(y*)|x]",
            "estimate": transformed_latent[variable],
        })

    
    nodes, weights = hermgauss(40)
    total_variance = sigma12 ** 2 + sigma_v2
    mu = x @ beta
    _, derivative_factor = expected_inverse_and_derivative_factor(
        mu,
        total_variance,
        lam,
        nodes,
        weights,
    )

    original_educ = (
        WAGE_GEOMETRIC_MEAN
        * np.mean(beta_educ * derivative_factor)
    )
    original_exper = (
        WAGE_GEOMETRIC_MEAN
        * np.mean(
            (beta_exper + 2.0 * x[:, 2] * beta_exper2)
            * derivative_factor
        )
    )

    x_city0 = x.copy()
    x_city1 = x.copy()
    x_city0[:, 4] = 0.0
    x_city1[:, 4] = 1.0
    expected_city0, _ = expected_inverse_and_derivative_factor(
        x_city0 @ beta,
        total_variance,
        lam,
        nodes,
        weights,
    )
    expected_city1, _ = expected_inverse_and_derivative_factor(
        x_city1 @ beta,
        total_variance,
        lam,
        nodes,
        weights,
    )
    original_city = WAGE_GEOMETRIC_MEAN * np.mean(
        expected_city1 - expected_city0
    )

    original_effects = {
        "educ": original_educ,
        "exper": original_exper,
        "city": original_city,
    }

    for variable, estimate in original_effects.items():
        estimand = (
            "average discrete change from city=0 to city=1"
            if variable == "city"
            else "average partial association with latent hourly wage"
        )
        rows.append({
            "equation": "outcome",
            "variable": variable,
            "scale": "original hourly wage dollars",
            "population": "all 753 observations",
            "estimand": estimand,
            "estimate": estimate,
        })

    return pd.DataFrame(rows)


def posterior_marginal_effect_tables(data, x, w, chain_results):
    alpha = np.stack([r["alpha"] for r in chain_results])
    beta = np.stack([r["beta"] for r in chain_results])
    sigma12 = np.stack([r["sigma12"] for r in chain_results])
    sigma_v2 = np.stack([r["sigma_v2"] for r in chain_results])
    lam = np.stack([r["lambda"] for r in chain_results])
    n_chains, n_draws = sigma12.shape
    names = [
        "selection_age", "selection_family_income", "selection_children",
        "selection_education", "outcome_education", "outcome_experience",
        "outcome_large_city",
    ]
    effects = {name: np.empty((n_chains, n_draws)) for name in names}
    nodes, weights = hermgauss(MARGINAL_EFFECT_GH_NODES)
    age = data["w_age"].to_numpy(dtype=float)

    for c in range(n_chains):
        for s in range(n_draws):
            a, b = alpha[c, s], beta[c, s]
            variance = sigma12[c, s] ** 2 + sigma_v2[c, s]
            draw_lambda = lam[c, s]
            index = w @ a
            pdf = np.exp(-0.5 * index ** 2) / math.sqrt(2.0 * math.pi)
            effects["selection_age"][c, s] = np.mean(
                pdf * (a[1] + 2.0 * age * a[2])
            )
            effects["selection_family_income"][c, s] = (
                    10000.0 * np.mean(pdf * a[3])
            )
            effects["selection_children"][c, s] = np.mean(pdf * a[4])
            effects["selection_education"][c, s] = np.mean(pdf * a[5])

            _, factor = expected_inverse_and_derivative_factor(
                x @ b, variance, draw_lambda, nodes, weights
            )
            effects["outcome_education"][c, s] = (
                WAGE_GEOMETRIC_MEAN * np.mean(b[1] * factor)
            )
            effects["outcome_experience"][c, s] = (
                WAGE_GEOMETRIC_MEAN
                * np.mean((b[2] + 2.0 * x[:, 2] * b[3]) * factor)
            )
            x0, x1 = x.copy(), x.copy()
            x0[:, 4], x1[:, 4] = 0.0, 1.0
            y0, _ = expected_inverse_and_derivative_factor(
                x0 @ b, variance, draw_lambda, nodes, weights
            )
            y1, _ = expected_inverse_and_derivative_factor(
                x1 @ b, variance, draw_lambda, nodes, weights
            )
            effects["outcome_large_city"][c, s] = (
                WAGE_GEOMETRIC_MEAN * np.mean(y1 - y0)
            )
        print(f"  marginal effects: chain {c + 1}/{n_chains} completed")

    metadata = {
        "selection_age": ("selection", "age", "probability", "one year"),
        "selection_family_income": (
            "selection", "family income", "probability", "$10,000"
        ),
        "selection_children": (
            "selection", "children", "probability", "one child"
        ),
        "selection_education": (
            "selection", "education", "probability", "one year"
        ),
        "outcome_education": (
            "outcome", "education", "latent hourly wage (dollars)", "one year"
        ),
        "outcome_experience": (
            "outcome", "experience", "latent hourly wage (dollars)", "one year"
        ),
        "outcome_large_city": (
            "outcome", "large city", "latent hourly wage (dollars)", "0 to 1"
        ),
    }
    draw_table = {
        "chain": np.repeat(np.arange(1, n_chains + 1), n_draws),
        "draw": np.tile(np.arange(1, n_draws + 1), n_chains),
    }
    rows = []
    for name, values in effects.items():
        pooled = values.reshape(-1)
        equation, variable, scale, change = metadata[name]
        rows.append({
            "effect": name, "equation": equation, "variable": variable,
            "scale": scale, "change": change,
            "posterior_mean": np.mean(pooled),
            "posterior_median": np.median(pooled),
            "posterior_sd": np.std(pooled, ddof=1),
            "lower_95": np.quantile(pooled, 0.025),
            "upper_95": np.quantile(pooled, 0.975),
            "probability_positive": np.mean(pooled > 0.0),
        })
        draw_table[name] = pooled
    return (
        pd.DataFrame(draw_table),
        pd.DataFrame(rows),
        diagnostic_table(effects),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if N_MCMC <= BURN_IN:
        raise ValueError("N_MCMC 必须大于 BURN_IN。")
    if N_CHAINS != len(CHAIN_SEEDS):
        raise ValueError("N_CHAINS 必须与 CHAIN_SEEDS 长度一致。")

    data, x, w, d, y_scaled = load_mroz_data()

    print("Mroz proposed method: prior sensitivity tau0=0.10, tau1=1.00")
    print(f"  n={len(data)}, observed wages={int(np.sum(d))}")
    print(
        f"  chains={N_CHAINS}, iterations={N_MCMC}, "
        f"burn-in={BURN_IN}"
    )

    start_time = time.perf_counter()
    chain_results = []
    chain_rows = []

    for chain in range(N_CHAINS):
        print(f"Chain {chain + 1}/{N_CHAINS}")
        chain_start = time.perf_counter()
        rng = np.random.default_rng(CHAIN_SEEDS[chain])
        result = run_mcmc(x, w, d, y_scaled, rng)
        chain_results.append(result)

        elapsed = time.perf_counter() - chain_start
        chain_rows.append({
            "chain": chain + 1,
            "seed": CHAIN_SEEDS[chain],
            "iterations": N_MCMC,
            "burn_in": BURN_IN,
            "kept_draws": N_MCMC - BURN_IN,
            "lambda_acceptance_rate": result["acceptance_rate"],
            "final_lambda_proposal_sd": result["proposal_sd"],
            "runtime_seconds": elapsed,
        })
        print(
            f"  lambda acceptance={result['acceptance_rate']:.3f}, "
            f"runtime={elapsed:.1f}s"
        )

    total_runtime = time.perf_counter() - start_time

    parameter_draws = stack_parameter_draws(chain_results)
    posterior_summary = posterior_summary_table(parameter_draws)
    diagnostics = diagnostic_table(parameter_draws)
    rhat_evolution = rhat_evolution_table(
        parameter_draws,
        method_label="Proposed GBC, narrower slab",
        burn_in=BURN_IN,
    )
    variable_selection = variable_selection_table(chain_results)
    (
        marginal_effect_draws,
        marginal_effect_summary,
        marginal_effect_diagnostics,
    ) = posterior_marginal_effect_tables(data, x, w, chain_results)
    chain_summary = pd.DataFrame(chain_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posterior_summary.to_csv(
        OUTPUT_DIR / "posterior_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    variable_selection.to_csv(
        OUTPUT_DIR / "variable_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "multichain_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rhat_evolution.to_csv(
        OUTPUT_DIR / "rhat_evolution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    marginal_effect_draws.to_csv(
        OUTPUT_DIR / "marginal_effect_posterior_draws.csv.gz",
        index=False,
        compression="gzip",
    )
    marginal_effect_summary.to_csv(
        OUTPUT_DIR / "marginal_effect_posterior_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    marginal_effect_diagnostics.to_csv(
        OUTPUT_DIR / "marginal_effect_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    chain_summary.to_csv(
        OUTPUT_DIR / "chain_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "method": (
            "Proposed Bayesian generalized Box-Cox Type II Tobit model "
            "with spike-and-slab variable selection: prior sensitivity tau1=1.00"
        ),
        "sensitivity_analysis": (
            "Only the slab SD for a one-SD covariate effect is changed "
            "from 2.50 to 1.00; all other settings match the formal model."
        ),
        "data_file": str(DATA_FILE.resolve()),
        "data_sha256": file_sha256(DATA_FILE),
        "n_total": int(len(data)),
        "n_observed_wage": int(np.sum(d)),
        "n_unobserved_wage": int(np.sum(1 - d)),
        "wage_geometric_mean": WAGE_GEOMETRIC_MEAN,
        "selection_columns": W_COLUMNS,
        "outcome_columns": X_COLUMNS,
        "chains": N_CHAINS,
        "iterations_per_chain": N_MCMC,
        "burn_in_per_chain": BURN_IN,
        "kept_draws_per_chain": N_MCMC - BURN_IN,
        "chain_seeds": CHAIN_SEEDS,
        "priors": {
            "intercept_variance": 100.0,
            "spike_sd_for_one_sd_covariate_effect": TAU0,
            "slab_sd_for_one_sd_covariate_effect": TAU1,
            "coefficient_scale_rule": (
                "Raw predictors are unchanged. Prior SD for raw coefficient j "
                "is tau / SD(predictor_j), matching the simulation prior on a "
                "one-standard-deviation covariate effect."
            ),
            "inclusion_probability": "Beta(1,1) separately by equation",
            "sigma12": "N(0,0.25), where 0.25 is the variance",
            "sigma_v2": "IG(3,2)",
            "lambda": f"Uniform({LAMBDA_LOWER},{LAMBDA_UPPER})",
        },
        "median_probability_rule": "selected when PIP > 0.5",
        "marginal_effects": (
            "Marginal effects are evaluated separately for every retained MCMC draw. "
            "Original-unit outcome effects use 40-node Gauss-Hermite integration. "
            "Posterior means and 95% equal-tail credible intervals summarize the "
            "resulting marginal-effect posterior distributions."
        ),
        "runtime_seconds": total_runtime,
    }

    with open(
        OUTPUT_DIR / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print("\nCompleted. Results:")
    for name in [
        "posterior_summary.csv",
        "variable_selection.csv",
        "multichain_diagnostics.csv",
        "rhat_evolution.csv",
        "marginal_effect_posterior_draws.csv.gz",
        "marginal_effect_posterior_summary.csv",
        "marginal_effect_diagnostics.csv",
        "chain_summary.csv",
        "run_manifest.json",
    ]:
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
