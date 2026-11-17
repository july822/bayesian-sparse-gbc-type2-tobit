"""Fit the proposed model with the income exclusion to the MEPS2001 data."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import invgamma, rankdata

REPO_ROOT = Path(__file__).resolve().parents[3]


DATA_FILE = REPO_ROOT / "data" / "processed" / "meps2001" / "meps2001_design_with_exclusion.csv"

OUTPUT_DIR = REPO_ROOT / "results" / "empirical" / "meps2001" / "proposed_income_exclusion"

N_CHAINS = 4
N_MCMC = 50000
BURN_IN = 25000
CHAIN_SEEDS = [2026102101, 2026102102, 2026102103, 2026102104]


MARGINAL_EFFECT_BATCH_SIZE = 200
MARGINAL_EFFECT_MC_SEED = 2026111202


RUN_CROSS_VALIDATION = True
CV_FOLDS = 5
CV_N_CHAINS = 4
CV_N_MCMC = 50000
CV_BURN_IN = 25000
CV_SEED = 2026111201
CV_MAX_PREDICTIVE_DRAWS = 4000
CV_FOLD_FILE = REPO_ROOT / "results" / "empirical" / "meps2001" / "iqbal_fixed_log" / "cross_validation" / "fold_assignments.csv"


TAU0 = 0.10
TAU1 = 2.50
SIGMA12_PRIOR_VARIANCE = 0.25
SIGMA_V2_PRIOR_SHAPE = 3.0
SIGMA_V2_PRIOR_SCALE = 2.0
LAMBDA_LOWER = 0.01
LAMBDA_UPPER = 5.0


AMBEXP_GEOMETRIC_MEAN = 702.795185881997


X_COLUMNS = [
    "x0",
    "x_age",
    "x_female",
    "x_educ",
    "x_blhisp",
    "x_totchr",
    "x_ins",
]

W_COLUMNS = [
    "w0",
    "w_age",
    "w_female",
    "w_educ",
    "w_blhisp",
    "w_totchr",
    "w_ins",
    "w_income",
]


def bd_transform(y, lam):
    return (np.sign(y) * np.abs(y) ** lam - 1.0) / lam


def bd_inverse(t, lam):
    value = 1.0 + lam * t
    return np.sign(value) * np.abs(value) ** (1.0 / lam)


def load_meps2001_data():
    data = pd.read_csv(DATA_FILE)

    required = [
        "row_id", "d", "ambexp", "y_observed", "y_scaled"
    ] + X_COLUMNS + W_COLUMNS

    if list(data.columns) != required:
        raise ValueError(
            "设计文件的列名或顺序已经改变。\n"
            f"应为：{required}\n"
            f"实际：{list(data.columns)}"
        )

    if len(data) != 3328:
        raise ValueError(f"样本量应为3328，实际为{len(data)}。")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id 存在重复。")

    d = data["d"].to_numpy(dtype=int)
    if not np.isin(d, [0, 1]).all():
        raise ValueError("d 必须只包含0和1。")

    if int(np.sum(d == 1)) != 2802 or int(np.sum(d == 0)) != 526:
        raise ValueError("正支出/零支出人数与冻结数据不一致。")

    selected = d == 1
    zeros = d == 0

    if data.loc[selected, "y_observed"].isna().any():
        raise ValueError("d=1时 y_observed 不应缺失。")
    if data.loc[selected, "y_scaled"].isna().any():
        raise ValueError("d=1时 y_scaled 不应缺失。")
    if (data.loc[selected, "ambexp"] <= 0).any():
        raise ValueError("d=1时 ambexp 必须严格为正。")
    if (data.loc[selected, "y_scaled"] <= 0).any():
        raise ValueError("d=1时 y_scaled 必须严格为正。")
    if data.loc[zeros, "ambexp"].ne(0).any():
        raise ValueError("d=0时 ambexp 必须为0。")
    if data.loc[zeros, "y_observed"].notna().any():
        raise ValueError("d=0时 y_observed 必须保持缺失。")
    if data.loc[zeros, "y_scaled"].notna().any():
        raise ValueError("d=0时 y_scaled 必须保持缺失，不能补成0。")

    if data[X_COLUMNS + W_COLUMNS].isna().any().any():
        raise ValueError("设计矩阵中存在缺失值。")

    expected_scaled = (
        data.loc[selected, "ambexp"].to_numpy(dtype=float)
        / AMBEXP_GEOMETRIC_MEAN
    )
    actual_scaled = data.loc[selected, "y_scaled"].to_numpy(dtype=float)
    if not np.allclose(expected_scaled, actual_scaled, rtol=0.0, atol=1e-12):
        raise ValueError("y_scaled 与冻结的支出几何均值不一致。")

    if not np.allclose(data["x0"], 1.0):
        raise ValueError("x0 必须全部等于1。")
    if not np.allclose(data["w0"], 1.0):
        raise ValueError("w0 必须全部等于1。")

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
    zeros = d == 0

    x_selected = x[selected]
    w_selected = w[selected]
    x_zero = x[zeros]
    w_zero = w[zeros]
    observed_y_selected = y_scaled[selected]
    n_zero = int(np.sum(zeros))

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
        z_star[zeros] = draw_truncated_normal(
            w_zero @ alpha,
            1.0,
            upper=0.0,
            rng=rng,
        )

        t_mean_zero = (
            x_zero @ beta
            + sigma12 * (z_star[zeros] - w_zero @ alpha)
        )
        t_star[zeros] = (
            t_mean_zero
            + rng.normal(0.0, math.sqrt(sigma_v2), size=n_zero)
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
            alpha[1:], alpha_scales, inclusion_alpha, rng
        )
        gamma_beta = update_gamma_vector(
            beta[1:], beta_scales, inclusion_beta, rng
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
        alpha_linear = w.T @ z_star - sigma12 / sigma_v2 * (w.T @ q)
        alpha = draw_normal_from_precision(
            alpha_precision, alpha_linear, rng
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
        beta_precision = xtx / sigma_v2 + np.diag(1.0 / beta_prior_variance)
        beta_linear = x.T @ adjusted_t / sigma_v2
        beta = draw_normal_from_precision(
            beta_precision, beta_linear, rng
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
        sigma12 = rng.normal(sigma12_mean, math.sqrt(sigma12_variance))

        
        v_residual = outcome_residual - sigma12 * omega
        posterior_shape = SIGMA_V2_PRIOR_SHAPE + n / 2.0
        posterior_scale = (
            SIGMA_V2_PRIOR_SCALE + 0.5 * np.sum(v_residual ** 2)
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
        transformed * np.conjugate(transformed), n=size
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


def clean_name(column):
    if column in {"x0", "w0"}:
        return "intercept"
    return column[2:]


def stack_parameter_draws(chain_results):
    parameter_draws = {}

    for j, column in enumerate(W_COLUMNS):
        parameter_draws[f"alpha_{clean_name(column)}"] = np.stack(
            [result["alpha"][:, j] for result in chain_results]
        )

    for j, column in enumerate(X_COLUMNS):
        parameter_draws[f"beta_{clean_name(column)}"] = np.stack(
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
            "method": "Proposed GBC model",
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
        [result["gamma_alpha"] for result in chain_results], axis=0
    )
    gamma_beta = np.concatenate(
        [result["gamma_beta"] for result in chain_results], axis=0
    )
    alpha = np.concatenate(
        [result["alpha"] for result in chain_results], axis=0
    )
    beta = np.concatenate(
        [result["beta"] for result in chain_results], axis=0
    )

    rows = []
    for j, column in enumerate(W_COLUMNS[1:]):
        draws = alpha[:, j + 1]
        lower, upper = np.quantile(draws, [0.025, 0.975])
        pip = float(np.mean(gamma_alpha[:, j]))
        rows.append({
            "equation": "selection",
            "variable": clean_name(column),
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
            "variable": clean_name(column),
            "pip": pip,
            "selected_pip_gt_0_5": int(pip > 0.5),
            "coefficient_mean": np.mean(draws),
            "coefficient_lower_95": lower,
            "coefficient_upper_95": upper,
        })

    return pd.DataFrame(rows)


def posterior_marginal_effect_tables(data, x, w, chain_results):
    x_index = {clean_name(c): j for j, c in enumerate(X_COLUMNS)}
    w_index = {clean_name(c): j for j, c in enumerate(W_COLUMNS)}
    continuous_selection = ["age", "educ", "totchr", "income"]
    continuous_outcome = ["age", "educ", "totchr"]
    binary = ["female", "blhisp", "ins"]
    names = (
        [f"selection_{v}" for v in continuous_selection + binary]
        + [f"outcome_{v}" for v in continuous_outcome + binary]
    )
    n_chains = len(chain_results)
    n_draws = chain_results[0]["lambda"].size
    effects = {
        name: np.empty((n_chains, n_draws), dtype=float) for name in names
    }
    rng = np.random.default_rng(MARGINAL_EFFECT_MC_SEED)

    for c, result in enumerate(chain_results):
        for start in range(0, n_draws, MARGINAL_EFFECT_BATCH_SIZE):
            stop = min(start + MARGINAL_EFFECT_BATCH_SIZE, n_draws)
            draw_slice = slice(start, stop)
            alpha = result["alpha"][draw_slice]
            beta = result["beta"][draw_slice]
            sigma12 = result["sigma12"][draw_slice]
            sigma_v2 = result["sigma_v2"][draw_slice]
            lam = result["lambda"][draw_slice]

            selection_index = w @ alpha.T
            selection_pdf = np.exp(-0.5 * selection_index ** 2) / math.sqrt(
                2.0 * math.pi
            )
            selection_probability = ndtr(selection_index)
            for variable in continuous_selection:
                j = w_index[variable]
                effects[f"selection_{variable}"][c, draw_slice] = np.mean(
                    selection_pdf * alpha[:, j][None, :], axis=0
                )
            for variable in binary:
                j = w_index[variable]
                observed_value = w[:, j]
                direction = 1.0 - 2.0 * observed_value
                alternate_index = (
                    selection_index
                    + direction[:, None] * alpha[:, j][None, :]
                )
                alternate_probability = ndtr(alternate_index)
                effects[f"selection_{variable}"][c, draw_slice] = np.mean(
                    direction[:, None]
                    * (alternate_probability - selection_probability),
                    axis=0,
                )

            total_variance = sigma12 ** 2 + sigma_v2
            predictive_error = rng.normal(size=(len(data), stop - start))
            transformed = (
                x @ beta.T
                + np.sqrt(total_variance)[None, :] * predictive_error
            )
            base_scaled = bd_inverse(transformed, lam[None, :])
            derivative_factor = np.abs(base_scaled) ** (1.0 - lam[None, :])
            for variable in continuous_outcome:
                j = x_index[variable]
                effects[f"outcome_{variable}"][c, draw_slice] = (
                    AMBEXP_GEOMETRIC_MEAN
                    * np.mean(
                        derivative_factor * beta[:, j][None, :], axis=0
                    )
                )
            for variable in binary:
                j = x_index[variable]
                observed_value = x[:, j]
                direction = 1.0 - 2.0 * observed_value
                alternate_transformed = (
                    transformed + direction[:, None] * beta[:, j][None, :]
                )
                alternate_scaled = bd_inverse(
                    alternate_transformed, lam[None, :]
                )
                effects[f"outcome_{variable}"][c, draw_slice] = (
                    AMBEXP_GEOMETRIC_MEAN
                    * np.mean(
                        direction[:, None]
                        * (alternate_scaled - base_scaled),
                        axis=0,
                    )
                )
        print(f"  marginal effects: chain {c + 1}/{n_chains} completed")

    if not all(np.isfinite(values).all() for values in effects.values()):
        raise FloatingPointError("边际效应后验样本中出现非有限值。")

    metadata = {}
    for variable in continuous_selection:
        metadata[f"selection_{variable}"] = (
            "selection", variable, "probability", "one sample SD"
        )
    for variable in binary:
        metadata[f"selection_{variable}"] = (
            "selection", variable, "probability", "0 to 1"
        )
    for variable in continuous_outcome:
        metadata[f"outcome_{variable}"] = (
            "outcome", variable,
            "latent ambulatory expenditure (dollars)", "one sample SD"
        )
    for variable in binary:
        metadata[f"outcome_{variable}"] = (
            "outcome", variable,
            "latent ambulatory expenditure (dollars)", "0 to 1"
        )

    draw_table = {
        "chain": np.repeat(np.arange(1, n_chains + 1), n_draws),
        "draw": np.tile(np.arange(1, n_draws + 1), n_chains),
    }
    rows = []
    for name in names:
        values = effects[name]
        pooled = values.reshape(-1)
        equation, variable, scale, change = metadata[name]
        rows.append({
            "effect": name,
            "equation": equation,
            "variable": variable,
            "scale": scale,
            "change": change,
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


def _load_shared_cv_folds(data, d):
    if not CV_FOLD_FILE.exists():
        raise FileNotFoundError(
            "没有找到共同五折划分文件。请先完整运行修正后的 Iqbal MEPS2001 "
            f"脚本：{CV_FOLD_FILE}"
        )
    table = pd.read_csv(CV_FOLD_FILE)
    required = {"row_id", "d", "fold"}
    if not required.issubset(table.columns) or table["row_id"].duplicated().any():
        raise ValueError("共同五折划分文件格式不正确。")
    aligned = data[["row_id"]].merge(
        table[["row_id", "d", "fold"]], on="row_id", how="left",
        validate="one_to_one",
    )
    if aligned[["d", "fold"]].isna().any().any():
        raise ValueError("共同五折划分不能覆盖全部 MEPS2001 观测。")
    if not np.array_equal(aligned["d"].to_numpy(int), d):
        raise ValueError("共同五折划分中的选择指标与当前数据不一致。")
    folds = aligned["fold"].to_numpy(int)
    if set(np.unique(folds)) != set(range(1, CV_FOLDS + 1)):
        raise ValueError("共同五折划分的折号不正确。")
    return folds, aligned


def _training_scaled_design(design, columns, train):
    output = design.copy()
    for j, column in enumerate(columns):
        variable = clean_name(column)
        if variable == "intercept" or variable in {"female", "blhisp", "ins"}:
            continue
        center = float(np.mean(design[train, j]))
        scale = float(np.std(design[train, j], ddof=1))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"交叉验证训练折中的 {variable} 没有变异。")
        output[:, j] = (design[:, j] - center) / scale
    return output


def _subset_posterior_draws(chain_results, maximum):
    per_chain = max(1, maximum // len(chain_results))
    selected = []
    for result in chain_results:
        n_draws = result["lambda"].size
        index = np.unique(np.rint(np.linspace(
            0, n_draws - 1, min(per_chain, n_draws)
        )).astype(int))
        selected.append({
            key: value[index] if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        })
    return selected


def _predict_held_out(chain_results, x, w, d, outcome_scale, rng):
    selected_results = _subset_posterior_draws(
        chain_results, CV_MAX_PREDICTIVE_DRAWS
    )
    alpha = np.concatenate([r["alpha"] for r in selected_results])
    beta = np.concatenate([r["beta"] for r in selected_results])
    sigma12 = np.concatenate([r["sigma12"] for r in selected_results])
    sigma_v2 = np.concatenate([r["sigma_v2"] for r in selected_results])
    lam = np.concatenate([r["lambda"] for r in selected_results])
    n_draws = len(lam)
    probability_sum = np.zeros(len(d))
    observed = d == 1
    predictive = np.empty((np.sum(observed), n_draws), dtype=np.float32)

    for start in range(0, n_draws, 500):
        stop = min(start + 500, n_draws)
        alpha_chunk = alpha[start:stop]
        beta_chunk = beta[start:stop]
        index = w @ alpha_chunk.T
        probability_sum += np.sum(ndtr(index), axis=1)
        observed_index = index[observed]
        lower_cdf = ndtr(-observed_index)
        uniforms = lower_cdf + (1.0 - lower_cdf) * rng.random(lower_cdf.shape)
        epsilon1 = ndtri(np.clip(uniforms, 1e-12, 1.0 - 1e-12))
        transformed = (
            x[observed] @ beta_chunk.T
            + epsilon1 * sigma12[None, start:stop]
            + rng.normal(size=epsilon1.shape)
            * np.sqrt(sigma_v2[start:stop])[None, :]
        )
        natural = outcome_scale * bd_inverse(
            transformed, lam[None, start:stop]
        )
        if not np.isfinite(natural).all():
            raise FloatingPointError("交叉验证后验预测中出现非有限值。")
        predictive[:, start:stop] = natural.astype(np.float32)

    return {
        "selection_probability": probability_sum / n_draws,
        "predictive_mean": np.mean(predictive, axis=1, dtype=np.float64),
        "predictive_lower_95": np.quantile(predictive, 0.025, axis=1),
        "predictive_upper_95": np.quantile(predictive, 0.975, axis=1),
    }


def _auc(y, probability):
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    ranks = rankdata(probability, method="average")
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def _cv_diagnostic_extrema(chain_results):
    diagnostics = diagnostic_table(stack_parameter_draws(chain_results))
    return {
        "max_rhat": diagnostics["rank_split_rhat"].max(),
        "min_bulk_ess": diagnostics["bulk_ess"].min(),
        "min_tail_ess": diagnostics["tail_ess"].min(),
    }


def cross_validated_predictive_performance(data, x, w, d):
    global N_MCMC, BURN_IN
    original_n_mcmc, original_burn_in = N_MCMC, BURN_IN
    folds, fold_assignments = _load_shared_cv_folds(data, d)
    prediction_rows, fold_rows = [], []
    raw_y = data["y_observed"].to_numpy(dtype=float)

    try:
        N_MCMC, BURN_IN = CV_N_MCMC, CV_BURN_IN
        for fold in range(1, CV_FOLDS + 1):
            train, test = folds != fold, folds == fold
            d_train, d_test = d[train], d[test]
            x_fold = _training_scaled_design(x, X_COLUMNS, train)
            w_fold = _training_scaled_design(w, W_COLUMNS, train)
            train_scale = float(np.exp(np.mean(np.log(
                raw_y[train & (d == 1)]
            ))))
            y_train = np.full(np.sum(train), np.nan)
            y_train[d_train == 1] = raw_y[train][d_train == 1] / train_scale

            fits = []
            for chain in range(CV_N_CHAINS):
                seed = CV_SEED + 10000 * fold + chain + 1
                fits.append(run_mcmc(
                    x_fold[train], w_fold[train], d_train, y_train,
                    np.random.default_rng(seed),
                ))
            fold_diagnostics = _cv_diagnostic_extrema(fits)
            prediction = _predict_held_out(
                fits, x_fold[test], w_fold[test], d_test, train_scale,
                np.random.default_rng(CV_SEED + 50000 + fold),
            )
            test_indices = np.flatnonzero(test)
            observed_indices = test_indices[d_test == 1]
            probability = prediction["selection_probability"]
            clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
            observed_y = raw_y[observed_indices]
            mean = prediction["predictive_mean"]
            lower = prediction["predictive_lower_95"]
            upper = prediction["predictive_upper_95"]
            fold_rows.append({
                "fold": fold,
                "AUC": _auc(d_test, probability),
                "Brier_score": np.mean((d_test - probability) ** 2),
                "log_loss": -np.mean(
                    d_test * np.log(clipped)
                    + (1 - d_test) * np.log1p(-clipped)
                ),
                "RMSE_d1": np.sqrt(np.mean((mean - observed_y) ** 2)),
                "MAE_d1": np.mean(np.abs(mean - observed_y)),
                "coverage_95_d1": np.mean(
                    (observed_y >= lower) & (observed_y <= upper)
                ),
                "mean_interval_length_d1": np.mean(upper - lower),
                "n_observed_outcomes": len(observed_y),
                **fold_diagnostics,
            })
            position = {index: j for j, index in enumerate(observed_indices)}
            for local, index in enumerate(test_indices):
                row = {
                    "row_id": data.iloc[index]["row_id"],
                    "fold": fold,
                    "d": int(d[index]),
                    "selection_probability": probability[local],
                    "outcome": np.nan,
                    "predictive_mean": np.nan,
                    "predictive_lower_95": np.nan,
                    "predictive_upper_95": np.nan,
                }
                if d[index] == 1:
                    j = position[index]
                    row.update({
                        "outcome": raw_y[index],
                        "predictive_mean": mean[j],
                        "predictive_lower_95": lower[j],
                        "predictive_upper_95": upper[j],
                    })
                prediction_rows.append(row)
            print(f"  cross-validation fold {fold}/{CV_FOLDS} completed")
    finally:
        N_MCMC, BURN_IN = original_n_mcmc, original_burn_in

    predictions = pd.DataFrame(prediction_rows).sort_values("row_id")
    observed = predictions["d"].eq(1)
    probability = predictions["selection_probability"].to_numpy(float)
    observed_d = predictions["d"].to_numpy(int)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    y = predictions.loc[observed, "outcome"].to_numpy(float)
    mean = predictions.loc[observed, "predictive_mean"].to_numpy(float)
    lower = predictions.loc[observed, "predictive_lower_95"].to_numpy(float)
    upper = predictions.loc[observed, "predictive_upper_95"].to_numpy(float)
    overall = pd.DataFrame([{
        "AUC": _auc(observed_d, probability),
        "Brier_score": np.mean((observed_d - probability) ** 2),
        "log_loss": -np.mean(
            observed_d * np.log(clipped)
            + (1 - observed_d) * np.log1p(-clipped)
        ),
        "RMSE_d1": np.sqrt(np.mean((mean - y) ** 2)),
        "MAE_d1": np.mean(np.abs(mean - y)),
        "coverage_95_d1": np.mean((y >= lower) & (y <= upper)),
        "mean_interval_length_d1": np.mean(upper - lower),
        "n_observed_outcomes": len(y),
    }])
    return fold_assignments, predictions, pd.DataFrame(fold_rows), overall


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
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"输出文件夹不是空的，请更换目录以免覆盖结果：{OUTPUT_DIR}"
        )
    if RUN_CROSS_VALIDATION and not CV_FOLD_FILE.exists():
        raise FileNotFoundError(
            "请先完整运行修正后的 Iqbal MEPS2001 脚本以生成共同折分："
            f"{CV_FOLD_FILE}"
        )

    data, x, w, d, y_scaled = load_meps2001_data()

    print("MEPS2001 proposed method: income exclusion specification")
    print(
        f"  n={len(data)}, positive expenditures={int(np.sum(d))}, "
        f"zero expenditures={int(np.sum(1-d))}"
    )
    print(
        f"  chains={N_CHAINS}, iterations={N_MCMC}, "
        f"burn-in={BURN_IN}"
    )
    print("  income enters the selection equation only")

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

    if RUN_CROSS_VALIDATION:
        cv_folds, cv_predictions, cv_by_fold, cv_overall = (
            cross_validated_predictive_performance(data, x, w, d)
        )
        cv_dir = OUTPUT_DIR / "cross_validation"
        cv_dir.mkdir(parents=True, exist_ok=True)
        cv_folds.to_csv(
            cv_dir / "fold_assignments.csv", index=False,
            encoding="utf-8-sig",
        )
        cv_predictions.to_csv(
            cv_dir / "out_of_fold_predictions.csv", index=False,
            encoding="utf-8-sig",
        )
        cv_by_fold.to_csv(
            cv_dir / "metrics_by_fold.csv", index=False,
            encoding="utf-8-sig",
        )
        cv_overall.to_csv(
            cv_dir / "metrics_overall.csv", index=False,
            encoding="utf-8-sig",
        )

    manifest = {
        "method": (
            "Proposed Bayesian generalized Box-Cox Type II Tobit model "
            "with spike-and-slab variable selection"
        ),
        "application": "MEPS2001 ambulatory expenditure",
        "specification": "income exclusion restriction",
        "data_file": str(DATA_FILE.resolve()),
        "data_sha256": file_sha256(DATA_FILE),
        "n_total": int(len(data)),
        "n_positive_expenditure": int(np.sum(d)),
        "n_zero_expenditure": int(np.sum(1 - d)),
        "positive_expenditure_geometric_mean": AMBEXP_GEOMETRIC_MEAN,
        "selection_columns": W_COLUMNS,
        "outcome_columns": X_COLUMNS,
        "exclusion_restriction": {
            "variable": "income",
            "role": "included in selection equation and excluded from outcome equation",
            "status": (
                "maintained identification assumption; not established by the data"
            ),
            "planned_sensitivity": (
                "rerun the same method with meps2001_design_no_exclusion.csv"
            ),
        },
        "age_coding": (
            "Source age is age in years divided by 10. The design uses a "
            "standardized age column, so continuous age associations are reported "
            "for a one-sample-SD increase."
        ),
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
                "Prior SD for coefficient j is tau / SD(predictor_j), so the "
                "spike and slab refer to a one-standard-deviation covariate effect."
            ),
            "inclusion_probability": "Beta(1,1) separately by equation",
            "sigma12": "N(0,0.25), where 0.25 is the variance",
            "sigma_v2": "IG(3,2)",
            "lambda": f"Uniform({LAMBDA_LOWER},{LAMBDA_UPPER})",
        },
        "median_probability_rule": "selected when PIP > 0.5",
        "marginal_effects": {
            "posterior_draw_level": True,
            "continuous_change": "one sample standard deviation",
            "binary_change": "zero to one",
            "outcome_scale": "latent ambulatory expenditure in dollars",
            "integration": (
                "one posterior-predictive Gaussian error draw per observation "
                "and retained MCMC draw; common random numbers are used for "
                "binary counterfactual contrasts"
            ),
            "interval": "95% equal-tail posterior credible interval",
            "interpretation": "model-based association, not a causal effect",
        },
        "cross_validation": {
            "enabled": RUN_CROSS_VALIDATION,
            "fold_file": str(CV_FOLD_FILE),
            "folds": CV_FOLDS,
            "chains_per_fold": CV_N_CHAINS,
            "iterations_per_chain": CV_N_MCMC,
            "burn_in_per_chain": CV_BURN_IN,
            "posterior_predictive_draws": CV_MAX_PREDICTIVE_DRAWS,
            "selection_metrics_population": "all held-out observations",
            "outcome_metrics_population": (
                "held-out observations with positive expenditure (d=1)"
            ),
        },
        "runtime_seconds": total_runtime,
    }

    with open(
        OUTPUT_DIR / "run_manifest.json", "w", encoding="utf-8"
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
    if RUN_CROSS_VALIDATION:
        for name in [
            "fold_assignments.csv", "out_of_fold_predictions.csv",
            "metrics_by_fold.csv", "metrics_overall.csv",
        ]:
            print(OUTPUT_DIR / "cross_validation" / name)


if __name__ == "__main__":
    main()
