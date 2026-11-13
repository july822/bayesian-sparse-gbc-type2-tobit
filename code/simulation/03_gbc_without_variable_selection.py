"""Fit the Bayesian generalized Box-Cox model without variable selection."""

import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import invgamma, rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]


N = 500                 
P = 10                  
N_REP = 500             

N_MCMC = 10000          
BURN_IN = 5000          

LAMBDA_TRUE = 0.5       
SIGMA12_TRUE = 0.5      
SIGMA_V2_TRUE = 1.0     

TAU1 = 2.50             


COMMON_DATA_DIR = REPO_ROOT / "data" / "simulation"


OUTPUT_DIR = REPO_ROOT / "results" / "simulation" / "gbc_without_variable_selection"


REPLICATION_LIMIT = None


MCMC_MASTER_SEED = 20261013
PREDICTION_MASTER_SEED = 20261014


N_PREDICTIVE_DRAWS = 500
METHOD_NAME = "bayesian_gbc_tobit_no_variable_selection"
SCENARIO_ID = "baseline_valid_exclusion"
EXCLUSION_COLUMN = "w3"
OUTCOME_ONLY_COLUMN = "x3"


ALPHA_TRUE = np.zeros(P + 1)
ALPHA_TRUE[1:4] = [2.0, -3.0, 0.5]

BETA_TRUE = np.zeros(P + 1)
BETA_TRUE[1:4] = [1.0, -1.0, 0.5]


def bd_transform(y, lam):
    return (np.sign(y) * np.abs(y) ** lam - 1.0) / lam


def bd_inverse(t, lam):
    value = 1.0 + lam * t
    return np.sign(value) * np.abs(value) ** (1.0 / lam)


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

    
    if lam < 0.01 or lam > 5.0:
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


def run_mcmc(x, w, d, y_observed, rng):

    n, k = x.shape
    selected = d == 1
    missing = d == 0

    
    x_selected = x[selected]
    w_selected = w[selected]
    x_missing = x[missing]
    w_missing = w[missing]
    observed_y_selected = y_observed[selected]
    n_missing = np.sum(missing)

    
    xtx = x.T @ x
    wtw = w.T @ w

    
    log_abs_y_sum = np.sum(np.log(np.abs(observed_y_selected)))

    
    alpha = np.zeros(k)
    beta = np.zeros(k)
    sigma12 = 0.0
    sigma_v2 = 1.0
    lam = 1.0

    
    z_star = np.where(selected, 0.5, -0.5).astype(float)

    
    t_star = np.zeros(n)
    t_star[selected] = bd_transform(observed_y_selected, lam)

    
    n_keep = N_MCMC - BURN_IN

    alpha_save = np.zeros((n_keep, k))
    beta_save = np.zeros((n_keep, k))
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

        
        
        

        
        alpha_prior_variance = np.r_[100.0, np.repeat(TAU1 ** 2, P)]

        q = t_star - x @ beta - sigma12 * z_star

        alpha_precision = (
            (1.0 + sigma12 ** 2 / sigma_v2) * wtw
            + np.diag(1.0 / alpha_prior_variance)
        )

        alpha_linear = (
            w.T @ z_star
            - sigma12 / sigma_v2 * (w.T @ q)
        )

        alpha = draw_normal_from_precision(alpha_precision, alpha_linear, rng)

        
        
        

        omega = z_star - w @ alpha
        adjusted_t = t_star - sigma12 * omega

        beta_prior_variance = np.r_[100.0, np.repeat(TAU1 ** 2, P)]

        beta_precision = (
            xtx / sigma_v2
            + np.diag(1.0 / beta_prior_variance)
        )
        beta_linear = x.T @ adjusted_t / sigma_v2
        beta = draw_normal_from_precision(beta_precision, beta_linear, rng)

        
        
        
        

        omega = z_star - w @ alpha
        outcome_residual = t_star - x @ beta

        sigma12_variance = 1.0 / (
            np.sum(omega ** 2) / sigma_v2 + 1.0 / 0.25
        )

        sigma12_mean = sigma12_variance * (
            np.sum(omega * outcome_residual) / sigma_v2
        )

        sigma12 = rng.normal(sigma12_mean, math.sqrt(sigma12_variance))

        
        
        
        

        v_residual = outcome_residual - sigma12 * omega

        posterior_shape = 3.0 + n / 2.0
        posterior_scale = 2.0 + 0.5 * np.sum(v_residual ** 2)

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
            proposal_sd = np.clip(proposal_sd, 0.005, 1.0)
            accepted_window = 0

        
        
        

        if iteration >= BURN_IN:
            save_index = iteration - BURN_IN
            alpha_save[save_index] = alpha
            beta_save[save_index] = beta
            sigma12_save[save_index] = sigma12
            sigma_v2_save[save_index] = sigma_v2
            lambda_save[save_index] = lam

    acceptance_rate = accepted_total / N_MCMC

    return {
        "alpha": alpha_save,
        "beta": beta_save,
        "sigma12": sigma12_save,
        "sigma_v2": sigma_v2_save,
        "lambda": lambda_save,
        "acceptance_rate": acceptance_rate,
    }


def effective_sample_size(draws):

    draws = np.asarray(draws, dtype=float)
    n = len(draws)
    centered = draws - np.mean(draws)

    
    variance = np.mean(centered ** 2)
    if variance < 1e-15:
        return float(n)

    
    fft_size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=fft_size,
    )[:n] / n

    autocorrelation = autocovariance / autocovariance[0]

    
    pair_sums = []
    for lag in range(0, n - 1, 2):
        pair_sum = autocorrelation[lag] + autocorrelation[lag + 1]
        if pair_sum < 0:
            break
        pair_sums.append(pair_sum)

    if len(pair_sums) == 0:
        return float(n)

    
    for j in range(1, len(pair_sums)):
        pair_sums[j] = min(pair_sums[j], pair_sums[j - 1])

    autocorrelation_time = max(-1.0 + 2.0 * sum(pair_sums), 1.0)
    ess = n / autocorrelation_time

    return float(min(ess, n))


def one_chain_diagnostics(draws):

    draws = np.asarray(draws, dtype=float)
    posterior_sd = np.std(draws, ddof=1)
    ess = effective_sample_size(draws)
    mcse = posterior_sd / math.sqrt(ess)

    
    lag1_autocorrelation = np.corrcoef(draws[:-1], draws[1:])[0, 1]

    
    half = len(draws) // 2
    first_half_mean = np.mean(draws[:half])
    second_half_mean = np.mean(draws[-half:])
    half_mean_difference = second_half_mean - first_half_mean

    if posterior_sd > 0:
        mcse_sd_ratio = mcse / posterior_sd
        half_difference_sd = abs(half_mean_difference) / posterior_sd
    else:
        mcse_sd_ratio = 0.0
        half_difference_sd = 0.0

    return {
        "ESS": ess,
        "MCSE": mcse,
        "PosteriorSD": posterior_sd,
        "MCSE_SD_Ratio": mcse_sd_ratio,
        "Lag1Autocorrelation": lag1_autocorrelation,
        "FirstHalfMean": first_half_mean,
        "SecondHalfMean": second_half_mean,
        "HalfMeanDifference": half_mean_difference,
        "HalfDifference_SD": half_difference_sd,

        
        "PrecisionAdequate": int(
            ess >= 400 and mcse_sd_ratio <= 0.05
        ),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_safely(content, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_csv_safely(table, path, use_gzip=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if use_gzip else None
    table.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def read_common_experiment():
    manifest_path = COMMON_DATA_DIR / "dataset_manifest.csv"
    config_path = COMMON_DATA_DIR / "simulation_config.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到{manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"找不到{config_path}")

    manifest = pd.read_csv(manifest_path)
    with config_path.open("r", encoding="utf-8") as file:
        simulation_config = json.load(file)

    required_columns = {
        "replication", "train_seed", "test_seed", "n_train", "n_test",
        "train_file", "test_file", "train_sha256", "test_sha256",
    }
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"dataset_manifest.csv缺少列：{sorted(missing)}")

    if len(manifest) != N_REP:
        raise ValueError(f"公共数据应有{N_REP}组，实际为{len(manifest)}组。")

    if manifest["replication"].tolist() != list(range(1, N_REP + 1)):
        raise ValueError("replication必须从1连续排列到500。")

    
    if int(simulation_config["p_without_intercept"]) != P:
        raise ValueError("公共数据与程序的P不一致。")
    if int(simulation_config["n_train_per_replication"]) != N:
        raise ValueError("公共训练集样本量与程序的N不一致。")
    if not np.allclose(simulation_config["alpha_true"], ALPHA_TRUE):
        raise ValueError("公共数据中的alpha真值与程序不一致。")
    if not np.allclose(simulation_config["beta_true"], BETA_TRUE):
        raise ValueError("公共数据中的beta真值与程序不一致。")
    if not np.isclose(simulation_config["lambda_true"], LAMBDA_TRUE):
        raise ValueError("公共数据中的lambda真值与程序不一致。")
    if not np.isclose(simulation_config["sigma12_true"], SIGMA12_TRUE):
        raise ValueError("公共数据中的sigma12真值与程序不一致。")
    if not np.isclose(simulation_config["sigma_v2_true"], SIGMA_V2_TRUE):
        raise ValueError("公共数据中的sigma_v2真值与程序不一致。")

    
    if simulation_config.get("scenario_id") != SCENARIO_ID:
        raise ValueError("公共数据不是baseline_valid_exclusion场景。")

    exclusion = simulation_config.get("exclusion_restriction", {})
    if exclusion.get("selection_only_column") != EXCLUSION_COLUMN:
        raise ValueError("公共数据中的选择方程排除变量不是w3。")
    if exclusion.get("outcome_only_column") != OUTCOME_ONLY_COLUMN:
        raise ValueError("公共数据中的结果方程专属变量不是x3。")
    if exclusion.get("excluded_from_outcome_equation") is not True:
        raise ValueError("公共数据没有确认w3被排除在结果方程之外。")

    return manifest, manifest_path, config_path


def prepare_run_manifest(manifest_path, config_path, number_to_run):
    run_manifest = {
        "method": METHOD_NAME,
        "scenario_id": SCENARIO_ID,
        "valid_exclusion_variable": EXCLUSION_COLUMN,
        "outcome_only_covariate": OUTCOME_ONLY_COLUMN,
        "program_file": str(Path(__file__).resolve()),
        "common_data_dir": str(COMMON_DATA_DIR),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "simulation_config_sha256": sha256_file(config_path),
        "n_replications_requested": number_to_run,
        "n_mcmc": N_MCMC,
        "burn_in": BURN_IN,
        "retained_draws": N_MCMC - BURN_IN,
        "n_predictive_draws": N_PREDICTIVE_DRAWS,
        "mcmc_master_seed": MCMC_MASTER_SEED,
        "prediction_master_seed": PREDICTION_MASTER_SEED,
        "variable_selection": False,
        "ablation_of": "proposed_full",
        "slope_prior": f"N(0, {TAU1 ** 2}) for every alpha/beta slope",
        "intercept_prior": "N(0, 100) for both intercepts",
        "fitting_rule": "Only the training file is used for fitting and MCMC tuning.",
        "prediction_rule": (
            "Only row_id and x/w columns from the test file enter prediction. "
            "Test truth is used only after predictions are fixed."
        ),
    }

    path = OUTPUT_DIR / "run_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            previous = json.load(file)
        if previous != run_manifest:
            raise RuntimeError(
                "输出目录中已有不同设置的结果。请更换OUTPUT_DIR，"
                "不要把不同链长或不同数据的结果混在一起。"
            )
    else:
        write_json_safely(run_manifest, path)


def make_parameter_table(replication, result):
    posterior_draws = np.column_stack(
        (
            result["alpha"],
            result["beta"],
            result["lambda"],
            result["sigma12"],
            result["sigma_v2"],
        )
    )
    parameter_names = (
        [f"alpha_{j}" for j in range(P + 1)]
        + [f"beta_{j}" for j in range(P + 1)]
        + ["lambda", "sigma12", "sigma_v2"]
    )
    parameter_truth = np.r_[
        ALPHA_TRUE, BETA_TRUE, LAMBDA_TRUE, SIGMA12_TRUE, SIGMA_V2_TRUE
    ]

    rows = []
    for j, name in enumerate(parameter_names):
        draws = posterior_draws[:, j]
        estimate = float(np.mean(draws))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "method": METHOD_NAME,
                "replication": replication,
                "parameter": name,
                "truth": float(parameter_truth[j]),
                "estimate": estimate,
                "posterior_sd": float(np.std(draws, ddof=1)),
                "lower_95": float(lower),
                "upper_95": float(upper),
                "bias": estimate - float(parameter_truth[j]),
                "squared_error": (estimate - float(parameter_truth[j])) ** 2,
                "covered_95": int(lower <= parameter_truth[j] <= upper),
                "interval_length": float(upper - lower),
            }
        )
    return pd.DataFrame(rows)


def choose_prediction_draws(number_saved):
    number = min(N_PREDICTIVE_DRAWS, number_saved)
    return np.linspace(0, number_saved - 1, number, dtype=int)


def make_test_predictions(
    replication, test_inputs, result, prediction_seed, x_columns, w_columns
):
    rng = np.random.default_rng(prediction_seed)
    indices = choose_prediction_draws(result["lambda"].shape[0])

    alpha = result["alpha"][indices]
    beta = result["beta"][indices]
    sigma12 = result["sigma12"][indices]
    sigma_v2 = result["sigma_v2"][indices]
    lam = result["lambda"][indices]

    x = test_inputs[x_columns].to_numpy(dtype=float)
    w = test_inputs[w_columns].to_numpy(dtype=float)

    
    eta = alpha @ w.T
    xb = beta @ x.T
    selection_probability = np.mean(ndtr(eta), axis=0)

    
    log_mills = (
        -0.5 * eta ** 2
        - 0.5 * math.log(2.0 * math.pi)
        - log_ndtr(eta)
    )
    mills = np.exp(np.clip(log_mills, -700.0, 700.0))
    t_selected_mean_draws = xb + sigma12[:, None] * mills

    
    epsilon_all = rng.normal(size=xb.shape)
    v_all = rng.normal(size=xb.shape) * np.sqrt(sigma_v2)[:, None]
    t_all = xb + sigma12[:, None] * epsilon_all + v_all
    y_all = bd_inverse(t_all, lam[:, None])

    
    lower_probability = ndtr(-eta)
    uniforms = lower_probability + (
        1.0 - lower_probability
    ) * rng.random(size=eta.shape)
    uniforms = np.clip(
        uniforms, np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    epsilon_selected = ndtri(uniforms)
    v_selected = rng.normal(size=xb.shape) * np.sqrt(sigma_v2)[:, None]
    t_selected = xb + sigma12[:, None] * epsilon_selected + v_selected
    y_selected = bd_inverse(t_selected, lam[:, None])

    if not np.all(np.isfinite(y_all)):
        raise FloatingPointError("潜在总体的原始尺度预测出现非有限值。")
    if not np.all(np.isfinite(y_selected)):
        raise FloatingPointError("已选择样本的原始尺度预测出现非有限值。")

    y_all_lower, y_all_upper = np.quantile(y_all, [0.025, 0.975], axis=0)
    y_selected_lower, y_selected_upper = np.quantile(
        y_selected, [0.025, 0.975], axis=0
    )
    t_latent_mean = np.mean(xb, axis=0)
    t_selected_mean = np.mean(t_selected_mean_draws, axis=0)

    return pd.DataFrame(
        {
            "method": METHOD_NAME,
            "replication": replication,
            "row_id": test_inputs["row_id"].to_numpy(),
            "selection_probability": selection_probability,
            "t_mean_latent": t_latent_mean,
            "t_mean_selected": t_selected_mean,
            "selection_bias_t": t_selected_mean - t_latent_mean,
            "y_mean_unconditional": np.mean(y_all, axis=0),
            "y_lower_unconditional": y_all_lower,
            "y_upper_unconditional": y_all_upper,
            "y_mean_selected": np.mean(y_selected, axis=0),
            "y_lower_selected": y_selected_lower,
            "y_upper_selected": y_selected_upper,
        }
    )


def calculate_auc(truth, probability):
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=float)
    number_positive = int(np.sum(truth == 1))
    number_negative = int(np.sum(truth == 0))
    if number_positive == 0 or number_negative == 0:
        return np.nan
    ranks = rankdata(probability, method="average")
    positive_rank_sum = np.sum(ranks[truth == 1])
    return (
        positive_rank_sum - number_positive * (number_positive + 1) / 2.0
    ) / (number_positive * number_negative)


def calculate_rmse(truth, estimate):
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2)))


def calculate_mae(truth, estimate):
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def evaluate_predictions(replication, predictions, test_truth):
    evaluation = predictions.merge(
        test_truth, on="row_id", how="inner", validate="one_to_one"
    )
    if len(evaluation) != len(predictions):
        raise ValueError("预测结果与测试真值没有一一对应。")

    d = evaluation["d"].to_numpy(dtype=int)
    probability = np.clip(
        evaluation["selection_probability"].to_numpy(dtype=float),
        1e-12,
        1.0 - 1e-12,
    )
    t_star = evaluation["t_star"].to_numpy(dtype=float)
    y_star = evaluation["y_star"].to_numpy(dtype=float)
    selected = d == 1

    metrics = {
        "method": METHOD_NAME,
        "replication": replication,
        "n_test": len(evaluation),
        "n_test_selected": int(np.sum(selected)),
        "test_observed_fraction": float(np.mean(d)),
        "selection_auc": float(calculate_auc(d, probability)),
        "selection_brier": float(np.mean((probability - d) ** 2)),
        "selection_log_loss": float(
            -np.mean(d * np.log(probability) + (1 - d) * np.log(1.0 - probability))
        ),
        "t_latent_rmse": calculate_rmse(t_star, evaluation["t_mean_latent"]),
        "t_latent_mae": calculate_mae(t_star, evaluation["t_mean_latent"]),
        "y_latent_rmse": calculate_rmse(y_star, evaluation["y_mean_unconditional"]),
        "y_latent_mae": calculate_mae(y_star, evaluation["y_mean_unconditional"]),
        "y_latent_coverage_95": float(
            np.mean(
                (y_star >= evaluation["y_lower_unconditional"].to_numpy())
                & (y_star <= evaluation["y_upper_unconditional"].to_numpy())
            )
        ),
        "y_latent_interval_length": float(
            np.mean(
                evaluation["y_upper_unconditional"].to_numpy()
                - evaluation["y_lower_unconditional"].to_numpy()
            )
        ),
    }

    if np.any(selected):
        selected_evaluation = evaluation.loc[selected]
        metrics.update(
            {
                "t_selected_rmse": calculate_rmse(
                    t_star[selected], selected_evaluation["t_mean_selected"]
                ),
                "t_selected_mae": calculate_mae(
                    t_star[selected], selected_evaluation["t_mean_selected"]
                ),
                "y_selected_rmse": calculate_rmse(
                    y_star[selected], selected_evaluation["y_mean_selected"]
                ),
                "y_selected_mae": calculate_mae(
                    y_star[selected], selected_evaluation["y_mean_selected"]
                ),
                "y_selected_coverage_95": float(
                    np.mean(
                        (y_star[selected] >= selected_evaluation["y_lower_selected"])
                        & (y_star[selected] <= selected_evaluation["y_upper_selected"])
                    )
                ),
                "y_selected_interval_length": float(
                    np.mean(
                        selected_evaluation["y_upper_selected"].to_numpy()
                        - selected_evaluation["y_lower_selected"].to_numpy()
                    )
                ),
            }
        )
    else:
        for name in [
            "t_selected_rmse", "t_selected_mae", "y_selected_rmse",
            "y_selected_mae", "y_selected_coverage_95",
            "y_selected_interval_length",
        ]:
            metrics[name] = np.nan

    return pd.DataFrame([metrics])


def checkpoint_complete(replication_folder):
    status_path = replication_folder / "status.json"
    required_paths = [
        replication_folder / "parameter_recovery.csv",
        replication_folder / "predictions.csv.gz",
        replication_folder / "prediction_metrics.csv",
        status_path,
    ]
    if not all(path.exists() for path in required_paths):
        return False
    try:
        with status_path.open("r", encoding="utf-8") as file:
            return bool(json.load(file).get("success", False))
    except (OSError, json.JSONDecodeError):
        return False


def run_one_replication(
    manifest_row, mcmc_seed, prediction_seed, x_columns, w_columns
):
    replication = int(manifest_row["replication"])
    replication_folder = OUTPUT_DIR / "_checkpoints" / f"rep{replication:03d}"
    replication_folder.mkdir(parents=True, exist_ok=True)

    if checkpoint_complete(replication_folder):
        print(f"Replication {replication}/{N_REP}: 已完成，跳过")
        return

    status = {
        "method": METHOD_NAME,
        "replication": replication,
        "train_seed": int(manifest_row["train_seed"]),
        "test_seed": int(manifest_row["test_seed"]),
        "mcmc_seed": int(mcmc_seed),
        "prediction_seed": int(prediction_seed),
        "n_train": int(manifest_row["n_train"]),
        "n_test": int(manifest_row["n_test"]),
        "observed_fraction": np.nan,
        "lambda_acceptance": np.nan,
        "fit_seconds": np.nan,
        "prediction_seconds": np.nan,
        "total_seconds": np.nan,
        "train_sha256_verified": False,
        "test_sha256_verified": False,
        "success": False,
        "error_type": "",
        "error_message": "",
    }

    try:
        train_path = COMMON_DATA_DIR / str(manifest_row["train_file"])
        test_path = COMMON_DATA_DIR / str(manifest_row["test_file"])

        if sha256_file(train_path) != str(manifest_row["train_sha256"]):
            raise ValueError(f"训练文件SHA-256不一致：{train_path}")
        status["train_sha256_verified"] = True

        train = pd.read_csv(train_path, na_values=["NA"])
        required_train_columns = {"row_id", "d", "y_observed", *x_columns, *w_columns}
        missing = required_train_columns.difference(train.columns)
        if missing:
            raise ValueError(f"训练集缺少列：{sorted(missing)}")
        if len(train) != int(manifest_row["n_train"]):
            raise ValueError("训练集行数与manifest不一致。")

        x = train[x_columns].to_numpy(dtype=float)
        w = train[w_columns].to_numpy(dtype=float)
        d = train["d"].to_numpy(dtype=int)
        y_observed = train["y_observed"].to_numpy(dtype=float)

        if not np.array_equal(np.isnan(y_observed), d == 0):
            raise ValueError("y_observed的缺失位置与d=0不一致。")

        status["observed_fraction"] = float(np.mean(d))

        
        rng = np.random.default_rng(mcmc_seed)
        fit_start = time.perf_counter()
        result = run_mcmc(x, w, d, y_observed, rng)
        status["fit_seconds"] = time.perf_counter() - fit_start
        status["lambda_acceptance"] = float(result["acceptance_rate"])

        parameter_table = make_parameter_table(replication, result)

        
        if sha256_file(test_path) != str(manifest_row["test_sha256"]):
            raise ValueError(f"测试文件SHA-256不一致：{test_path}")
        status["test_sha256_verified"] = True

        test = pd.read_csv(test_path, na_values=["NA"])
        if len(test) != int(manifest_row["n_test"]):
            raise ValueError("测试集行数与manifest不一致。")

        
        test_inputs = test[["row_id", *x_columns, *w_columns]].copy()
        prediction_start = time.perf_counter()
        prediction_table = make_test_predictions(
            replication, test_inputs, result, prediction_seed, x_columns, w_columns
        )
        status["prediction_seconds"] = time.perf_counter() - prediction_start

        
        test_truth = test[["row_id", "d", "y_star", "t_star"]].copy()
        prediction_metrics = evaluate_predictions(
            replication, prediction_table, test_truth
        )

        status["total_seconds"] = status["fit_seconds"] + status["prediction_seconds"]
        status["success"] = True

        write_csv_safely(parameter_table, replication_folder / "parameter_recovery.csv")
        write_csv_safely(
            prediction_table,
            replication_folder / "predictions.csv.gz",
            use_gzip=True,
        )
        write_csv_safely(
            prediction_metrics, replication_folder / "prediction_metrics.csv"
        )
        
        write_json_safely(status, replication_folder / "status.json")

        print(
            f"Replication {replication}/{N_REP}: "
            f"observed={status['observed_fraction']:.3f}, "
            f"lambda acceptance={status['lambda_acceptance']:.3f}, "
            f"fit={status['fit_seconds']:.1f}s"
        )

    except Exception as error:
        status["error_type"] = type(error).__name__
        status["error_message"] = str(error)
        status["traceback"] = traceback.format_exc()
        write_json_safely(status, replication_folder / "status.json")
        print(
            f"Replication {replication}/{N_REP}: 失败；"
            f"{status['error_type']}: {status['error_message']}"
        )


def collect_status(replications):
    rows = []
    for replication in replications:
        status_path = (
            OUTPUT_DIR / "_checkpoints" / f"rep{replication:03d}" / "status.json"
        )
        if status_path.exists():
            with status_path.open("r", encoding="utf-8") as file:
                rows.append(json.load(file))
        else:
            rows.append(
                {
                    "method": METHOD_NAME,
                    "replication": replication,
                    "success": False,
                    "error_type": "NotRun",
                    "error_message": "No checkpoint was found.",
                }
            )
    return pd.DataFrame(rows)


def combine_tables(successful_replications, file_name):
    tables = []
    for replication in successful_replications:
        path = OUTPUT_DIR / "_checkpoints" / f"rep{replication:03d}" / file_name
        tables.append(pd.read_csv(path))
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def summarize_parameters(parameter_table):
    if parameter_table.empty:
        return pd.DataFrame()
    return (
        parameter_table.groupby("parameter", sort=False)
        .agg(
            Truth=("truth", "first"),
            Mean=("estimate", "mean"),
            Bias=("bias", "mean"),
            RMSE=("squared_error", lambda value: np.sqrt(np.mean(value))),
            ESD=("estimate", "std"),
            ASD=("posterior_sd", "mean"),
            Coverage=("covered_95", "mean"),
            AverageLength=("interval_length", "mean"),
            NumberSuccessful=("replication", "count"),
        )
        .reset_index()
    )


def summarize_prediction_metrics(metric_table):
    if metric_table.empty:
        return pd.DataFrame()

    excluded = {
        "method", "replication", "n_test",
        "n_test_selected", "test_observed_fraction",
    }
    metric_names = [column for column in metric_table.columns if column not in excluded]

    rows = []
    for metric in metric_names:
        values = metric_table[metric].dropna().to_numpy(dtype=float)
        number = len(values)
        rows.append(
            {
                "metric": metric,
                "Mean": np.mean(values) if number else np.nan,
                "MonteCarloSD": np.std(values, ddof=1) if number > 1 else np.nan,
                "MonteCarloSE": (
                    np.std(values, ddof=1) / np.sqrt(number)
                    if number > 1 else np.nan
                ),
                "Median": np.median(values) if number else np.nan,
                "Q025": np.quantile(values, 0.025) if number else np.nan,
                "Q975": np.quantile(values, 0.975) if number else np.nan,
                "NumberSuccessful": number,
            }
        )
    return pd.DataFrame(rows)


def consolidate_outputs(replications):
    status_table = collect_status(replications)
    write_csv_safely(status_table, OUTPUT_DIR / "replication_status.csv")

    successful = (
        status_table.loc[
            status_table["success"].fillna(False).astype(bool), "replication"
        ]
        .astype(int)
        .tolist()
    )

    parameter_table = combine_tables(successful, "parameter_recovery.csv")
    prediction_metric_table = combine_tables(successful, "prediction_metrics.csv")
    prediction_table = combine_tables(successful, "predictions.csv.gz")

    if not parameter_table.empty:
        write_csv_safely(
            parameter_table, OUTPUT_DIR / "parameter_recovery_by_replication.csv"
        )
        write_csv_safely(
            summarize_parameters(parameter_table),
            OUTPUT_DIR / "parameter_recovery_summary.csv",
        )

    if not prediction_metric_table.empty:
        write_csv_safely(
            prediction_metric_table,
            OUTPUT_DIR / "prediction_metrics_by_replication.csv",
        )
        write_csv_safely(
            summarize_prediction_metrics(prediction_metric_table),
            OUTPUT_DIR / "prediction_summary.csv",
        )

    if not prediction_table.empty:
        write_csv_safely(
            prediction_table,
            OUTPUT_DIR / "predictions.csv.gz",
            use_gzip=True,
        )

    print(
        f"\n汇总完成：成功{len(successful)}/{len(replications)}，"
        f"失败或未完成{len(replications) - len(successful)}。"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path, config_path = read_common_experiment()

    number_to_run = N_REP
    if REPLICATION_LIMIT is not None:
        number_to_run = min(int(REPLICATION_LIMIT), N_REP)
    manifest_to_run = manifest.iloc[:number_to_run].copy()

    prepare_run_manifest(manifest_path, config_path, number_to_run)

    x_columns = [f"x{j}" for j in range(P + 1)]
    w_columns = [f"w{j}" for j in range(P + 1)]

    
    mcmc_seed_generator = np.random.default_rng(MCMC_MASTER_SEED)
    prediction_seed_generator = np.random.default_rng(PREDICTION_MASTER_SEED)
    mcmc_seeds = mcmc_seed_generator.integers(
        0, 2 ** 32 - 1, size=N_REP, dtype=np.uint32
    )
    prediction_seeds = prediction_seed_generator.integers(
        0, 2 ** 32 - 1, size=N_REP, dtype=np.uint32
    )

    print("Settings")
    print(f"  method={METHOD_NAME}")
    print(f"  common data={COMMON_DATA_DIR}")
    print(f"  replications={number_to_run}")
    print(f"  MCMC={N_MCMC}, burn-in={BURN_IN}")
    print(f"  predictive draws={N_PREDICTIVE_DRAWS}")
    print(f"  output={OUTPUT_DIR}")

    for _, row in manifest_to_run.iterrows():
        replication = int(row["replication"])
        run_one_replication(
            manifest_row=row,
            mcmc_seed=int(mcmc_seeds[replication - 1]),
            prediction_seed=int(prediction_seeds[replication - 1]),
            x_columns=x_columns,
            w_columns=w_columns,
        )

    replications = manifest_to_run["replication"].astype(int).tolist()
    consolidate_outputs(replications)

    print("\n主要输出文件：")
    for file_name in [
        "parameter_recovery_by_replication.csv",
        "prediction_metrics_by_replication.csv",
        "predictions.csv.gz",
        "replication_status.csv",
        "parameter_recovery_summary.csv",
        "prediction_summary.csv",
        "run_manifest.json",
    ]:
        print(OUTPUT_DIR / file_name)


if __name__ == "__main__":
    main()
