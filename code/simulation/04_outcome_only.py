"""Fit the Bickel-Doksum outcome-only benchmark in the primary simulation."""

import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[2]


N = 500
P = 10
N_REP = 500

LAMBDA_TRUE = 0.5
SIGMA12_TRUE = 0.5
SIGMA_V2_TRUE = 1.0


SIGMA2_OUTCOME_TRUE = SIGMA12_TRUE ** 2 + SIGMA_V2_TRUE

ALPHA_TRUE = np.zeros(P + 1)
ALPHA_TRUE[1:4] = [2.0, -3.0, 0.5]

BETA_TRUE = np.zeros(P + 1)
BETA_TRUE[1:4] = [1.0, -1.0, 0.5]

COMMON_DATA_DIR = REPO_ROOT / "data" / "simulation"


OUTPUT_DIR = REPO_ROOT / "results" / "simulation" / "outcome_only"


REPLICATION_LIMIT = None


LAMBDA_LOWER = 0.01
LAMBDA_UPPER = 2.0
LAMBDA_GRID_SIZE = 101


N_PREDICTIVE_DRAWS = 500
PREDICTION_MASTER_SEED = 20261015

METHOD_NAME = "bd_outcome_only_mle"
SCENARIO_ID = "baseline_valid_exclusion"
EXCLUSION_COLUMN = "w3"
OUTCOME_ONLY_COLUMN = "x3"


def bd_transform(y, lam):
    return (np.sign(y) * np.abs(y) ** lam - 1.0) / lam


def bd_inverse(t_value, lam):
    value = 1.0 + lam * t_value
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sign(value) * np.abs(value) ** (1.0 / lam)


def bd_lambda_derivative(y, lam):
    log_abs_y = np.log(np.abs(y))
    signed_power = np.sign(y) * np.exp(lam * log_abs_y)
    return (
        lam * signed_power * log_abs_y
        - (signed_power - 1.0)
    ) / lam ** 2


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_common_experiment():
    manifest_path = COMMON_DATA_DIR / "dataset_manifest.csv"
    config_path = COMMON_DATA_DIR / "simulation_config.json"

    if not manifest_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "公共数据文件夹中缺少dataset_manifest.csv或simulation_config.json。"
        )

    manifest = pd.read_csv(manifest_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_columns = {
        "replication",
        "train_seed",
        "test_seed",
        "n_train",
        "n_test",
        "train_file",
        "test_file",
        "train_sha256",
        "test_sha256",
    }
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"dataset_manifest.csv缺少列：{sorted(missing)}")

    if len(manifest) != N_REP:
        raise ValueError(f"公共数据应有{N_REP}组，实际为{len(manifest)}组。")
    if manifest["replication"].tolist() != list(range(1, N_REP + 1)):
        raise ValueError("replication必须从1连续排列到500。")

    if int(config["p_without_intercept"]) != P:
        raise ValueError("公共数据中的P与程序设置不一致。")
    if int(config["n_train_per_replication"]) != N:
        raise ValueError("公共训练集样本量与程序设置不一致。")
    if not np.allclose(config["alpha_true"], ALPHA_TRUE):
        raise ValueError("公共数据中的alpha真值不一致。")
    if not np.allclose(config["beta_true"], BETA_TRUE):
        raise ValueError("公共数据中的beta真值不一致。")
    if not np.isclose(config["lambda_true"], LAMBDA_TRUE):
        raise ValueError("公共数据中的lambda真值不一致。")
    if not np.isclose(config["sigma12_true"], SIGMA12_TRUE):
        raise ValueError("公共数据中的sigma12真值不一致。")
    if not np.isclose(config["sigma_v2_true"], SIGMA_V2_TRUE):
        raise ValueError("公共数据中的sigma_v2真值不一致。")

    
    if config.get("scenario_id") != SCENARIO_ID:
        raise ValueError("公共数据不是baseline_valid_exclusion场景。")
    exclusion = config.get("exclusion_restriction", {})
    if exclusion.get("selection_only_column") != EXCLUSION_COLUMN:
        raise ValueError("公共数据中的有效排除变量不是w3。")
    if exclusion.get("outcome_only_column") != OUTCOME_ONLY_COLUMN:
        raise ValueError("公共数据中的结果方程专属变量不是x3。")
    if exclusion.get("excluded_from_outcome_equation") is not True:
        raise ValueError("公共数据没有确认w3被排除在结果方程之外。")

    return manifest, manifest_path, config_path


def fit_beta_sigma2(y, x, lam):
    transformed_y = bd_transform(y, lam)
    beta = np.linalg.lstsq(x, transformed_y, rcond=None)[0]
    residual = transformed_y - x @ beta
    sigma2 = float(np.mean(residual ** 2))
    return beta, sigma2, residual


def profile_negative_loglikelihood(lam, y, x, log_abs_y_sum):
    if not LAMBDA_LOWER <= lam <= LAMBDA_UPPER:
        return np.inf

    _, sigma2, _ = fit_beta_sigma2(y, x, lam)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return np.inf

    n = len(y)
    return (
        0.5 * n * (math.log(2.0 * math.pi) + 1.0 + math.log(sigma2))
        - (lam - 1.0) * log_abs_y_sum
    )


def full_negative_loglikelihood(theta, y, x, log_abs_y_sum):
    beta = theta[: P + 1]
    log_sigma2 = theta[P + 1]
    lam = theta[P + 2]

    if not LAMBDA_LOWER / 2.0 < lam < LAMBDA_UPPER * 2.0:
        return np.inf

    sigma2 = math.exp(log_sigma2)
    residual = bd_transform(y, lam) - x @ beta
    n = len(y)
    return (
        0.5 * n * (math.log(2.0 * math.pi) + log_sigma2)
        + 0.5 * np.sum(residual ** 2) / sigma2
        - (lam - 1.0) * log_abs_y_sum
    )


def likelihood_gradient(theta, y, x, log_abs_y_sum):
    beta = theta[: P + 1]
    log_sigma2 = theta[P + 1]
    lam = theta[P + 2]

    sigma2 = math.exp(log_sigma2)
    transformed_y = bd_transform(y, lam)
    residual = transformed_y - x @ beta
    derivative_lambda = bd_lambda_derivative(y, lam)

    gradient_beta = -(x.T @ residual) / sigma2
    gradient_log_sigma2 = 0.5 * len(y) - 0.5 * np.sum(residual ** 2) / sigma2
    gradient_lambda = (
        residual @ derivative_lambda / sigma2 - log_abs_y_sum
    )

    return np.r_[gradient_beta, gradient_log_sigma2, gradient_lambda]


def numerical_hessian(theta, y, x, log_abs_y_sum):
    dimension = len(theta)
    hessian = np.zeros((dimension, dimension))

    for j in range(dimension):
        step = 1e-5 * (abs(theta[j]) + 1.0)
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += step
        theta_minus[j] -= step

        gradient_plus = likelihood_gradient(
            theta_plus, y, x, log_abs_y_sum
        )
        gradient_minus = likelihood_gradient(
            theta_minus, y, x, log_abs_y_sum
        )
        hessian[:, j] = (gradient_plus - gradient_minus) / (2.0 * step)

    return 0.5 * (hessian + hessian.T)


def fit_outcome_only(y, x):
    if np.any(y == 0.0):
        raise ValueError("观测结果中存在精确的0，Bickel-Doksum Jacobian无法计算。")

    log_abs_y_sum = float(np.sum(np.log(np.abs(y))))

    
    lambda_grid = np.linspace(
        LAMBDA_LOWER, LAMBDA_UPPER, LAMBDA_GRID_SIZE
    )
    objective_grid = np.array(
        [
            profile_negative_loglikelihood(
                lam, y, x, log_abs_y_sum
            )
            for lam in lambda_grid
        ]
    )
    best_index = int(np.argmin(objective_grid))
    lower_index = max(best_index - 1, 0)
    upper_index = min(best_index + 1, LAMBDA_GRID_SIZE - 1)
    lower = float(lambda_grid[lower_index])
    upper = float(lambda_grid[upper_index])

    if lower == upper:
        raise RuntimeError("lambda网格搜索未能形成有效优化区间。")

    optimization = minimize_scalar(
        profile_negative_loglikelihood,
        bounds=(lower, upper),
        args=(y, x, log_abs_y_sum),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 500},
    )
    if not optimization.success:
        raise RuntimeError(f"lambda优化失败：{optimization.message}")

    lam = float(optimization.x)
    beta, sigma2, _ = fit_beta_sigma2(y, x, lam)
    theta = np.r_[beta, math.log(sigma2), lam]

    hessian = numerical_hessian(theta, y, x, log_abs_y_sum)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    minimum_eigenvalue = float(np.min(eigenvalues))

    
    adjusted_eigenvalues = np.maximum(eigenvalues, 1e-8)
    covariance = (
        eigenvectors
        @ np.diag(1.0 / adjusted_eigenvalues)
        @ eigenvectors.T
    )
    covariance = 0.5 * (covariance + covariance.T)

    return {
        "beta": beta,
        "lambda": lam,
        "sigma2": sigma2,
        "theta": theta,
        "covariance": covariance,
        "negative_loglikelihood": float(optimization.fun),
        "minimum_hessian_eigenvalue": minimum_eigenvalue,
        "hessian_adjusted": int(minimum_eigenvalue < 1e-8),
    }


def make_parameter_table(replication, fit):
    covariance = fit["covariance"]
    rows = []

    for j in range(P + 1):
        estimate = float(fit["beta"][j])
        standard_error = math.sqrt(max(float(covariance[j, j]), 0.0))
        lower = estimate - 1.96 * standard_error
        upper = estimate + 1.96 * standard_error
        truth = float(BETA_TRUE[j])
        rows.append(
            {
                "method": METHOD_NAME,
                "replication": replication,
                "parameter": f"beta_{j}",
                "truth": truth,
                "estimate": estimate,
                "standard_error": standard_error,
                "lower_95": lower,
                "upper_95": upper,
                "bias": estimate - truth,
                "squared_error": (estimate - truth) ** 2,
                "covered_95": int(lower <= truth <= upper),
                "interval_length": upper - lower,
            }
        )

    
    lambda_estimate = float(fit["lambda"])
    lambda_se = math.sqrt(max(float(covariance[P + 2, P + 2]), 0.0))
    lambda_log_se = lambda_se / lambda_estimate
    lambda_lower = lambda_estimate * math.exp(-1.96 * lambda_log_se)
    lambda_upper = lambda_estimate * math.exp(1.96 * lambda_log_se)
    rows.append(
        {
            "method": METHOD_NAME,
            "replication": replication,
            "parameter": "lambda",
            "truth": LAMBDA_TRUE,
            "estimate": lambda_estimate,
            "standard_error": lambda_se,
            "lower_95": lambda_lower,
            "upper_95": lambda_upper,
            "bias": lambda_estimate - LAMBDA_TRUE,
            "squared_error": (lambda_estimate - LAMBDA_TRUE) ** 2,
            "covered_95": int(lambda_lower <= LAMBDA_TRUE <= lambda_upper),
            "interval_length": lambda_upper - lambda_lower,
        }
    )

    
    sigma2_estimate = float(fit["sigma2"])
    log_sigma2_se = math.sqrt(max(float(covariance[P + 1, P + 1]), 0.0))
    sigma2_se = sigma2_estimate * log_sigma2_se
    sigma2_lower = sigma2_estimate * math.exp(-1.96 * log_sigma2_se)
    sigma2_upper = sigma2_estimate * math.exp(1.96 * log_sigma2_se)
    rows.append(
        {
            "method": METHOD_NAME,
            "replication": replication,
            "parameter": "sigma2_outcome",
            "truth": SIGMA2_OUTCOME_TRUE,
            "estimate": sigma2_estimate,
            "standard_error": sigma2_se,
            "lower_95": sigma2_lower,
            "upper_95": sigma2_upper,
            "bias": sigma2_estimate - SIGMA2_OUTCOME_TRUE,
            "squared_error": (
                sigma2_estimate - SIGMA2_OUTCOME_TRUE
            ) ** 2,
            "covered_95": int(
                sigma2_lower <= SIGMA2_OUTCOME_TRUE <= sigma2_upper
            ),
            "interval_length": sigma2_upper - sigma2_lower,
        }
    )

    return pd.DataFrame(rows)


def make_not_applicable_selection_row(replication):
    row = {
        "method": METHOD_NAME,
        "replication": replication,
        "equation": "outcome",
        "applicable": 0,
    }
    for j in range(1, P + 1):
        row[f"PIP_{j}"] = np.nan
    for name in [
        "TPR",
        "FPR",
        "FDR",
        "Specificity",
        "MCC",
        "SelectedSize",
    ]:
        row[name] = np.nan
    return row


def positive_semidefinite_covariance(covariance):
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def draw_parameter_values(fit, rng):
    covariance = positive_semidefinite_covariance(fit["covariance"])
    accepted = []

    while sum(len(batch) for batch in accepted) < N_PREDICTIVE_DRAWS:
        batch = rng.multivariate_normal(
            fit["theta"],
            covariance,
            size=2 * N_PREDICTIVE_DRAWS,
        )
        lambda_values = batch[:, P + 2]
        valid = (
            np.isfinite(batch).all(axis=1)
            & (lambda_values >= LAMBDA_LOWER)
            & (lambda_values <= LAMBDA_UPPER)
        )
        accepted.append(batch[valid])

        if len(accepted) > 100:
            raise RuntimeError("无法生成足够的有效参数预测抽样。")

    return np.vstack(accepted)[:N_PREDICTIVE_DRAWS]


def make_test_predictions(
    replication,
    test_inputs,
    fit,
    prediction_seed,
    x_columns,
):
    rng = np.random.default_rng(prediction_seed)
    parameter_draws = draw_parameter_values(fit, rng)

    beta_draws = parameter_draws[:, : P + 1]
    log_sigma2_draws = parameter_draws[:, P + 1]
    lambda_draws = parameter_draws[:, P + 2]
    sigma2_draws = np.exp(log_sigma2_draws)

    x_test = test_inputs[x_columns].to_numpy(dtype=float)
    transformed_mean_draws = beta_draws @ x_test.T

    errors = rng.normal(size=transformed_mean_draws.shape)
    errors *= np.sqrt(sigma2_draws)[:, None]
    transformed_predictive_draws = transformed_mean_draws + errors
    y_predictive_draws = bd_inverse(
        transformed_predictive_draws,
        lambda_draws[:, None],
    )

    if not np.all(np.isfinite(y_predictive_draws)):
        raise FloatingPointError("原始尺度预测出现非有限值。")

    y_lower, y_upper = np.quantile(
        y_predictive_draws, [0.025, 0.975], axis=0
    )
    t_mean = np.mean(transformed_mean_draws, axis=0)
    y_mean = np.mean(y_predictive_draws, axis=0)
    number_test = len(test_inputs)

    
    
    return pd.DataFrame(
        {
            "method": METHOD_NAME,
            "replication": replication,
            "row_id": test_inputs["row_id"].to_numpy(),
            "selection_probability": np.full(number_test, np.nan),
            "t_mean_latent": t_mean,
            "t_mean_selected": t_mean,
            "selection_bias_t": np.zeros(number_test),
            "y_mean_unconditional": y_mean,
            "y_lower_unconditional": y_lower,
            "y_upper_unconditional": y_upper,
            "y_mean_selected": y_mean,
            "y_lower_selected": y_lower,
            "y_upper_selected": y_upper,
        }
    )


def calculate_rmse(truth, estimate):
    return float(
        np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2))
    )


def calculate_mae(truth, estimate):
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def evaluate_predictions(replication, predictions, test_truth):
    evaluation = predictions.merge(
        test_truth,
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(evaluation) != len(predictions):
        raise ValueError("预测结果与测试真值没有一一对应。")

    d = evaluation["d"].to_numpy(dtype=int)
    selected = d == 1
    t_star = evaluation["t_star"].to_numpy(dtype=float)
    y_star = evaluation["y_star"].to_numpy(dtype=float)

    metrics = {
        "method": METHOD_NAME,
        "replication": replication,
        "n_test": len(evaluation),
        "n_test_selected": int(np.sum(selected)),
        "test_observed_fraction": float(np.mean(d)),
        
        "selection_auc": np.nan,
        "selection_brier": np.nan,
        "selection_log_loss": np.nan,
        "t_latent_rmse": calculate_rmse(
            t_star, evaluation["t_mean_latent"]
        ),
        "t_latent_mae": calculate_mae(
            t_star, evaluation["t_mean_latent"]
        ),
        "y_latent_rmse": calculate_rmse(
            y_star, evaluation["y_mean_unconditional"]
        ),
        "y_latent_mae": calculate_mae(
            y_star, evaluation["y_mean_unconditional"]
        ),
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
        "t_selected_rmse": calculate_rmse(
            t_star[selected], evaluation.loc[selected, "t_mean_selected"]
        ),
        "t_selected_mae": calculate_mae(
            t_star[selected], evaluation.loc[selected, "t_mean_selected"]
        ),
        "y_selected_rmse": calculate_rmse(
            y_star[selected], evaluation.loc[selected, "y_mean_selected"]
        ),
        "y_selected_mae": calculate_mae(
            y_star[selected], evaluation.loc[selected, "y_mean_selected"]
        ),
        "y_selected_coverage_95": float(
            np.mean(
                (y_star[selected] >= evaluation.loc[selected, "y_lower_selected"])
                & (y_star[selected] <= evaluation.loc[selected, "y_upper_selected"])
            )
        ),
        "y_selected_interval_length": float(
            np.mean(
                evaluation.loc[selected, "y_upper_selected"].to_numpy()
                - evaluation.loc[selected, "y_lower_selected"].to_numpy()
            )
        ),
    }
    return pd.DataFrame([metrics])


def summarize_parameters(parameter_table):
    return (
        parameter_table.groupby("parameter", sort=False)
        .agg(
            Truth=("truth", "first"),
            Mean=("estimate", "mean"),
            Bias=("bias", "mean"),
            RMSE=("squared_error", lambda value: np.sqrt(np.mean(value))),
            ESD=("estimate", "std"),
            ASE=("standard_error", "mean"),
            Coverage=("covered_95", "mean"),
            AverageLength=("interval_length", "mean"),
            NumberSuccessful=("replication", "count"),
        )
        .reset_index()
    )


def summarize_predictions(metric_table):
    excluded = {
        "method",
        "replication",
        "n_test",
        "n_test_selected",
        "test_observed_fraction",
    }
    rows = []

    for metric in [column for column in metric_table.columns if column not in excluded]:
        values = metric_table[metric].dropna().to_numpy(dtype=float)
        number = len(values)
        rows.append(
            {
                "metric": metric,
                "Mean": np.mean(values) if number else np.nan,
                "MonteCarloSD": (
                    np.std(values, ddof=1) if number > 1 else np.nan
                ),
                "MonteCarloSE": (
                    np.std(values, ddof=1) / np.sqrt(number)
                    if number > 1
                    else np.nan
                ),
                "Median": np.median(values) if number else np.nan,
                "Q025": np.quantile(values, 0.025) if number else np.nan,
                "Q975": np.quantile(values, 0.975) if number else np.nan,
                "NumberSuccessful": number,
            }
        )
    return pd.DataFrame(rows)


def make_selection_summary():
    row = {
        "equation": "outcome",
        "applicable": 0,
        "NumberSuccessful": 0,
    }
    for name in [
        "TPR",
        "FPR",
        "FDR",
        "Specificity",
        "MCC",
        "SelectedSize",
    ]:
        row[name] = np.nan
    return pd.DataFrame([row])


def main():
    manifest, manifest_path, config_path = read_common_experiment()

    number_to_run = N_REP
    if REPLICATION_LIMIT is not None:
        number_to_run = min(int(REPLICATION_LIMIT), N_REP)
    manifest_to_run = manifest.iloc[:number_to_run].copy()

    x_columns = [f"x{j}" for j in range(P + 1)]

    seed_generator = np.random.default_rng(PREDICTION_MASTER_SEED)
    prediction_seeds = seed_generator.integers(
        0,
        2 ** 32 - 1,
        size=N_REP,
        dtype=np.uint32,
    )

    parameter_results = []
    selection_results = []
    prediction_results = []
    prediction_metrics_results = []
    status_results = []

    print("Settings")
    print(f"  method={METHOD_NAME}")
    print(f"  common data={COMMON_DATA_DIR}")
    print(f"  replications={number_to_run}")
    print(f"  lambda range=[{LAMBDA_LOWER}, {LAMBDA_UPPER}]")
    print(f"  predictive draws={N_PREDICTIVE_DRAWS}")
    print(f"  output={OUTPUT_DIR}")

    for _, manifest_row in manifest_to_run.iterrows():
        replication = int(manifest_row["replication"])
        print(f"Replication {replication}/{number_to_run}")

        status = {
            "method": METHOD_NAME,
            "replication": replication,
            "train_seed": int(manifest_row["train_seed"]),
            "test_seed": int(manifest_row["test_seed"]),
            "prediction_seed": int(prediction_seeds[replication - 1]),
            "n_train": int(manifest_row["n_train"]),
            "n_train_selected": np.nan,
            "n_test": int(manifest_row["n_test"]),
            "observed_fraction": np.nan,
            "lambda_estimate": np.nan,
            "minimum_hessian_eigenvalue": np.nan,
            "hessian_adjusted": np.nan,
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
                raise ValueError("训练文件SHA-256校验失败。")
            status["train_sha256_verified"] = True

            train = pd.read_csv(train_path, na_values=["NA"])
            required_train_columns = {"row_id", "d", "y_observed", *x_columns}
            missing = required_train_columns.difference(train.columns)
            if missing:
                raise ValueError(f"训练集缺少列：{sorted(missing)}")

            d_train = train["d"].to_numpy(dtype=int)
            selected_train = d_train == 1
            y_selected = train.loc[selected_train, "y_observed"].to_numpy(
                dtype=float
            )
            x_selected = train.loc[selected_train, x_columns].to_numpy(
                dtype=float
            )

            if np.any(~np.isfinite(y_selected)):
                raise ValueError("d=1子样本的y_observed含有缺失或非有限值。")
            if np.any(~np.isfinite(x_selected)):
                raise ValueError("d=1子样本的x含有缺失或非有限值。")

            status["n_train_selected"] = int(np.sum(selected_train))
            status["observed_fraction"] = float(np.mean(d_train))

            fit_start = time.perf_counter()
            fit = fit_outcome_only(y_selected, x_selected)
            status["fit_seconds"] = time.perf_counter() - fit_start
            status["lambda_estimate"] = float(fit["lambda"])
            status["minimum_hessian_eigenvalue"] = float(
                fit["minimum_hessian_eigenvalue"]
            )
            status["hessian_adjusted"] = int(fit["hessian_adjusted"])

            parameter_results.append(make_parameter_table(replication, fit))
            selection_results.append(
                make_not_applicable_selection_row(replication)
            )

            
            if sha256_file(test_path) != str(manifest_row["test_sha256"]):
                raise ValueError("测试文件SHA-256校验失败。")
            status["test_sha256_verified"] = True

            test = pd.read_csv(test_path, na_values=["NA"])

            
            test_inputs = test[["row_id", *x_columns]].copy()
            prediction_start = time.perf_counter()
            predictions = make_test_predictions(
                replication,
                test_inputs,
                fit,
                int(prediction_seeds[replication - 1]),
                x_columns,
            )
            status["prediction_seconds"] = (
                time.perf_counter() - prediction_start
            )

            
            test_truth = test[["row_id", "d", "y_star", "t_star"]].copy()
            metrics = evaluate_predictions(
                replication,
                predictions,
                test_truth,
            )

            prediction_results.append(predictions)
            prediction_metrics_results.append(metrics)

            status["total_seconds"] = (
                status["fit_seconds"] + status["prediction_seconds"]
            )
            status["success"] = True

            print(
                f"  selected={status['n_train_selected']}, "
                f"lambda={status['lambda_estimate']:.4f}, "
                f"fit={status['fit_seconds']:.3f}s"
            )

        except Exception as error:
            status["error_type"] = type(error).__name__
            status["error_message"] = str(error)
            status["traceback"] = traceback.format_exc()
            print(
                f"  FAILED: {status['error_type']}: "
                f"{status['error_message']}"
            )

        status_results.append(status)

    
    
    

    if not parameter_results:
        raise RuntimeError("所有重复均失败，没有可汇总的参数结果。")

    parameter_table = pd.concat(parameter_results, ignore_index=True)
    selection_table = pd.DataFrame(selection_results)
    prediction_table = pd.concat(prediction_results, ignore_index=True)
    prediction_metrics_table = pd.concat(
        prediction_metrics_results, ignore_index=True
    )
    status_table = pd.DataFrame(status_results)

    parameter_summary = summarize_parameters(parameter_table)
    selection_summary = make_selection_summary()
    prediction_summary = summarize_predictions(prediction_metrics_table)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parameter_table.to_csv(
        OUTPUT_DIR / "parameter_recovery_by_replication.csv",
        index=False,
    )
    parameter_summary.to_csv(
        OUTPUT_DIR / "parameter_recovery_summary.csv",
        index=False,
    )
    selection_table.to_csv(
        OUTPUT_DIR / "variable_selection_by_replication.csv",
        index=False,
    )
    selection_summary.to_csv(
        OUTPUT_DIR / "variable_selection_summary.csv",
        index=False,
    )
    prediction_table.to_csv(
        OUTPUT_DIR / "predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    prediction_metrics_table.to_csv(
        OUTPUT_DIR / "prediction_metrics_by_replication.csv",
        index=False,
    )
    prediction_summary.to_csv(
        OUTPUT_DIR / "prediction_summary.csv",
        index=False,
    )
    status_table.to_csv(
        OUTPUT_DIR / "replication_status.csv",
        index=False,
    )

    run_manifest = {
        "method": METHOD_NAME,
        "scenario_id": SCENARIO_ID,
        "valid_exclusion_variable": EXCLUSION_COLUMN,
        "outcome_only_covariate": OUTCOME_ONLY_COLUMN,
        "model": "Bickel-Doksum outcome-only regression on d=1 observations",
        "estimator": "maximum likelihood",
        "uses_selection_equation": False,
        "uses_w_covariates": False,
        "performs_variable_selection": False,
        "common_data_dir": str(COMMON_DATA_DIR),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "simulation_config_sha256": sha256_file(config_path),
        "n_replications_requested": number_to_run,
        "n_replications_successful": int(status_table["success"].sum()),
        "lambda_bounds": [LAMBDA_LOWER, LAMBDA_UPPER],
        "lambda_grid_size": LAMBDA_GRID_SIZE,
        "n_predictive_draws": N_PREDICTIVE_DRAWS,
        "prediction_master_seed": PREDICTION_MASTER_SEED,
        "selection_metrics": "not applicable",
        "variable_selection_metrics": "not applicable",
    }
    with (OUTPUT_DIR / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_manifest, file, ensure_ascii=False, indent=2)

    number_successful = int(status_table["success"].sum())
    print(
        f"\nCompleted: {number_successful}/{number_to_run} "
        "replications succeeded."
    )
    print("Main outputs:")
    for file_name in [
        "parameter_recovery_by_replication.csv",
        "parameter_recovery_summary.csv",
        "variable_selection_by_replication.csv",
        "variable_selection_summary.csv",
        "predictions.csv.gz",
        "prediction_metrics_by_replication.csv",
        "prediction_summary.csv",
        "replication_status.csv",
        "run_manifest.json",
    ]:
        print(OUTPUT_DIR / file_name)


if __name__ == "__main__":
    main()
