"""Run the same-covariate sample-size experiment for n = 500."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import invgamma

REPO_ROOT = Path(__file__).resolve().parents[2]


N = 500                 
P = 10                  
N_REP = 500             

N_MCMC = 10000          
BURN_IN = 5000          

LAMBDA_TRUE = 0.5       
SIGMA12_TRUE = 0.5      
SIGMA_V2_TRUE = 1.0     

TAU0 = 0.10             
TAU1 = 2.50             

SEED = 20261008         
OUTPUT_DIR = REPO_ROOT / "results" / "sensitivity" / "sample_size_n500"


ALPHA_TRUE = np.zeros(P + 1)
ALPHA_TRUE[1:4] = [2.0, -3.0, 0.5]

BETA_TRUE = np.zeros(P + 1)
BETA_TRUE[1:4] = [1.0, -1.0, 0.5]


def bd_transform(y, lam):
    return (np.sign(y) * np.abs(y) ** lam - 1.0) / lam


def bd_inverse(t, lam):
    value = 1.0 + lam * t
    return np.sign(value) * np.abs(value) ** (1.0 / lam)


def generate_data(rng):

    
    covariates = rng.normal(size=(N, P))

    
    x = np.column_stack((np.ones(N), covariates))

    
    w = x.copy()

    
    epsilon1 = rng.normal(size=N)

    
    v = rng.normal(0.0, math.sqrt(SIGMA_V2_TRUE), size=N)

    
    z_star = w @ ALPHA_TRUE + epsilon1

    
    d = (z_star > 0).astype(int)

    
    t_star = x @ BETA_TRUE + SIGMA12_TRUE * epsilon1 + v

    
    y_star = bd_inverse(t_star, LAMBDA_TRUE)

    
    y_observed = np.where(d == 1, y_star, np.nan)

    return x, w, d, y_observed


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


def update_gamma_vector(coefficients, inclusion_rate, rng):

    
    safe_p = np.clip(inclusion_rate, 1e-12, 1.0 - 1e-12)

    log_slab = (
        np.log(safe_p)
        - np.log(TAU1)
        - 0.5 * (coefficients / TAU1) ** 2
    )

    log_spike = (
        np.log1p(-safe_p)
        - np.log(TAU0)
        - 0.5 * (coefficients / TAU0) ** 2
    )

    
    log_difference = np.clip(log_spike - log_slab, -700.0, 700.0)
    inclusion_probability = 1.0 / (1.0 + np.exp(log_difference))

    
    return (rng.random(coefficients.size) < inclusion_probability).astype(int)


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

    
    gamma_alpha = np.ones(P, dtype=int)
    gamma_beta = np.ones(P, dtype=int)
    p_alpha = 0.5
    p_beta = 0.5

    
    z_star = np.where(selected, 0.5, -0.5).astype(float)

    
    t_star = np.zeros(n)
    t_star[selected] = bd_transform(observed_y_selected, lam)

    
    n_keep = N_MCMC - BURN_IN

    alpha_save = np.zeros((n_keep, k))
    beta_save = np.zeros((n_keep, k))
    gamma_alpha_save = np.zeros((n_keep, P))
    gamma_beta_save = np.zeros((n_keep, P))
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

        
        
        

        p_alpha = rng.beta(1 + np.sum(gamma_alpha), 1 + P - np.sum(gamma_alpha))
        p_beta = rng.beta(1 + np.sum(gamma_beta), 1 + P - np.sum(gamma_beta))

        
        
        

        gamma_alpha = update_gamma_vector(alpha[1:], p_alpha, rng)
        gamma_beta = update_gamma_vector(beta[1:], p_beta, rng)

        
        
        

        
        alpha_prior_variance = np.r_[
            100.0,
            np.where(gamma_alpha == 1, TAU1 ** 2, TAU0 ** 2),
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

        alpha = draw_normal_from_precision(alpha_precision, alpha_linear, rng)

        
        
        

        omega = z_star - w @ alpha
        adjusted_t = t_star - sigma12 * omega

        beta_prior_variance = np.r_[
            100.0,
            np.where(gamma_beta == 1, TAU1 ** 2, TAU0 ** 2),
        ]

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
            gamma_alpha_save[save_index] = gamma_alpha
            gamma_beta_save[save_index] = gamma_beta
            sigma12_save[save_index] = sigma12
            sigma_v2_save[save_index] = sigma_v2
            lambda_save[save_index] = lam

    acceptance_rate = accepted_total / N_MCMC

    return {
        "alpha": alpha_save,
        "beta": beta_save,
        "gamma_alpha": gamma_alpha_save,
        "gamma_beta": gamma_beta_save,
        "sigma12": sigma12_save,
        "sigma_v2": sigma_v2_save,
        "lambda": lambda_save,
        "acceptance_rate": acceptance_rate,
    }


def selection_metrics(pip):

    truth = np.zeros(P, dtype=int)
    truth[:3] = 1
    chosen = (pip >= 0.5).astype(int)

    tp = np.sum((chosen == 1) & (truth == 1))
    fp = np.sum((chosen == 1) & (truth == 0))
    tn = np.sum((chosen == 0) & (truth == 0))
    fn = np.sum((chosen == 0) & (truth == 1))

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fdr = fp / (tp + fp) if tp + fp > 0 else 0.0
    specificity = tn / (tn + fp)

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else np.nan

    return {
        "TPR": tpr,
        "FPR": fpr,
        "FDR": fdr,
        "Specificity": specificity,
        "MCC": mcc,
        "SelectedSize": np.sum(chosen),
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


def main():
    rng_master = np.random.default_rng(SEED)

    parameter_results = []
    selection_results = []
    diagnostic_results = []

    
    for rep in range(N_REP):
        print(f"Replication {rep + 1}/{N_REP}")

        
        rep_seed = rng_master.integers(0, 2 ** 32 - 1)
        rng = np.random.default_rng(rep_seed)

        
        x, w, d, y_observed = generate_data(rng)
        result = run_mcmc(x, w, d, y_observed, rng)

        print(
            f"  observed fraction={np.mean(d):.3f}, "
            f"lambda acceptance={result['acceptance_rate']:.3f}"
        )

        
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
            ALPHA_TRUE,
            BETA_TRUE,
            LAMBDA_TRUE,
            SIGMA12_TRUE,
            SIGMA_V2_TRUE,
        ]

        
        for j, name in enumerate(parameter_names):
            draws = posterior_draws[:, j]
            estimate = np.mean(draws)
            lower, upper = np.quantile(draws, [0.025, 0.975])

            parameter_results.append(
                {
                    "replication": rep + 1,
                    "parameter": name,
                    "truth": parameter_truth[j],
                    "estimate": estimate,
                    "posterior_sd": np.std(draws, ddof=1),
                    "bias": estimate - parameter_truth[j],
                    "squared_error": (estimate - parameter_truth[j]) ** 2,
                    "covered": int(lower <= parameter_truth[j] <= upper),
                    "interval_length": upper - lower,
                }
            )

            
            diagnostic_row = {
                "replication": rep + 1,
                "parameter": name,
                "observed_fraction": np.mean(d),
                "lambda_acceptance": result["acceptance_rate"],
            }
            diagnostic_row.update(one_chain_diagnostics(draws))
            diagnostic_results.append(diagnostic_row)

        
        pip_alpha = np.mean(result["gamma_alpha"], axis=0)
        pip_beta = np.mean(result["gamma_beta"], axis=0)

        for equation, pip in [("selection", pip_alpha), ("outcome", pip_beta)]:
            row = {
                "replication": rep + 1,
                "equation": equation,
            }
            row.update(selection_metrics(pip))
            selection_results.append(row)

    
    
    

    parameter_results = pd.DataFrame(parameter_results)
    selection_results = pd.DataFrame(selection_results)
    diagnostic_results = pd.DataFrame(diagnostic_results)

    parameter_summary = (
        parameter_results.groupby("parameter", sort=False)
        .agg(
            Truth=("truth", "first"),
            Mean=("estimate", "mean"),
            Bias=("bias", "mean"),
            RMSE=("squared_error", lambda value: np.sqrt(np.mean(value))),
            ESD=("estimate", "std"),
            ASD=("posterior_sd", "mean"),
            Coverage=("covered", "mean"),
            AverageLength=("interval_length", "mean"),
        )
        .reset_index()
    )

    selection_summary = (
        selection_results.groupby("equation", sort=False)[
            ["TPR", "FPR", "FDR", "Specificity", "MCC", "SelectedSize"]
        ]
        .mean()
        .reset_index()
    )

    
    diagnostic_summary = (
        diagnostic_results.groupby("parameter", sort=False)
        .agg(
            NumberOfDatasets=("replication", "count"),
            MedianESS=("ESS", "median"),
            ESS_5_Percentile=("ESS", lambda value: np.quantile(value, 0.05)),
            MinimumESS=("ESS", "min"),
            MedianMCSE_SD_Ratio=("MCSE_SD_Ratio", "median"),
            MCSE_SD_Ratio_95_Percentile=(
                "MCSE_SD_Ratio",
                lambda value: np.quantile(value, 0.95),
            ),
            MedianLag1Autocorrelation=("Lag1Autocorrelation", "median"),
            HalfDifference_SD_95_Percentile=(
                "HalfDifference_SD",
                lambda value: np.quantile(value, 0.95),
            ),
            PrecisionAdequateRate=("PrecisionAdequate", "mean"),
        )
        .reset_index()
    )

    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parameter_summary.to_csv(
        OUTPUT_DIR / "parameter_recovery_summary.csv",
        index=False,
    )

    selection_summary.to_csv(
        OUTPUT_DIR / "variable_selection_summary.csv",
        index=False,
    )

    diagnostic_results.to_csv(
        OUTPUT_DIR / "single_chain_diagnostics.csv",
        index=False,
    )

    diagnostic_summary.to_csv(
        OUTPUT_DIR / "single_chain_diagnostics_summary.csv",
        index=False,
    )

    print("\nCompleted. Four files were generated:")
    print(OUTPUT_DIR / "parameter_recovery_summary.csv")
    print(OUTPUT_DIR / "variable_selection_summary.csv")
    print(OUTPUT_DIR / "single_chain_diagnostics.csv")
    print(OUTPUT_DIR / "single_chain_diagnostics_summary.csv")


if __name__ == "__main__":
    main()
