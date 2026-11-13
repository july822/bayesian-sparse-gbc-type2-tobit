"""Generate the common training and test data for the valid-exclusion simulation."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


N_REP = 500
N_TRAIN = 500
N_TEST = 1000
P = 10

TRAIN_MASTER_SEED = 20261008
TEST_MASTER_SEED = 20261009

LAMBDA_TRUE = 0.5
SIGMA12_TRUE = 0.5
SIGMA_V2_TRUE = 1.0

ALPHA_TRUE = np.zeros(P + 1)
ALPHA_TRUE[1:4] = [2.0, -3.0, 0.5]

BETA_TRUE = np.zeros(P + 1)
BETA_TRUE[1:4] = [1.0, -1.0, 0.5]


EXCLUSION_INDEX = 3


OUTPUT_DIR = REPO_ROOT / "data" / "simulation"


def bd_inverse(t: np.ndarray, lam: float) -> np.ndarray:

    value = 1.0 + lam * t
    return np.sign(value) * np.abs(value) ** (1.0 / lam)


def generate_dataset(seed: int, sample_size: int) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    
    x_covariates = rng.normal(size=(sample_size, P))
    x = np.column_stack((np.ones(sample_size), x_covariates))

    
    epsilon1 = rng.normal(size=sample_size)
    v = rng.normal(0.0, np.sqrt(SIGMA_V2_TRUE), size=sample_size)

    
    
    
    exclusion_variable = rng.normal(size=sample_size)

    
    w = x.copy()
    w[:, EXCLUSION_INDEX] = exclusion_variable

    
    if np.array_equal(w[:, EXCLUSION_INDEX], x[:, EXCLUSION_INDEX]):
        raise RuntimeError("排除变量w3错误地等于结果变量x3。")

    shared_indices = [j for j in range(P + 1) if j != EXCLUSION_INDEX]
    if not np.array_equal(w[:, shared_indices], x[:, shared_indices]):
        raise RuntimeError("w3以外的共享设计列发生了意外变化。")

    
    z_star = w @ ALPHA_TRUE + epsilon1
    d = (z_star > 0.0).astype(np.int8)

    
    t_star = x @ BETA_TRUE + SIGMA12_TRUE * epsilon1 + v
    y_star = bd_inverse(t_star, LAMBDA_TRUE)

    
    y_observed = np.where(d == 1, y_star, np.nan)

    data = {
        "row_id": np.arange(1, sample_size + 1, dtype=np.int32),
        "d": d,
        "y_observed": y_observed,
        "y_star": y_star,
        "t_star": t_star,
        "z_star": z_star,
        "epsilon1": epsilon1,
        "v": v,
    }

    
    for j in range(P + 1):
        data[f"x{j}"] = x[:, j]
        data[f"w{j}"] = w[:, j]

    return pd.DataFrame(data)


def save_csv_gzip(data: pd.DataFrame, path: Path) -> None:

    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=6,
            mtime=0,
        ) as gzip_file:
            with io.TextIOWrapper(
                gzip_file, encoding="utf-8", newline=""
            ) as text_file:
                data.to_csv(
                    text_file,
                    index=False,
                    float_format="%.17g",
                    na_rep="NA",
                )


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    train_dir = OUTPUT_DIR / "train"
    test_dir = OUTPUT_DIR / "test"
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    config_path = OUTPUT_DIR / "simulation_config.json"

    
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"输出文件夹已经存在且非空：{OUTPUT_DIR}\n"
            "请修改OUTPUT_DIR，或人工确认旧文件后再处理。"
        )

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_master_rng = np.random.default_rng(TRAIN_MASTER_SEED)
    train_seeds = train_master_rng.integers(
        0, 2**32 - 1, size=N_REP, dtype=np.uint32
    )

    test_master_rng = np.random.default_rng(TEST_MASTER_SEED)
    test_seeds = test_master_rng.integers(
        0, 2**32 - 1, size=N_REP, dtype=np.uint32
    )

    manifest_rows = []

    design_columns = (
        [f"x{j}" for j in range(P + 1)]
        + [f"w{j}" for j in range(P + 1)]
    )

    for rep in range(1, N_REP + 1):
        train_seed = int(train_seeds[rep - 1])
        test_seed = int(test_seeds[rep - 1])

        train_data = generate_dataset(train_seed, N_TRAIN)
        test_data = generate_dataset(test_seed, N_TEST)

        
        train_public_columns = [
            "row_id", "d", "y_observed"
        ] + design_columns
        train_public_data = train_data[train_public_columns]

        
        train_name = f"train_rep{rep:03d}.csv.gz"
        test_name = f"test_rep{rep:03d}.csv.gz"
        train_path = train_dir / train_name
        test_path = test_dir / test_name

        save_csv_gzip(train_public_data, train_path)
        save_csv_gzip(test_data, test_path)

        manifest_rows.append(
            {
                "replication": rep,
                "train_seed": train_seed,
                "test_seed": test_seed,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "train_observed_fraction": float(
                    train_data["d"].mean()
                ),
                "test_observed_fraction": float(
                    test_data["d"].mean()
                ),
                "train_file": f"train/{train_name}",
                "test_file": f"test/{test_name}",
                "train_sha256": sha256_file(train_path),
                "test_sha256": sha256_file(test_path),
            }
        )

        print(
            f"Replication {rep}/{N_REP}: "
            f"train observed={train_data['d'].mean():.3f}, "
            f"test observed={test_data['d'].mean():.3f}"
        )

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    config = {
        "scenario_id": "baseline_valid_exclusion",
        "purpose": (
            "Primary common-data baseline with one valid exclusion "
            "variable for all method comparisons."
        ),
        "n_replications": N_REP,
        "n_train_per_replication": N_TRAIN,
        "n_test_per_replication": N_TEST,
        "p_without_intercept": P,
        "train_master_seed": TRAIN_MASTER_SEED,
        "test_master_seed": TEST_MASTER_SEED,
        "alpha_true": ALPHA_TRUE.tolist(),
        "beta_true": BETA_TRUE.tolist(),
        "lambda_true": LAMBDA_TRUE,
        "sigma12_true": SIGMA12_TRUE,
        "sigma_v2_true": SIGMA_V2_TRUE,
        "covariate_distribution": "Independent standard normal",
        "selection_design_equals_outcome_design": False,
        "exclusion_restriction": {
            "selection_only_column": "w3",
            "selection_coefficient": float(ALPHA_TRUE[3]),
            "excluded_from_outcome_equation": True,
            "outcome_only_column": "x3",
            "outcome_coefficient": float(BETA_TRUE[3]),
            "shared_columns": [
                "intercept", "1", "2", "4", "5", "6",
                "7", "8", "9", "10"
            ],
        },
        "common_random_numbers_across_scenarios": {
            "same_master_seeds_as_no_exclusion_scenario": True,
            "same_x_epsilon1_v_t_star_y_star_within_replication": True,
            "quantity_changed_between_scenarios": "w3 and therefore d",
        },
        "transformation": (
            "t(y,lambda)=(sign(y)*abs(y)^lambda-1)/lambda"
        ),
        "inverse_transformation": (
            "y=sign(1+lambda*t)*abs(1+lambda*t)^(1/lambda)"
        ),
        "missing_value_in_csv": "NA",
        "column_roles": {
            "d": "Observed selection indicator",
            "y_observed": "Observed outcome; NA when d=0",
            "y_star": (
                "Latent original-scale outcome; evaluation only"
            ),
            "t_star": (
                "Latent transformed-scale outcome; evaluation only"
            ),
            "z_star": "Latent selection variable; evaluation only",
            "epsilon1": "Selection error; evaluation only",
            "v": "Independent outcome error; evaluation only",
            "x0_to_x10": "Outcome-equation design matrix",
            "w0_to_w10": "Selection-equation design matrix",
            "w3": (
                "Valid exclusion variable: enters selection equation "
                "and is excluded from outcome equation"
            ),
            "x3": (
                "Outcome-only covariate distinct from exclusion "
                "variable w3"
            ),
        },
    }

    with config_path.open("w", encoding="utf-8") as file_handle:
        json.dump(config, file_handle, ensure_ascii=False, indent=2)

    observed_fraction = pd.DataFrame(manifest_rows)[
        "train_observed_fraction"
    ]

    print("\nCompleted. Baseline data were written to:")
    print(OUTPUT_DIR)
    print(f"Training sets: {N_REP}")
    print(f"Independent test sets: {N_REP}")
    print(
        "Mean training observed fraction: "
        f"{observed_fraction.mean():.3f}"
    )
    print("Valid exclusion variable: w3")
    print(manifest_path)
    print(config_path)


if __name__ == "__main__":
    main()
