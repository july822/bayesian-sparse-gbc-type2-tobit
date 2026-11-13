"""Prepare the Mroz data and construct the analysis design matrices."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


DATA_FILE = REPO_ROOT / "data" / "raw" / "TableF5-1.csv"

OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "mroz"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(table, path):
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def save_json(content, path):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_and_check_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到原始数据：{DATA_FILE}")

    data = pd.read_csv(DATA_FILE, skipinitialspace=True)

    
    data.columns = data.columns.str.strip()

    expected_columns = [
        "LFP", "WHRS", "KL6", "K618", "WA", "WE", "WW", "RPWG",
        "HHRS", "HA", "HE", "HW", "FAMINC", "MTR", "WMED", "WFED",
        "UN", "CIT", "AX",
    ]

    if data.columns.tolist() != expected_columns:
        raise ValueError(
            "原始数据列名或顺序发生变化。\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {data.columns.tolist()}"
        )

    data = data.apply(pd.to_numeric, errors="raise")

    if len(data) != 753:
        raise ValueError(f"Mroz数据应有753人，实际为{len(data)}人。")
    if data.isna().any().any():
        raise ValueError("原始CSV中出现缺失值，请先核对数据来源。")
    if not data["LFP"].isin([0, 1]).all():
        raise ValueError("LFP必须只包含0和1。")
    if int(data["LFP"].sum()) != 428:
        raise ValueError("LFP=1应有428人。")
    if int((data["LFP"] == 0).sum()) != 325:
        raise ValueError("LFP=0应有325人。")
    if not (data.loc[data["LFP"] == 1, "WW"] > 0).all():
        raise ValueError("LFP=1的工资WW必须全部为正。")
    if not (data.loc[data["LFP"] == 0, "WW"] == 0).all():
        raise ValueError("原始文件中LFP=0的WW应全部为占位值0。")
    if data.duplicated().any():
        raise ValueError("原始数据中存在完全重复的行。")

    return data


def build_clean_data(raw):
    clean = raw.copy()
    clean.insert(0, "row_id", np.arange(1, len(clean) + 1))

    selected = clean["LFP"] == 1
    geometric_mean = float(
        np.exp(np.log(clean.loc[selected, "WW"]).mean())
    )

    
    clean["wage_observed"] = clean["WW"].where(selected, np.nan)
    clean["wage_scaled"] = clean["wage_observed"] / geometric_mean

    
    clean["child"] = clean["KL6"] + clean["K618"]
    clean["age2"] = clean["WA"] ** 2
    clean["exper2"] = clean["AX"] ** 2

    if not np.isclose(geometric_mean, 3.288, atol=0.001):
        raise ValueError(
            f"正工资几何均值应约为3.288，实际为{geometric_mean:.8f}。"
        )

    return clean, geometric_mean


def build_zuehlke_design(clean):
    design = pd.DataFrame({
        "row_id": clean["row_id"],
        "d": clean["LFP"].astype(int),
        "y_observed": clean["wage_observed"],
        "y_scaled": clean["wage_scaled"],

        
        "w_intercept": 1.0,
        "w_age": clean["WA"],
        "w_age2": clean["age2"],
        "w_faminc": clean["FAMINC"],
        "w_child": clean["child"],
        "w_educ": clean["WE"],

        
        "x_intercept": 1.0,
        "x_educ": clean["WE"],
        "x_exper": clean["AX"],
        "x_exper2": clean["exper2"],
        "x_city": clean["CIT"].astype(int),
    })

    if not design.loc[design["d"] == 0, "y_observed"].isna().all():
        raise ValueError("d=0的y_observed必须全部为空。")
    if not design.loc[design["d"] == 1, "y_observed"].notna().all():
        raise ValueError("d=1的y_observed必须全部可观察。")

    return design


def build_data_summary(clean):
    variables = [
        "LFP", "wage_observed", "wage_scaled", "WA", "age2", "WE",
        "AX", "exper2", "CIT", "KL6", "K618", "child", "FAMINC",
        "HW", "MTR", "WMED", "WFED", "UN",
    ]

    groups = [
        ("All", clean),
        ("LFP=0", clean.loc[clean["LFP"] == 0]),
        ("LFP=1", clean.loc[clean["LFP"] == 1]),
    ]

    rows = []
    for group_name, group in groups:
        for variable in variables:
            values = group[variable].dropna()
            rows.append({
                "Group": group_name,
                "Variable": variable,
                "GroupN": len(group),
                "ObservedN": len(values),
                "MissingN": int(group[variable].isna().sum()),
                "Mean": values.mean() if len(values) else np.nan,
                "SD": values.std(ddof=1) if len(values) > 1 else np.nan,
                "Median": values.median() if len(values) else np.nan,
                "Minimum": values.min() if len(values) else np.nan,
                "Maximum": values.max() if len(values) else np.nan,
            })

    return pd.DataFrame(rows)


def build_variable_dictionary():
    rows = [
        ("LFP", "LFP", "selection outcome", "1 if the wife participated in the labor force in 1975; 0 otherwise", "none", "outcome", "no"),
        ("WHRS", "WHRS", "excluded from models", "wife's hours worked in 1975", "none", "no", "no"),
        ("KL6", "KL6", "candidate selection predictor", "number of children younger than 6", "none", "through child", "no"),
        ("K618", "K618", "candidate selection predictor", "number of children aged 6--18", "none", "through child", "no"),
        ("WA", "WA", "predictor", "wife's age in years", "none", "yes", "no"),
        ("WE", "WE", "predictor", "wife's years of education", "none", "yes", "yes"),
        ("WW", "WW", "outcome source", "wife's average hourly wage in 1975 dollars; zero is a missing-outcome placeholder when LFP=0", "none", "no", "outcome"),
        ("RPWG", "RPWG", "excluded from models", "wife's wage reported at the 1976 interview", "none", "no", "no"),
        ("HHRS", "HHRS", "candidate selection predictor", "husband's hours worked in 1975", "none", "no", "no"),
        ("HA", "HA", "candidate selection predictor", "husband's age in years", "none", "no", "no"),
        ("HE", "HE", "candidate selection predictor", "husband's years of education", "none", "no", "no"),
        ("HW", "HW", "candidate selection predictor", "husband's hourly wage in 1975 dollars", "none", "no", "no"),
        ("FAMINC", "FAMINC", "selection predictor", "family income in 1975 dollars", "none", "yes", "no"),
        ("MTR", "MTR", "candidate selection predictor", "wife's marginal tax rate", "none", "no", "no"),
        ("WMED", "WMED", "candidate predictor", "wife's mother's years of education", "none", "no", "no"),
        ("WFED", "WFED", "candidate predictor", "wife's father's years of education", "none", "no", "no"),
        ("UN", "UN", "candidate predictor", "county unemployment rate in percentage points", "none", "no", "no"),
        ("CIT", "CIT", "outcome predictor", "1 if living in a large city/SMSA; 0 otherwise", "none", "no", "yes"),
        ("AX", "AX", "outcome predictor", "wife's previous actual labor-market experience in years", "none", "no", "yes"),
        ("wage_observed", "WW", "observed outcome", "hourly wage observed only when LFP=1", "WW if LFP=1; missing otherwise", "no", "outcome"),
        ("wage_scaled", "WW", "transformed-model outcome", "hourly wage divided by the geometric mean among LFP=1", "wage_observed / wage_geometric_mean", "no", "outcome"),
        ("child", "KL6,K618", "selection predictor", "total number of children younger than 19", "KL6 + K618", "yes", "no"),
        ("age2", "WA", "selection predictor", "squared wife age", "WA^2", "yes", "no"),
        ("exper2", "AX", "outcome predictor", "squared previous experience", "AX^2", "no", "yes"),
    ]

    return pd.DataFrame(rows, columns=[
        "Variable", "SourceColumn", "Role", "Definition", "Construction",
        "UsedInZuehlkeSelection", "UsedInZuehlkeOutcome",
    ])


def build_scaling_parameters(clean, geometric_mean):
    return pd.DataFrame([
        ("n_total", len(clean), "number of women in the full sample"),
        ("n_selected", int(clean["LFP"].sum()), "number with LFP=1 and observed positive wage"),
        ("n_unselected", int((clean["LFP"] == 0).sum()), "number with LFP=0 and unobserved wage"),
        ("observed_fraction", clean["LFP"].mean(), "fraction with observed wage"),
        ("wage_geometric_mean", geometric_mean, "geometric mean of WW among LFP=1"),
        ("wage_scaling_rule", "WW / wage_geometric_mean", "applied only when LFP=1"),
        ("standardization", "none", "Zuehlke replication design retains original predictor units"),
    ], columns=["Parameter", "Value", "Description"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_and_check_data()
    clean, geometric_mean = build_clean_data(raw)
    design = build_zuehlke_design(clean)
    summary = build_data_summary(clean)
    dictionary = build_variable_dictionary()
    scaling = build_scaling_parameters(clean, geometric_mean)

    output_paths = {
        "mroz_clean.csv": OUTPUT_DIR / "mroz_clean.csv",
        "mroz_data_summary.csv": OUTPUT_DIR / "mroz_data_summary.csv",
        "mroz_variable_dictionary.csv": OUTPUT_DIR / "mroz_variable_dictionary.csv",
        "mroz_scaling_parameters.csv": OUTPUT_DIR / "mroz_scaling_parameters.csv",
        "mroz_zuehlke_design.csv": OUTPUT_DIR / "mroz_zuehlke_design.csv",
    }

    save_csv(clean, output_paths["mroz_clean.csv"])
    save_csv(summary, output_paths["mroz_data_summary.csv"])
    save_csv(dictionary, output_paths["mroz_variable_dictionary.csv"])
    save_csv(scaling, output_paths["mroz_scaling_parameters.csv"])
    save_csv(design, output_paths["mroz_zuehlke_design.csv"])

    manifest = {
        "dataset": "Mroz female labor-force participation data (Greene Table F5.1)",
        "purpose": "Clean data and construct the Zuehlke Greene/Mroz specification; no MCMC is run.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(DATA_FILE.resolve()),
        "source_sha256": sha256_file(DATA_FILE),
        "script_file": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "n_total": len(clean),
        "n_selected": int(clean["LFP"].sum()),
        "n_unselected": int((clean["LFP"] == 0).sum()),
        "wage_geometric_mean": geometric_mean,
        "selection_equation": [
            "w_intercept", "w_age", "w_age2", "w_faminc", "w_child", "w_educ"
        ],
        "outcome_equation": [
            "x_intercept", "x_educ", "x_exper", "x_exper2", "x_city"
        ],
        "important_rules": [
            "WW=0 when LFP=0 is treated as an unobserved wage, not a true zero wage.",
            "The wage is divided by the geometric mean among LFP=1.",
            "Predictors remain in original units for literature replication.",
            "WHRS and RPWG are not used as Zuehlke predictors.",
        ],
        "output_sha256": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
    }
    save_json(manifest, OUTPUT_DIR / "data_manifest.json")

    print("Mroz Zuehlke data preparation completed.")
    print(f"  n_total={len(clean)}")
    print(f"  n_selected={int(clean['LFP'].sum())}")
    print(f"  n_unselected={int((clean['LFP'] == 0).sum())}")
    print(f"  wage geometric mean={geometric_mean:.9f}")
    print(f"  output={OUTPUT_DIR}")
    print("\nGenerated files:")
    for name in [
        "mroz_clean.csv",
        "mroz_data_summary.csv",
        "mroz_variable_dictionary.csv",
        "mroz_scaling_parameters.csv",
        "mroz_zuehlke_design.csv",
        "data_manifest.json",
    ]:
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
