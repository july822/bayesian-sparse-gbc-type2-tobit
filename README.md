# Bayesian sparse generalized Box--Cox Type II Tobit model

This repository contains the code used to generate the simulation data, fit the five methods in the Monte Carlo comparison, reproduce the Mroz and MEPS2001 empirical analyses, and construct the manuscript figures.

The repository contains source code only. Generated data, posterior draws, numerical results, and figures are written locally when the scripts are run and are not tracked by Git.

## Repository structure

```text
code/
  simulation/          Simulation data generation, five-method comparison,
                       and sensitivity analyses
  empirical/mroz/      Mroz data preparation and model fits
  empirical/meps2001/  MEPS2001 data preparation and model fits
  figures/             Manuscript figure generation and layout checks
data/
  raw/                 User-supplied source data; not tracked by Git
external/              Instructions for obtaining external benchmark code
```

All paths are resolved relative to the repository root. No script requires a machine-specific absolute path.

## Software requirements

- Python 3.10 or later
- R 4.2 or later
- Python packages listed in `requirements.txt`
- R packages listed in `R-packages.txt`
- The Iqbal et al. reference implementation described in `external/README.md`

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install the required R packages from an R session with:

```r
install.packages(c(
  "digest", "jsonlite", "MASS", "matrixStats", "posterior",
  "sampleSelection", "ssmrob", "truncnorm"
))
```

## Simulation study

Run the scripts from the repository root in the following order.

```bash
python code/simulation/01_generate_data.py
python code/simulation/02_proposed_method.py
python code/simulation/03_gbc_without_variable_selection.py
python code/simulation/04_outcome_only.py
Rscript code/simulation/05_heckman_two_step.R
Rscript code/simulation/06_iqbal_spike_slab.R
```

The five fitted methods use the same generated training and test samples. Output is written under `results/simulation/`.

The supplementary sample-size and chain-length analyses are reproduced with:

```bash
python code/simulation/07_sample_size_n500.py
python code/simulation/08_sample_size_n1000.py
python code/simulation/09_sample_size_n2000.py
python code/simulation/10_chain_length_10000.py
python code/simulation/11_chain_length_50000.py
```

## Mroz application

Place the source file `TableF5-1.csv` in `data/raw/`, then run:

```bash
python code/empirical/mroz/01_prepare_mroz_data.py
python code/empirical/mroz/02_proposed_method.py
python code/empirical/mroz/03_gbc_without_variable_selection.py
python code/empirical/mroz/04_prior_sensitivity_tau0.py
python code/empirical/mroz/05_prior_sensitivity_tau1.py
```

The two main model scripts include the fivefold cross-validation analysis used in the manuscript. The full-data proposed model and narrower-slab specification use 50,000 iterations per chain. The model without variable selection and the narrower-spike specification use 75,000 iterations per chain. The first 25,000 iterations are discarded in all full-data fits. Cross-validation uses 10,000 iterations per chain with the first 5,000 discarded. Output is written under `results/empirical/mroz/`.

## MEPS2001 application

The preparation script reads the `MEPS2001` data distributed with the R package `ssmrob`. Run:

```bash
Rscript code/empirical/meps2001/01_prepare_meps2001_data.R
Rscript code/empirical/meps2001/04_iqbal_fixed_log.R
python code/empirical/meps2001/02_proposed_income_exclusion.py
```

The Iqbal script is run before the proposed-model script because it creates the common stratified cross-validation folds. Both full-data models use 50,000 iterations per chain. The proposed model discards the first 25,000 iterations, whereas the Iqbal model discards the first 5,000. Output is written under `results/empirical/meps2001/`.

The no-exclusion specification is retained as an optional sensitivity analysis that is not reported in the main manuscript:

```bash
python code/empirical/meps2001/03_proposed_no_exclusion.py
```

## Figures

After the simulation analyses have completed, generate the simulation figures with:

```bash
python code/figures/generate_simulation_figures.py
```

Generate the model-structure diagram with:

```bash
python code/figures/generate_model_structure_figure.py
```

After completing the full-data Mroz and MEPS2001 fits, generate the multichain diagnostic figures with:

```bash
python code/figures/plot_mroz_rhat_evolution.py
python code/figures/plot_meps_rhat_evolution.py
```

The empirical fitting scripts write the cumulative rank-normalized split-Rhat summaries required by these two figure scripts.

Figures and their source-data files are written to `figures/`.

## Computational note

The complete Monte Carlo study and the empirical MCMC analyses are computationally intensive. The scripts record their settings, random seeds, input hashes, and output manifests where applicable.

## Citation

If this code is used in published work, please cite the accompanying article and the software record in `CITATION.cff`. Bibliographic details can be updated after the article is accepted.

