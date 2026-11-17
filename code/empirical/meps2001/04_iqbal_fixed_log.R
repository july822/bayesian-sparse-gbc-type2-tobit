# Fit the fixed-log Iqbal model to the MEPS2001 data.

.command_args <- commandArgs(trailingOnly = FALSE)
.file_arg <- .command_args[grepl("^--file=", .command_args)]
.script_dir <- if (length(.file_arg) == 1L) {
  dirname(normalizePath(sub("^--file=", "", .file_arg)))
} else {
  normalizePath(getwd())
}
REPO_ROOT <- normalizePath(file.path(.script_dir, "..", "..", ".."), winslash = "/", mustWork = FALSE)
rm(.command_args, .file_arg, .script_dir)

DATA_DIR <- file.path(REPO_ROOT, "data", "processed", "meps2001")
AUTHOR_ROOT <- file.path(REPO_ROOT, "external", "selection-spike-slab")
args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grepl("^--file=", args)]
SCRIPT_DIR <- if (length(file_arg) == 1L) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  normalizePath(getwd())
}
OUTPUT_DIR <- file.path(REPO_ROOT, "results", "empirical", "meps2001", "iqbal_fixed_log")
N_CHAINS <- 4L
N_MCMC <- 50000L
BURN_IN <- 5000L
CHAIN_SEEDS <- c(20261031L, 20261032L, 20261033L, 20261034L)
PIP_CUTOFF <- 0.5
MARGINAL_EFFECT_CHUNK_SIZE <- 500L
RUN_CROSS_VALIDATION <- TRUE
CV_FOLDS <- 5L
CV_N_CHAINS <- 4L
CV_N_MCMC <- 50000L
CV_BURN_IN <- 5000L
CV_SEED <- 2026111201L
CV_PREDICTION_CHUNK_SIZE <- 500L
CV_MAX_PREDICTIVE_DRAWS <- 4000L
packages <- c(
  "sampleSelection", "MASS", "truncnorm", "posterior", "jsonlite", "digest"
)
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0L) {
  stop(
    "缺少 R 包：", paste(missing, collapse = ", "),
    "\n请先运行 install.packages(c(",
    paste(sprintf("\"%s\"", missing), collapse = ", "), "))"
  )
}
suppressPackageStartupMessages({
  library(MASS)
  library(truncnorm)
})
author_sampler <- file.path(AUTHOR_ROOT, "methods", "gibbs_spike_slab.R")
author_utils <- file.path(AUTHOR_ROOT, "methods", "utils.R")
if (!file.exists(author_sampler) || !file.exists(author_utils)) {
  stop("没有找到作者原始代码，请检查 AUTHOR_ROOT。")
}
old_dir <- getwd()
setwd(file.path(AUTHOR_ROOT, "methods"))
source("gibbs_spike_slab.R")
setwd(old_dir)
data_file <- file.path(DATA_DIR, "meps2001_design_with_exclusion.csv")
manifest_file <- file.path(DATA_DIR, "data_manifest.json")
if (!file.exists(data_file) || !file.exists(manifest_file)) {
  stop("没有找到冻结的 MEPS2001 数据。")
}
manifest <- jsonlite::fromJSON(manifest_file)
data_hash <- digest::digest(file = data_file, algo = "sha256", serialize = FALSE)
if (tolower(data_hash) !=
    tolower(manifest$output_file_sha256$with_exclusion)) {
  stop("数据文件与冻结清单不一致。")
}
dat <- read.csv(data_file, na.strings = c("", "NA"), check.names = FALSE)
x_columns <- c(
  "x_age", "x_female", "x_educ", "x_blhisp", "x_totchr", "x_ins"
)
w_columns <- c(
  "w_age", "w_female", "w_educ", "w_blhisp", "w_totchr", "w_ins",
  "w_income"
)
x_names <- c("age", "female", "educ", "blhisp", "totchr", "ins")
w_names <- c("age", "female", "educ", "blhisp", "totchr", "ins", "income")
required <- c("row_id", "d", "ambexp", x_columns, w_columns)
if (!all(required %in% names(dat))) stop("MEPS2001 设计文件缺少必要列。")
if (nrow(dat) != 3328L) stop("样本量不是 3328。")
if (sum(dat$d == 0L) != 526L) stop("零支出人数不是 526。")
if (sum(dat$d == 1L) != 2802L) stop("正支出人数不是 2802。")
if (!all(dat$d == as.integer(dat$ambexp > 0))) stop("d 与 ambexp 不一致。")
y <- ifelse(dat$d == 1L, log(dat$ambexp), NA_real_)
x <- scale(as.matrix(dat[, x_columns, drop = FALSE]))
w <- scale(as.matrix(dat[, w_columns, drop = FALSE]))
colnames(x) <- x_names
colnames(w) <- w_names
n <- nrow(dat)
p <- ncol(w)
q <- ncol(x)
scaling_table <- rbind(
  data.frame(
    equation = "outcome", variable = x_names,
    center = attr(x, "scaled:center"), scale = attr(x, "scaled:scale")
  ),
  data.frame(
    equation = "selection", variable = w_names,
    center = attr(w, "scaled:center"), scale = attr(w, "scaled:scale")
  )
)
tau_0_alpha <- 1 / sqrt(p * n)
tau_1_alpha <- sqrt(3) / pi
tau_0_beta <- 1 / sqrt(q * n)
tau_1_beta <- sqrt(log(n)) * (4 * log(500))^(-0.5)
r_params <- c(1, p + q)
rho_param <- 5
var_params <- c(1, 1)
alpha_intercept_sd <- 10
beta_intercept_sd <- 10
initial_data <- data.frame(d = dat$d, y = y, w, x, check.names = FALSE)
names(initial_data)[3:(2 + p)] <- paste0("w_", w_names)
names(initial_data)[(3 + p):(2 + p + q)] <- paste0("x_", x_names)
selection_formula <- as.formula(
  paste("d ~", paste(paste0("w_", w_names), collapse = " + "))
)
outcome_formula <- as.formula(
  paste("y ~", paste(paste0("x_", x_names), collapse = " + "))
)
initial_fit <- tryCatch(
  suppressWarnings(sampleSelection::selection(
    selection_formula, outcome_formula,
    data = initial_data, method = "ml"
  )),
  error = function(e) NULL
)
init_alpha <- rep(0, p + 1L)
init_beta <- rep(0, q + 1L)
init_rho <- 0
init_var <- 1
init_gamma_alpha <- rep(0L, p)
init_gamma_beta <- rep(0L, q)
initial_source <- "zeros"
if (!is.null(initial_fit) && all(is.finite(initial_fit$estimate))) {
  estimate <- initial_fit$estimate
  init_alpha <- estimate[1:(p + 1L)]
  init_beta <- estimate[(p + 2L):(p + q + 2L)]
  init_sigma <- estimate[p + q + 3L]
  init_correlation <- estimate[p + q + 4L]
  init_rho <- init_sigma * init_correlation
  init_var <- init_sigma^2 - init_rho^2
  if (!is.finite(init_var) || init_var <= 0) {
    init_rho <- 0
    init_var <- 1
  }
  mle_table <- tryCatch(summary(initial_fit)$estimate, error = function(e) NULL)
  if (!is.null(mle_table)) {
    remove_rows <- c(1L, p + 2L, p + q + 3L, p + q + 4L)
    slope_p <- mle_table[-remove_rows, 4L]
    if (length(slope_p) == p + q && all(is.finite(slope_p))) {
      initial_gamma <- as.integer(slope_p < 0.05)
      init_gamma_alpha <- initial_gamma[1:p]
      init_gamma_beta <- initial_gamma[(p + 1L):(p + q)]
    }
  }
  initial_source <- "sampleSelection_ML"
}
if (dir.exists(OUTPUT_DIR) && length(list.files(OUTPUT_DIR)) > 0L) {
  stop("输出文件夹不是空的，请更换文件夹，避免覆盖已有结果。")
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
fits <- vector("list", N_CHAINS)
chain_summary <- data.frame()
cat("Iqbal spike-and-slab MEPS2001\n")
cat("  n=3328, positive=2802, zero=526\n")
cat("  outcome=log(ambexp), income is the exclusion variable\n")
cat("  chains=4, iterations=50000, burn-in=5000\n")
for (chain in 1:N_CHAINS) {
  cat("Chain ", chain, "/", N_CHAINS, "\n", sep = "")
  set.seed(CHAIN_SEEDS[chain])
  start_time <- proc.time()[["elapsed"]]
  fits[[chain]] <- gibbs_spike_slab(
    n = N_MCMC,
    y = y,
    x_samp = x,
    w_samp = w,
    model_select = TRUE,
    burn_in = BURN_IN,
    init_alpha = init_alpha,
    init_beta = init_beta,
    init_rho = init_rho,
    init_var = init_var,
    init_gamma_alpha = init_gamma_alpha,
    init_gamma_beta = init_gamma_beta,
    init_r = 0.5,
    rho_param = rho_param,
    var_params = var_params,
    r_params = r_params,
    alpha_spike = "normal",
    alpha_slab = "normal",
    beta_spike = "normal",
    beta_slab = "normal",
    tau_0_alpha = tau_0_alpha,
    tau_1_alpha = tau_1_alpha,
    tau_0_beta = tau_0_beta,
    tau_1_beta = tau_1_beta,
    var_scaling = FALSE,
    alpha_intercept_sd = alpha_intercept_sd,
    beta_intercept_sd = beta_intercept_sd,
    fixed_r = FALSE
  )
  runtime <- proc.time()[["elapsed"]] - start_time
  fit <- fits[[chain]]
  if (any(!is.finite(fit$alpha)) || any(!is.finite(fit$beta)) ||
      any(!is.finite(fit$other_params))) {
    stop("第 ", chain, " 条链含有非有限值。")
  }
  one_row <- data.frame(
    chain = chain,
    seed = CHAIN_SEEDS[chain],
    retained_draws = nrow(fit$alpha),
    mean_rho = mean(fit$other_params[, "rho"]),
    mean_sigma = mean(fit$other_params[, "sigma"]),
    mean_r = mean(fit$other_params[, "r"]),
    runtime_seconds = runtime
  )
  chain_summary <- rbind(chain_summary, one_row)
  cat("  rho=", sprintf("%.4f", one_row$mean_rho),
      ", runtime=", sprintf("%.1fs", runtime), "\n", sep = "")
}
parameter_names <- c(
  "alpha_intercept", paste0("alpha_", w_names),
  "beta_intercept", paste0("beta_", x_names),
  "rho", "sigma", "sigma12", "sigma_v2", "r"
)
make_parameter_matrix <- function(fit) {
  rho <- fit$other_params[, "rho"]
  sigma <- fit$other_params[, "sigma"]
  output <- cbind(
    fit$alpha, fit$beta,
    rho, sigma,
    sigma12 = rho * sigma,
    sigma_v2 = sigma^2 * (1 - rho^2),
    r = fit$other_params[, "r"]
  )
  colnames(output) <- parameter_names
  output
}
chain_matrices <- lapply(fits, make_parameter_matrix)
n_keep <- nrow(chain_matrices[[1]])
draw_array <- array(
  NA_real_,
  dim = c(n_keep, N_CHAINS, length(parameter_names)),
  dimnames = list(NULL, paste0("chain", 1:N_CHAINS), parameter_names)
)
for (chain in 1:N_CHAINS) draw_array[, chain, ] <- chain_matrices[[chain]]

make_rhat_evolution <- function(
    draw_array,
    burn_in,
    n_points = 30L,
    min_kept_draws = 1000L
) {
  if (length(dim(draw_array)) != 3L) {
    stop("draw_array must have dimensions iterations x chains x parameters.")
  }
  n_kept <- dim(draw_array)[1L]
  if (n_kept < min_kept_draws) {
    stop(
      "Only ", n_kept,
      " retained draws are available; at least ",
      min_kept_draws, " are required."
    )
  }

  cutoffs <- unique(as.integer(round(seq(
    from = min_kept_draws,
    to = n_kept,
    length.out = n_points
  ))))
  result <- vector("list", length(cutoffs))

  for (i in seq_along(cutoffs)) {
    cutoff <- cutoffs[i]
    prefix_draws <- posterior::as_draws_array(
      draw_array[seq_len(cutoff), , , drop = FALSE]
    )
    diagnostic <- as.data.frame(
      posterior::summarise_draws(prefix_draws, "rhat")
    )
    diagnostic <- diagnostic[
      is.finite(diagnostic$rhat),
      ,
      drop = FALSE
    ]
    if (nrow(diagnostic) == 0L) {
      stop("No finite Rhat values at cutoff ", cutoff, ".")
    }
    worst <- which.max(diagnostic$rhat)
    result[[i]] <- data.frame(
      method = "Iqbal fixed-log model",
      retained_draws_per_chain = cutoff,
      total_iteration = burn_in + cutoff,
      max_rhat = diagnostic$rhat[worst],
      median_rhat = median(diagnostic$rhat),
      parameters_above_1_01 = sum(diagnostic$rhat > 1.01),
      worst_parameter = diagnostic$variable[worst],
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, result)
}

draws <- posterior::as_draws_array(draw_array)
q025 <- function(z) as.numeric(quantile(z, 0.025))
q975 <- function(z) as.numeric(quantile(z, 0.975))
posterior_summary <- as.data.frame(
  posterior::summarise_draws(draws, mean, sd, median, q025, q975)
)
posterior_summary$estimated <- TRUE
posterior_summary <- rbind(
  posterior_summary,
  data.frame(
    variable = "lambda_fixed_log", mean = 0, sd = NA_real_, median = 0,
    q025 = NA_real_, q975 = NA_real_, estimated = FALSE
  )
)
diagnostics <- as.data.frame(
  posterior::summarise_draws(
    draws, "rhat", "ess_bulk", "ess_tail", "mcse_mean"
  )
)
rhat_evolution <- make_rhat_evolution(
  draw_array = draw_array,
  burn_in = BURN_IN
)
make_pip_table <- function(equation, variables, indicator_name) {
  pip_by_chain <- vapply(
    fits,
    function(fit) colMeans(fit[[indicator_name]]),
    numeric(length(variables))
  )
  output <- data.frame(
    equation = equation,
    variable = variables,
    PIP = rowMeans(pip_by_chain),
    selected_PIP_gt_0_5 = as.integer(rowMeans(pip_by_chain) > PIP_CUTOFF)
  )
  for (chain in 1:N_CHAINS) {
    output[[paste0("PIP_chain_", chain)]] <- pip_by_chain[, chain]
  }
  output$PIP_chain_range <- apply(pip_by_chain, 1, function(z) max(z) - min(z))
  output
}
variable_selection <- rbind(
  make_pip_table("selection", w_names, "alpha_indicators"),
  make_pip_table("outcome", x_names, "beta_indicators")
)
compute_posterior_marginal_effects <- function(fits, x, w) {
  n_chains <- length(fits)
  n_keep <- nrow(fits[[1]]$alpha)
  n_draws <- n_chains * n_keep
  alpha <- do.call(rbind, lapply(fits, function(fit) fit$alpha))
  beta <- do.call(rbind, lapply(fits, function(fit) fit$beta))
  sigma <- unlist(lapply(fits, function(fit) fit$other_params[, "sigma"]))
  effect_names <- c(
    "selection_age", "selection_education", "selection_chronic_conditions",
    "selection_income", "selection_female", "selection_black_or_hispanic",
    "selection_insurance", "outcome_age", "outcome_education",
    "outcome_chronic_conditions", "outcome_female",
    "outcome_black_or_hispanic", "outcome_insurance"
  )
  effect_matrix <- matrix(
    NA_real_, nrow = n_draws, ncol = length(effect_names),
    dimnames = list(NULL, effect_names)
  )
  w_center <- attr(w, "scaled:center")
  w_scale <- attr(w, "scaled:scale")
  x_center <- attr(x, "scaled:center")
  x_scale <- attr(x, "scaled:scale")
  x_design <- cbind(intercept = 1, x)
  w_design <- cbind(intercept = 1, w)
  for (start in seq.int(1L, n_draws, by = MARGINAL_EFFECT_CHUNK_SIZE)) {
    stop_index <- min(start + MARGINAL_EFFECT_CHUNK_SIZE - 1L, n_draws)
    index <- start:stop_index
    a <- alpha[index, , drop = FALSE]
    b <- beta[index, , drop = FALSE]
    draw_sigma <- sigma[index]
    selection_index <- w_design %*% t(a)
    average_density <- colMeans(dnorm(selection_index))
    effect_matrix[index, "selection_age"] <-
      average_density * a[, 1L + match("age", w_names)]
    effect_matrix[index, "selection_education"] <-
      average_density * a[, 1L + match("educ", w_names)]
    effect_matrix[index, "selection_chronic_conditions"] <-
      average_density * a[, 1L + match("totchr", w_names)]
    effect_matrix[index, "selection_income"] <-
      average_density * a[, 1L + match("income", w_names)]
    binary_selection_change <- function(variable) {
      j <- match(variable, w_names)
      zero_value <- (0 - w_center[j]) / w_scale[j]
      one_value <- (1 - w_center[j]) / w_scale[j]
      coefficient <- a[, 1L + j]
      index_zero <- selection_index + outer(
        zero_value - w[, j], coefficient
      )
      index_one <- selection_index + outer(
        one_value - w[, j], coefficient
      )
      colMeans(pnorm(index_one) - pnorm(index_zero))
    }
    effect_matrix[index, "selection_female"] <-
      binary_selection_change("female")
    effect_matrix[index, "selection_black_or_hispanic"] <-
      binary_selection_change("blhisp")
    effect_matrix[index, "selection_insurance"] <-
      binary_selection_change("ins")
    linear_predictor <- x_design %*% t(b)
    conditional_mean <- exp(sweep(
      linear_predictor, 2L, 0.5 * draw_sigma^2, FUN = "+"
    ))
    if (any(!is.finite(conditional_mean))) {
      stop("Non-finite outcome marginal effect encountered.")
    }
    average_outcome <- colMeans(conditional_mean)
    effect_matrix[index, "outcome_age"] <-
      average_outcome * b[, 1L + match("age", x_names)]
    effect_matrix[index, "outcome_education"] <-
      average_outcome * b[, 1L + match("educ", x_names)]
    effect_matrix[index, "outcome_chronic_conditions"] <-
      average_outcome * b[, 1L + match("totchr", x_names)]
    binary_outcome_change <- function(variable) {
      j <- match(variable, x_names)
      zero_value <- (0 - x_center[j]) / x_scale[j]
      one_value <- (1 - x_center[j]) / x_scale[j]
      coefficient <- b[, 1L + j]
      index_zero <- linear_predictor + outer(
        zero_value - x[, j], coefficient
      )
      index_one <- linear_predictor + outer(
        one_value - x[, j], coefficient
      )
      mean_one <- exp(sweep(
        index_one, 2L, 0.5 * draw_sigma^2, FUN = "+"
      ))
      mean_zero <- exp(sweep(
        index_zero, 2L, 0.5 * draw_sigma^2, FUN = "+"
      ))
      colMeans(mean_one - mean_zero)
    }
    effect_matrix[index, "outcome_female"] <-
      binary_outcome_change("female")
    effect_matrix[index, "outcome_black_or_hispanic"] <-
      binary_outcome_change("blhisp")
    effect_matrix[index, "outcome_insurance"] <-
      binary_outcome_change("ins")
  }
  effect_metadata <- data.frame(
    effect = effect_names,
    equation = c(rep("selection", 7L), rep("outcome", 6L)),
    variable = c(
      "age", "education", "chronic conditions", "income", "female",
      "Black or Hispanic", "insurance", "age", "education",
      "chronic conditions", "female", "Black or Hispanic", "insurance"
    ),
    scale = c(
      rep("probability", 7L),
      rep("latent ambulatory expenditure (dollars)", 6L)
    ),
    change = c(
      rep("one sample SD", 4L), rep("0 to 1", 3L),
      rep("one sample SD", 3L), rep("0 to 1", 3L)
    ),
    stringsAsFactors = FALSE
  )
  summary_table <- do.call(rbind, lapply(effect_names, function(effect) {
    values <- effect_matrix[, effect]
    data.frame(
      effect = effect,
      posterior_mean = mean(values),
      posterior_median = median(values),
      posterior_sd = sd(values),
      lower_95 = as.numeric(quantile(values, 0.025)),
      upper_95 = as.numeric(quantile(values, 0.975)),
      probability_positive = mean(values > 0),
      stringsAsFactors = FALSE
    )
  }))
  summary_table <- merge(
    effect_metadata, summary_table, by = "effect", sort = FALSE
  )
  summary_table <- summary_table[match(effect_names, summary_table$effect), ]
  effect_array <- array(
    NA_real_, dim = c(n_keep, n_chains, length(effect_names)),
    dimnames = list(NULL, paste0("chain", seq_len(n_chains)), effect_names)
  )
  for (chain in seq_len(n_chains)) {
    rows <- ((chain - 1L) * n_keep + 1L):(chain * n_keep)
    effect_array[, chain, ] <- effect_matrix[rows, , drop = FALSE]
  }
  diagnostics <- as.data.frame(posterior::summarise_draws(
    posterior::as_draws_array(effect_array),
    "rhat", "ess_bulk", "ess_tail", "mcse_mean"
  ))
  draw_table <- data.frame(
    chain = rep(seq_len(n_chains), each = n_keep),
    draw = rep(seq_len(n_keep), times = n_chains),
    effect_matrix,
    check.names = FALSE
  )
  list(draws = draw_table, summary = summary_table, diagnostics = diagnostics)
}
marginal_effects <- compute_posterior_marginal_effects(fits, x, w)
make_stratified_folds <- function(d, n_folds, seed) {
  set.seed(seed)
  folds <- integer(length(d))
  for (value in c(0L, 1L)) {
    indices <- which(d == value)
    indices <- sample(indices, length(indices), replace = FALSE)
    folds[indices] <- rep(seq_len(n_folds), length.out = length(indices))
  }
  folds
}
auc_score <- function(y, probability) {
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  ranks <- rank(probability, ties.method = "average")
  (sum(ranks[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
initial_values_for_data <- function(y, x, w, d) {
  p_local <- ncol(w)
  q_local <- ncol(x)
  initial_data <- data.frame(d = d, y = y, w, x, check.names = FALSE)
  names(initial_data)[3:(2 + p_local)] <- paste0("w_", colnames(w))
  names(initial_data)[(3 + p_local):(2 + p_local + q_local)] <-
    paste0("x_", colnames(x))
  selection_formula <- as.formula(
    paste("d ~", paste(paste0("w_", colnames(w)), collapse = " + "))
  )
  outcome_formula <- as.formula(
    paste("y ~", paste(paste0("x_", colnames(x)), collapse = " + "))
  )
  fit <- tryCatch(
    suppressWarnings(sampleSelection::selection(
      selection_formula, outcome_formula,
      data = initial_data, method = "ml"
    )),
    error = function(e) NULL
  )
  output <- list(
    alpha = rep(0, p_local + 1L), beta = rep(0, q_local + 1L),
    rho = 0, var = 1, gamma_alpha = rep(0L, p_local),
    gamma_beta = rep(0L, q_local), source = "zeros"
  )
  if (is.null(fit) || any(!is.finite(fit$estimate))) return(output)
  estimate <- fit$estimate
  output$alpha <- estimate[1:(p_local + 1L)]
  output$beta <- estimate[(p_local + 2L):(p_local + q_local + 2L)]
  initial_sigma <- estimate[p_local + q_local + 3L]
  initial_correlation <- estimate[p_local + q_local + 4L]
  output$rho <- initial_sigma * initial_correlation
  output$var <- initial_sigma^2 - output$rho^2
  if (!is.finite(output$var) || output$var <= 0) {
    output$rho <- 0
    output$var <- 1
  }
  mle_table <- tryCatch(summary(fit)$estimate, error = function(e) NULL)
  if (!is.null(mle_table)) {
    remove_rows <- c(
      1L, p_local + 2L, p_local + q_local + 3L,
      p_local + q_local + 4L
    )
    slope_p <- mle_table[-remove_rows, 4L]
    if (length(slope_p) == p_local + q_local && all(is.finite(slope_p))) {
      initial_gamma <- as.integer(slope_p < 0.05)
      output$gamma_alpha <- initial_gamma[1:p_local]
      output$gamma_beta <- initial_gamma[(p_local + 1L):(p_local + q_local)]
    }
  }
  output$source <- "sampleSelection_ML"
  output
}
predict_iqbal_holdout <- function(fits, x, w, d, seed) {
  draws_per_chain <- max(1L, floor(CV_MAX_PREDICTIVE_DRAWS / length(fits)))
  draw_indices <- lapply(fits, function(fit) {
    unique(as.integer(round(seq(
      1, nrow(fit$alpha), length.out = min(draws_per_chain, nrow(fit$alpha))
    ))))
  })
  alpha <- do.call(rbind, Map(
    function(fit, index) fit$alpha[index, , drop = FALSE],
    fits, draw_indices
  ))
  beta <- do.call(rbind, Map(
    function(fit, index) fit$beta[index, , drop = FALSE],
    fits, draw_indices
  ))
  correlation <- unlist(Map(
    function(fit, index) fit$other_params[index, "rho"],
    fits, draw_indices
  ))
  sigma <- unlist(Map(
    function(fit, index) fit$other_params[index, "sigma"],
    fits, draw_indices
  ))
  sigma12 <- correlation * sigma
  sigma_v <- sigma * sqrt(pmax(1 - correlation^2, 0))
  n_draws <- nrow(alpha)
  x_design <- cbind(intercept = 1, x)
  w_design <- cbind(intercept = 1, w)
  selection_probability_sum <- numeric(length(d))
  observed <- d == 1L
  predictive <- matrix(NA_real_, nrow = sum(observed), ncol = n_draws)
  set.seed(seed)
  for (start in seq.int(1L, n_draws, by = CV_PREDICTION_CHUNK_SIZE)) {
    stop_index <- min(start + CV_PREDICTION_CHUNK_SIZE - 1L, n_draws)
    index <- start:stop_index
    a <- alpha[index, , drop = FALSE]
    b <- beta[index, , drop = FALSE]
    selection_index <- w_design %*% t(a)
    selection_probability_sum <- selection_probability_sum +
      rowSums(pnorm(selection_index))
    observed_index <- selection_index[observed, , drop = FALSE]
    lower_probability <- pnorm(-observed_index)
    uniforms <- lower_probability +
      (1 - lower_probability) * matrix(
        runif(length(lower_probability)), nrow = nrow(lower_probability)
      )
    epsilon1 <- qnorm(pmin(pmax(uniforms, 1e-12), 1 - 1e-12))
    log_outcome <- x_design[observed, , drop = FALSE] %*% t(b) +
      sweep(epsilon1, 2L, sigma12[index], FUN = "*") +
      sweep(
        matrix(rnorm(length(epsilon1)), nrow = nrow(epsilon1)),
        2L, sigma_v[index], FUN = "*"
      )
    outcome <- exp(log_outcome)
    if (any(!is.finite(outcome))) {
      stop("Non-finite cross-validated predictive draw encountered.")
    }
    predictive[, index] <- outcome
  }
  interval <- t(apply(
    predictive, 1L, quantile, probs = c(0.025, 0.975), names = FALSE
  ))
  list(
    selection_probability = selection_probability_sum / n_draws,
    predictive_mean = rowMeans(predictive),
    predictive_lower_95 = interval[, 1L],
    predictive_upper_95 = interval[, 2L]
  )
}
summarize_cv_diagnostics <- function(fits) {
  matrices <- lapply(fits, make_parameter_matrix)
  n_keep_local <- nrow(matrices[[1]])
  n_chains_local <- length(matrices)
  values <- array(
    NA_real_,
    dim = c(n_keep_local, n_chains_local, length(parameter_names)),
    dimnames = list(
      NULL, paste0("chain", seq_len(n_chains_local)), parameter_names
    )
  )
  for (chain in seq_len(n_chains_local)) {
    values[, chain, ] <- matrices[[chain]]
  }
  diagnostics <- as.data.frame(posterior::summarise_draws(
    posterior::as_draws_array(values), "rhat", "ess_bulk", "ess_tail"
  ))
  data.frame(
    max_rhat = max(diagnostics$rhat, na.rm = TRUE),
    min_bulk_ess = min(diagnostics$ess_bulk, na.rm = TRUE),
    min_tail_ess = min(diagnostics$ess_tail, na.rm = TRUE)
  )
}
cross_validated_predictive_performance <- function(dat) {
  d_all <- as.integer(dat$d)
  raw_x <- as.matrix(dat[, x_columns, drop = FALSE])
  raw_w <- as.matrix(dat[, w_columns, drop = FALSE])
  folds <- make_stratified_folds(d_all, CV_FOLDS, CV_SEED)
  fold_assignments <- data.frame(
    row_id = dat$row_id, d = d_all, fold = folds
  )
  prediction_rows <- list()
  metric_rows <- list()
  for (fold in seq_len(CV_FOLDS)) {
    cat("Cross-validation fold ", fold, "/", CV_FOLDS, "\n", sep = "")
    train <- folds != fold
    test <- folds == fold
    d_train <- d_all[train]
    d_test <- d_all[test]
    x_center <- colMeans(raw_x[train, , drop = FALSE])
    x_scale <- apply(raw_x[train, , drop = FALSE], 2L, sd)
    w_center <- colMeans(raw_w[train, , drop = FALSE])
    w_scale <- apply(raw_w[train, , drop = FALSE], 2L, sd)
    if (any(x_scale <= 0) || any(w_scale <= 0)) {
      stop("A cross-validation training fold contains a constant predictor.")
    }
    x_train <- scale(raw_x[train, , drop = FALSE], x_center, x_scale)
    x_test <- scale(raw_x[test, , drop = FALSE], x_center, x_scale)
    w_train <- scale(raw_w[train, , drop = FALSE], w_center, w_scale)
    w_test <- scale(raw_w[test, , drop = FALSE], w_center, w_scale)
    colnames(x_train) <- colnames(x_test) <- x_names
    colnames(w_train) <- colnames(w_test) <- w_names
    y_train <- ifelse(
      d_train == 1L, log(dat$ambexp[train]), NA_real_
    )
    initial <- initial_values_for_data(y_train, x_train, w_train, d_train)
    n_train <- sum(train)
    tau_0_alpha_cv <- 1 / sqrt(p * n_train)
    tau_1_alpha_cv <- sqrt(3) / pi
    tau_0_beta_cv <- 1 / sqrt(q * n_train)
    tau_1_beta_cv <- sqrt(log(n_train)) * (4 * log(500))^(-0.5)
    fold_fits <- vector("list", CV_N_CHAINS)
    for (chain in seq_len(CV_N_CHAINS)) {
      set.seed(CV_SEED + 10000L * fold + chain)
      fold_fits[[chain]] <- gibbs_spike_slab(
        n = CV_N_MCMC,
        y = y_train,
        x_samp = x_train,
        w_samp = w_train,
        model_select = TRUE,
        burn_in = CV_BURN_IN,
        init_alpha = initial$alpha,
        init_beta = initial$beta,
        init_rho = initial$rho,
        init_var = initial$var,
        init_gamma_alpha = initial$gamma_alpha,
        init_gamma_beta = initial$gamma_beta,
        init_r = 0.5,
        rho_param = rho_param,
        var_params = var_params,
        r_params = r_params,
        alpha_spike = "normal",
        alpha_slab = "normal",
        beta_spike = "normal",
        beta_slab = "normal",
        tau_0_alpha = tau_0_alpha_cv,
        tau_1_alpha = tau_1_alpha_cv,
        tau_0_beta = tau_0_beta_cv,
        tau_1_beta = tau_1_beta_cv,
        var_scaling = FALSE,
        alpha_intercept_sd = alpha_intercept_sd,
        beta_intercept_sd = beta_intercept_sd,
        fixed_r = FALSE
      )
    }
    prediction <- predict_iqbal_holdout(
      fold_fits, x_test, w_test, d_test,
      seed = CV_SEED + 50000L + fold
    )
    fold_diagnostics <- summarize_cv_diagnostics(fold_fits)
    test_rows <- which(test)
    observed <- d_test == 1L
    observed_outcome <- dat$ambexp[test_rows[observed]]
    probability <- prediction$selection_probability
    clipped <- pmin(pmax(probability, 1e-12), 1 - 1e-12)
    metric_rows[[fold]] <- data.frame(
      fold = fold,
      AUC = auc_score(d_test, probability),
      Brier_score = mean((d_test - probability)^2),
      log_loss = -mean(
        d_test * log(clipped) + (1 - d_test) * log1p(-clipped)
      ),
      RMSE_d1 = sqrt(mean(
        (prediction$predictive_mean - observed_outcome)^2
      )),
      MAE_d1 = mean(abs(
        prediction$predictive_mean - observed_outcome
      )),
      coverage_95_d1 = mean(
        observed_outcome >= prediction$predictive_lower_95 &
          observed_outcome <= prediction$predictive_upper_95
      ),
      mean_interval_length_d1 = mean(
        prediction$predictive_upper_95 - prediction$predictive_lower_95
      ),
      n_observed_outcomes = length(observed_outcome),
      max_rhat = fold_diagnostics$max_rhat,
      min_bulk_ess = fold_diagnostics$min_bulk_ess,
      min_tail_ess = fold_diagnostics$min_tail_ess
    )
    fold_predictions <- data.frame(
      row_id = dat$row_id[test_rows], fold = fold, d = d_test,
      selection_probability = probability, outcome = NA_real_,
      predictive_mean = NA_real_, predictive_lower_95 = NA_real_,
      predictive_upper_95 = NA_real_
    )
    fold_predictions$outcome[observed] <- observed_outcome
    fold_predictions$predictive_mean[observed] <- prediction$predictive_mean
    fold_predictions$predictive_lower_95[observed] <-
      prediction$predictive_lower_95
    fold_predictions$predictive_upper_95[observed] <-
      prediction$predictive_upper_95
    prediction_rows[[fold]] <- fold_predictions
  }
  predictions <- do.call(rbind, prediction_rows)
  predictions <- predictions[order(predictions$row_id), ]
  observed <- predictions$d == 1L
  probability <- predictions$selection_probability
  clipped <- pmin(pmax(probability, 1e-12), 1 - 1e-12)
  observed_outcome <- predictions$outcome[observed]
  predictive_mean <- predictions$predictive_mean[observed]
  lower <- predictions$predictive_lower_95[observed]
  upper <- predictions$predictive_upper_95[observed]
  overall <- data.frame(
    AUC = auc_score(predictions$d, probability),
    Brier_score = mean((predictions$d - probability)^2),
    log_loss = -mean(
      predictions$d * log(clipped) +
        (1 - predictions$d) * log1p(-clipped)
    ),
    RMSE_d1 = sqrt(mean((predictive_mean - observed_outcome)^2)),
    MAE_d1 = mean(abs(predictive_mean - observed_outcome)),
    coverage_95_d1 = mean(observed_outcome >= lower & observed_outcome <= upper),
    mean_interval_length_d1 = mean(upper - lower),
    n_observed_outcomes = length(observed_outcome)
  )
  list(
    folds = fold_assignments,
    predictions = predictions,
    by_fold = do.call(rbind, metric_rows),
    overall = overall
  )
}
cv_results <- NULL
if (RUN_CROSS_VALIDATION) {
  cv_results <- cross_validated_predictive_performance(dat)
}
write.csv(posterior_summary,
          file.path(OUTPUT_DIR, "posterior_summary.csv"), row.names = FALSE)
write.csv(variable_selection,
          file.path(OUTPUT_DIR, "variable_selection.csv"), row.names = FALSE)
write.csv(diagnostics,
          file.path(OUTPUT_DIR, "multichain_diagnostics.csv"), row.names = FALSE)
write.csv(
  rhat_evolution,
  file.path(OUTPUT_DIR, "rhat_evolution.csv"),
  row.names = FALSE
)
write.csv(chain_summary,
          file.path(OUTPUT_DIR, "chain_summary.csv"), row.names = FALSE)
write.csv(scaling_table,
          file.path(OUTPUT_DIR, "iqbal_predictor_scaling.csv"), row.names = FALSE)
write.csv(
  marginal_effects$draws,
  gzfile(file.path(OUTPUT_DIR, "marginal_effect_posterior_draws.csv.gz")),
  row.names = FALSE
)
write.csv(
  marginal_effects$summary,
  file.path(OUTPUT_DIR, "marginal_effect_posterior_summary.csv"),
  row.names = FALSE
)
write.csv(
  marginal_effects$diagnostics,
  file.path(OUTPUT_DIR, "marginal_effect_diagnostics.csv"),
  row.names = FALSE
)
if (RUN_CROSS_VALIDATION) {
  cv_dir <- file.path(OUTPUT_DIR, "cross_validation")
  dir.create(cv_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(cv_results$folds,
            file.path(cv_dir, "fold_assignments.csv"), row.names = FALSE)
  write.csv(cv_results$predictions,
            file.path(cv_dir, "out_of_fold_predictions.csv"), row.names = FALSE)
  write.csv(cv_results$by_fold,
            file.path(cv_dir, "metrics_by_fold.csv"), row.names = FALSE)
  write.csv(cv_results$overall,
            file.path(cv_dir, "metrics_overall.csv"), row.names = FALSE)
}
run_manifest <- list(
  method = "Iqbal normal-normal Class I spike-and-slab sample-selection model",
  reference = "Iqbal, Ogundimu and Rubio (2026), doi:10.1007/s00180-026-01745-3",
  author_repository = "https://github.com/adam-iqbal/selection-spike-slab",
  author_sampler_modified = FALSE,
  wrapper_extended_for_posterior_marginal_effects = TRUE,
  author_sampler_sha256 = digest::digest(
    file = author_sampler, algo = "sha256", serialize = FALSE
  ),
  author_utils_sha256 = digest::digest(
    file = author_utils, algo = "sha256", serialize = FALSE
  ),
  data_sha256 = data_hash,
  outcome = "log(ambexp), calculated from original ambexp",
  exclusion_variable = "income",
  x_variables = x_names,
  w_variables = w_names,
  chains = N_CHAINS,
  iterations = N_MCMC,
  burn_in = BURN_IN,
  seeds = CHAIN_SEEDS,
  initialization = initial_source,
  prior = list(
    family = "normal-normal Class I",
    tau_0_alpha = tau_0_alpha,
    tau_1_alpha = tau_1_alpha,
    tau_0_beta = tau_0_beta,
    tau_1_beta = tau_1_beta,
    r_prior = r_params,
    rho_param = rho_param,
    variance_IG = var_params,
    alpha_intercept_sd = alpha_intercept_sd,
    beta_intercept_sd = beta_intercept_sd,
    source = "authors' ambulatory_testing.R empirical-application settings"
  ),
  pip_cutoff = PIP_CUTOFF,
  marginal_effects = list(
    posterior_draw_level = TRUE,
    continuous_change = "one sample standard deviation",
    binary_change = "zero to one",
    outcome_scale = "latent ambulatory expenditure in dollars",
    interval = "95% equal-tail posterior credible interval"
  ),
  cross_validation = list(
    enabled = RUN_CROSS_VALIDATION,
    folds = CV_FOLDS,
    chains_per_fold = CV_N_CHAINS,
    iterations_per_chain = CV_N_MCMC,
    burn_in_per_chain = CV_BURN_IN,
    posterior_predictive_draws = CV_MAX_PREDICTIVE_DRAWS,
    seed = CV_SEED,
    outcome_metrics = "held-out observations with positive expenditure",
    selection_metrics = "all held-out observations"
  ),
  note = paste(
    "The author sampler was not modified.",
    "The wrapper adds multichain diagnostics, draw-level marginal effects,",
    "and optional five-fold cross-validation."
  )
)
jsonlite::write_json(
  run_manifest, file.path(OUTPUT_DIR, "run_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE, digits = NA
)
cat("\nCompleted. Results:\n")
for (name in c(
  "posterior_summary.csv", "variable_selection.csv",
  "multichain_diagnostics.csv", "rhat_evolution.csv",
  "chain_summary.csv",
  "iqbal_predictor_scaling.csv",
  "marginal_effect_posterior_draws.csv.gz",
  "marginal_effect_posterior_summary.csv",
  "marginal_effect_diagnostics.csv", "run_manifest.json"
)) {
  cat(file.path(OUTPUT_DIR, name), "\n")
}
if (RUN_CROSS_VALIDATION) {
  for (name in c(
    "fold_assignments.csv", "out_of_fold_predictions.csv",
    "metrics_by_fold.csv", "metrics_overall.csv"
  )) {
    cat(file.path(OUTPUT_DIR, "cross_validation", name), "\n")
  }
}
