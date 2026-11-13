# Fit the standard Heckman two-step benchmark in the primary simulation.

.command_args <- commandArgs(trailingOnly = FALSE)
.file_arg <- .command_args[grepl("^--file=", .command_args)]
.script_dir <- if (length(.file_arg) == 1L) {
  dirname(normalizePath(sub("^--file=", "", .file_arg)))
} else {
  normalizePath(getwd())
}
REPO_ROOT <- normalizePath(file.path(.script_dir, "..", ".."), winslash = "/", mustWork = FALSE)
rm(.command_args, .file_arg, .script_dir)

COMMON_DATA_DIR <- file.path(REPO_ROOT, "data", "simulation")
OUTPUT_DIR <- file.path(REPO_ROOT, "results", "simulation", "heckman_two_step")
REPLICATION_LIMIT <- NULL
N_PREDICTIVE_DRAWS <- 500
PREDICTION_MASTER_SEED <- 20261016
METHOD_NAME <- "normal_heckman_two_step"
ALLOW_OVERWRITE <- FALSE
packages <- c("sampleSelection", "jsonlite", "digest", "matrixStats", "MASS")
missing_packages <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "缺少R包：", paste(missing_packages, collapse = ", "),
    "\n请先运行：install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "), "))"
  )
}
sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}
auc_rank <- function(truth, probability) {
  truth <- as.integer(truth)
  n1 <- sum(truth == 1L)
  n0 <- sum(truth == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  r <- rank(probability, ties.method = "average")
  (sum(r[truth == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
make_psd <- function(v) {
  v <- (v + t(v)) / 2
  e <- eigen(v, symmetric = TRUE)
  e$vectors %*% diag(pmax(e$values, 1e-10)) %*% t(e$vectors)
}
safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  mean(x)
}
safe_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(NA_real_)
  stats::sd(x)
}
safe_quantile <- function(x, p) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  as.numeric(stats::quantile(x, p, names = FALSE))
}
manifest_path <- file.path(COMMON_DATA_DIR, "dataset_manifest.csv")
config_path <- file.path(COMMON_DATA_DIR, "simulation_config.json")
if (!file.exists(manifest_path) || !file.exists(config_path)) {
  stop("没有找到公共数据的dataset_manifest.csv或simulation_config.json。")
}
manifest <- utils::read.csv(
  manifest_path, stringsAsFactors = FALSE, check.names = FALSE
)
config <- jsonlite::fromJSON(config_path)
if (!identical(as.character(config$scenario_id),
               "baseline_valid_exclusion")) {
  stop("当前公共数据不是有效排除变量基准场景。")
}
if (!identical(as.character(
  config$exclusion_restriction$selection_only_column
), "w3")) {
  stop("配置中没有确认w3为选择方程专属排除变量。")
}
if (!is.null(REPLICATION_LIMIT)) {
  manifest <- manifest[seq_len(min(REPLICATION_LIMIT, nrow(manifest))), ]
}
p <- as.integer(config$p_without_intercept)
x_columns <- c("x0", paste0("x", seq_len(p)))
w_columns <- c("w0", paste0("w", seq_len(p)))
selection_formula <- stats::as.formula(
  paste("d ~", paste(paste0("w", seq_len(p)), collapse = " + "))
)
outcome_formula <- stats::as.formula(
  paste("t_identity ~", paste(paste0("x", seq_len(p)), collapse = " + "))
)
if (dir.exists(OUTPUT_DIR) && length(list.files(OUTPUT_DIR)) > 0L &&
    !ALLOW_OVERWRITE) {
  stop(
    "结果文件夹不是空的。为保护旧结果，请更换OUTPUT_DIR，",
    "或确认后设置ALLOW_OVERWRITE <- TRUE。\n", OUTPUT_DIR
  )
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
parameter_results <- list()
variable_results <- list()
metric_results <- list()
status_results <- list()
prediction_path <- file.path(OUTPUT_DIR, "predictions.csv.gz")
prediction_connection <- gzfile(prediction_path, open = "wt")
prediction_header <- data.frame(
  method = character(), replication = integer(), row_id = integer(),
  selection_probability = numeric(),
  t_mean_latent = numeric(), t_mean_selected = numeric(),
  selection_bias_t = numeric(),
  y_mean_unconditional = numeric(),
  y_lower_unconditional = numeric(), y_upper_unconditional = numeric(),
  y_mean_selected = numeric(),
  y_lower_selected = numeric(), y_upper_selected = numeric()
)
utils::write.table(
  prediction_header, prediction_connection,
  sep = ",", row.names = FALSE, col.names = TRUE,
  quote = FALSE, na = "NA"
)
cat("标准Heckman two-step有效排除变量公共数据实验\n")
cat("重复次数：", nrow(manifest), "\n", sep = "")
cat("结果路径：", OUTPUT_DIR, "\n", sep = "")
for (i in seq_len(nrow(manifest))) {
  rep_id <- as.integer(manifest$replication[i])
  total_start <- proc.time()[["elapsed"]]
  cat("Replication ", i, "/", nrow(manifest), "\n", sep = "")
  result <- tryCatch({
    train_path <- file.path(COMMON_DATA_DIR, manifest$train_file[i])
    test_path <- file.path(COMMON_DATA_DIR, manifest$test_file[i])
    train_hash_ok <- identical(
      tolower(sha256_file(train_path)),
      tolower(manifest$train_sha256[i])
    )
    if (!train_hash_ok) stop("训练文件SHA-256不一致。")
    train <- utils::read.csv(
      train_path, na.strings = "NA", check.names = FALSE
    )
    required_train <- c("row_id", "d", "y_observed", x_columns, w_columns)
    missing_train <- setdiff(required_train, names(train))
    if (length(missing_train) > 0L) {
      stop("训练集缺少列：", paste(missing_train, collapse = ", "))
    }
    train$t_identity <- train$y_observed - 1
    fit_start <- proc.time()[["elapsed"]]
    fit <- suppressWarnings(
      sampleSelection::selection(
        selection = selection_formula,
        outcome = outcome_formula,
        data = train,
        method = "2step"
      )
    )
    fit_seconds <- proc.time()[["elapsed"]] - fit_start
    return_code <- as.integer(fit$probit$code)
    if (!return_code %in% c(1L, 2L, 8L)) {
      stop("第一步Probit没有正常收敛，return code = ", return_code)
    }
    coefficient <- stats::coef(fit)
    covariance <- stats::vcov(fit)
    index_alpha <- fit$param$index$betaS
    index_beta <- fit$param$index$betaO
    index_mills <- unname(fit$param$index$Mills)
    index_sigma <- unname(fit$param$index$sigma)
    index_rho <- unname(fit$param$index$rho)
    alpha_hat <- unname(coefficient[index_alpha])
    beta_hat <- unname(coefficient[index_beta])
    sigma_hat <- unname(coefficient[index_sigma])
    rho_hat <- unname(coefficient[index_rho])
    covariance_alpha <- covariance[index_alpha, index_alpha, drop = FALSE]
    covariance_beta <- covariance[index_beta, index_beta, drop = FALSE]
    estimated_indices <- c(index_alpha, index_beta, index_mills)
    if (any(!is.finite(coefficient)) ||
        any(!is.finite(covariance_alpha)) ||
        any(!is.finite(covariance_beta)) ||
        any(!is.finite(diag(covariance)[estimated_indices]))) {
      stop("two-step直接估计参数或其有效方差块包含非有限值。")
    }
    if (!is.finite(sigma_hat) || sigma_hat <= 0 ||
        !is.finite(rho_hat) || abs(rho_hat) >= 1) {
      stop("two-step得到无效的sigma或|rho|>=1，无法构造预测分布。")
    }
    sigma12_hat <- unname(coefficient[index_mills])
    sigma_v2_hat <- sigma_hat^2 * (1 - rho_hat^2)
    se_alpha <- sqrt(diag(covariance)[index_alpha])
    se_beta <- sqrt(diag(covariance)[index_beta])
    se_sigma12 <- sqrt(diag(covariance)[index_mills])
    se_sigma_v2 <- NA_real_
    parameter_name <- c(
      paste0("alpha_", 0:p),
      paste0("beta_", 0:p),
      "sigma12",
      "sigma_v2",
      "lambda_fixed"
    )
    reference_truth <- c(
      config$alpha_true,
      config$beta_true,
      config$sigma12_true,
      config$sigma_v2_true,
      config$lambda_true
    )
    estimate <- c(
      alpha_hat,
      beta_hat,
      sigma12_hat,
      sigma_v2_hat,
      1.0
    )
    standard_error <- c(
      se_alpha,
      se_beta,
      se_sigma12,
      se_sigma_v2,
      NA_real_
    )
    comparable_to_dgp <- startsWith(parameter_name, "alpha_")
    lower_95 <- estimate - stats::qnorm(0.975) * standard_error
    upper_95 <- estimate + stats::qnorm(0.975) * standard_error
    bias <- ifelse(
      comparable_to_dgp,
      estimate - reference_truth,
      NA_real_
    )
    squared_error <- ifelse(
      comparable_to_dgp,
      (estimate - reference_truth)^2,
      NA_real_
    )
    covered_95 <- rep(NA_integer_, length(parameter_name))
    valid_coverage <- comparable_to_dgp &
      is.finite(lower_95) &
      is.finite(upper_95)
    covered_95[valid_coverage] <- as.integer(
      reference_truth[valid_coverage] >= lower_95[valid_coverage] &
        reference_truth[valid_coverage] <= upper_95[valid_coverage]
    )
    parameter_table <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      parameter = parameter_name,
      truth = reference_truth,
      comparable_to_dgp = comparable_to_dgp,
      estimate = estimate,
      standard_error = standard_error,
      lower_95 = lower_95,
      upper_95 = upper_95,
      bias = bias,
      squared_error = squared_error,
      covered_95 = covered_95,
      interval_length = upper_95 - lower_95,
      stringsAsFactors = FALSE
    )
    variable_table <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      equation = c("selection", "outcome"),
      applicable = FALSE,
      matrix(NA_real_, nrow = 2, ncol = p,
             dimnames = list(NULL, paste0("PIP_", 1:p))),
      TPR = NA_real_, FPR = NA_real_, FDR = NA_real_,
      Specificity = NA_real_, MCC = NA_real_, SelectedSize = NA_real_,
      check.names = FALSE
    )
    test_hash_ok <- identical(
      tolower(sha256_file(test_path)),
      tolower(manifest$test_sha256[i])
    )
    if (!test_hash_ok) stop("测试文件SHA-256不一致。")
    test <- utils::read.csv(
      test_path, na.strings = "NA", check.names = FALSE
    )
    required_test <- c("row_id", "d", "y_star", "t_star",
                       x_columns, w_columns)
    missing_test <- setdiff(required_test, names(test))
    if (length(missing_test) > 0L) {
      stop("测试集缺少列：", paste(missing_test, collapse = ", "))
    }
    x_test <- as.matrix(test[, x_columns, drop = FALSE])
    w_test <- as.matrix(test[, w_columns, drop = FALSE])
    prediction_start <- proc.time()[["elapsed"]]
    eta_hat <- as.vector(w_test %*% alpha_hat)
    selection_probability <- stats::pnorm(eta_hat)
    mills_hat <- exp(
      stats::dnorm(eta_hat, log = TRUE) -
        stats::pnorm(eta_hat, log.p = TRUE)
    )
    t_mean_latent <- as.vector(x_test %*% beta_hat)
    selection_bias_t <- sigma12_hat * mills_hat
    t_mean_selected <- t_mean_latent + selection_bias_t
    y_mean_unconditional <- t_mean_latent + 1
    y_mean_selected <- t_mean_selected + 1
    set.seed(PREDICTION_MASTER_SEED + rep_id)
    alpha_draws <- MASS::mvrnorm(
      n = N_PREDICTIVE_DRAWS,
      mu = alpha_hat,
      Sigma = make_psd(covariance_alpha),
      tol = 1e-8
    )
    beta_draws <- MASS::mvrnorm(
      n = N_PREDICTIVE_DRAWS,
      mu = beta_hat,
      Sigma = make_psd(covariance_beta),
      tol = 1e-8
    )
    if (is.null(dim(alpha_draws))) alpha_draws <- matrix(alpha_draws, nrow = 1L)
    if (is.null(dim(beta_draws))) beta_draws <- matrix(beta_draws, nrow = 1L)
    sigma_draws <- rep(sigma_hat, N_PREDICTIVE_DRAWS)
    rho_draws <- rep(rho_hat, N_PREDICTIVE_DRAWS)
    mean_draws <- x_test %*% t(beta_draws)
    eta_draws <- w_test %*% t(alpha_draws)
    n_test <- nrow(test)
    z_unconditional <- matrix(
      stats::rnorm(n_test * N_PREDICTIVE_DRAWS),
      nrow = n_test
    )
    error_unconditional <- sweep(z_unconditional, 2L, sigma_draws, "*")
    y_draws_unconditional <- mean_draws + error_unconditional + 1
    lower_probability <- stats::pnorm(-eta_draws)
    selected_probability <- stats::pnorm(eta_draws)
    uniform_draws <- matrix(
      stats::runif(n_test * N_PREDICTIVE_DRAWS),
      nrow = n_test
    )
    truncated_probability <- lower_probability +
      uniform_draws * selected_probability
    truncated_probability <- pmin(
      pmax(truncated_probability, 1e-12), 1 - 1e-12
    )
    epsilon_selected <- stats::qnorm(truncated_probability)
    z_independent <- matrix(
      stats::rnorm(n_test * N_PREDICTIVE_DRAWS),
      nrow = n_test
    )
    correlated_error <- sweep(
      epsilon_selected, 2L, rho_draws * sigma_draws, "*"
    )
    independent_error <- sweep(
      z_independent, 2L,
      sigma_draws * sqrt(1 - rho_draws^2), "*"
    )
    y_draws_selected <- mean_draws + correlated_error +
      independent_error + 1
    interval_unconditional <- matrixStats::rowQuantiles(
      y_draws_unconditional, probs = c(0.025, 0.975)
    )
    interval_selected <- matrixStats::rowQuantiles(
      y_draws_selected, probs = c(0.025, 0.975)
    )
    prediction_seconds <- proc.time()[["elapsed"]] - prediction_start
    prediction_table <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      row_id = test$row_id,
      selection_probability = selection_probability,
      t_mean_latent = t_mean_latent,
      t_mean_selected = t_mean_selected,
      selection_bias_t = selection_bias_t,
      y_mean_unconditional = y_mean_unconditional,
      y_lower_unconditional = interval_unconditional[, 1],
      y_upper_unconditional = interval_unconditional[, 2],
      y_mean_selected = y_mean_selected,
      y_lower_selected = interval_selected[, 1],
      y_upper_selected = interval_selected[, 2],
      stringsAsFactors = FALSE
    )
    d_test <- as.integer(test$d)
    selected <- d_test == 1L
    probability_safe <- pmin(
      pmax(selection_probability, 1e-12), 1 - 1e-12
    )
    metric_table <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      n_test = n_test,
      n_test_selected = sum(selected),
      test_observed_fraction = mean(d_test),
      selection_auc = auc_rank(d_test, selection_probability),
      selection_brier = mean((d_test - selection_probability)^2),
      selection_log_loss = -mean(
        d_test * log(probability_safe) +
          (1 - d_test) * log(1 - probability_safe)
      ),
      t_latent_rmse = NA_real_,
      t_latent_mae = NA_real_,
      y_latent_rmse = sqrt(mean(
        (test$y_star - y_mean_unconditional)^2
      )),
      y_latent_mae = mean(abs(
        test$y_star - y_mean_unconditional
      )),
      y_latent_coverage_95 = mean(
        test$y_star >= interval_unconditional[, 1] &
          test$y_star <= interval_unconditional[, 2]
      ),
      y_latent_interval_length = mean(
        interval_unconditional[, 2] - interval_unconditional[, 1]
      ),
      t_selected_rmse = NA_real_,
      t_selected_mae = NA_real_,
      y_selected_rmse = sqrt(mean(
        (test$y_star[selected] - y_mean_selected[selected])^2
      )),
      y_selected_mae = mean(abs(
        test$y_star[selected] - y_mean_selected[selected]
      )),
      y_selected_coverage_95 = mean(
        test$y_star[selected] >= interval_selected[selected, 1] &
          test$y_star[selected] <= interval_selected[selected, 2]
      ),
      y_selected_interval_length = mean(
        interval_selected[selected, 2] - interval_selected[selected, 1]
      ),
      stringsAsFactors = FALSE
    )
    status_table <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      train_seed = manifest$train_seed[i],
      test_seed = manifest$test_seed[i],
      prediction_seed = PREDICTION_MASTER_SEED + rep_id,
      n_train = nrow(train),
      n_train_selected = sum(train$d == 1L),
      n_test = nrow(test),
      observed_fraction = mean(train$d),
      sigma_estimate = sigma_hat,
      rho_estimate = rho_hat,
      sigma12_estimate = sigma12_hat,
      return_code = return_code,
      return_message = as.character(fit$probit$message),
      log_likelihood = NA_real_,
      fit_seconds = fit_seconds,
      prediction_seconds = prediction_seconds,
      total_seconds = proc.time()[["elapsed"]] - total_start,
      train_sha256_verified = train_hash_ok,
      test_sha256_verified = test_hash_ok,
      success = TRUE,
      error_type = "",
      error_message = "",
      stringsAsFactors = FALSE
    )
    list(
      parameters = parameter_table,
      variables = variable_table,
      predictions = prediction_table,
      metrics = metric_table,
      status = status_table
    )
  }, error = function(e) e)
  if (inherits(result, "error")) {
    status_results[[length(status_results) + 1L]] <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      train_seed = manifest$train_seed[i],
      test_seed = manifest$test_seed[i],
      prediction_seed = PREDICTION_MASTER_SEED + rep_id,
      n_train = manifest$n_train[i],
      n_train_selected = NA_integer_,
      n_test = manifest$n_test[i],
      observed_fraction = manifest$train_observed_fraction[i],
      sigma_estimate = NA_real_, rho_estimate = NA_real_,
      sigma12_estimate = NA_real_, return_code = NA_integer_,
      return_message = "", log_likelihood = NA_real_,
      fit_seconds = NA_real_, prediction_seconds = NA_real_,
      total_seconds = proc.time()[["elapsed"]] - total_start,
      train_sha256_verified = FALSE,
      test_sha256_verified = FALSE,
      success = FALSE,
      error_type = class(result)[1],
      error_message = conditionMessage(result),
      stringsAsFactors = FALSE
    )
    cat("  FAILED: ", conditionMessage(result), "\n", sep = "")
    next
  }
  parameter_results[[length(parameter_results) + 1L]] <- result$parameters
  variable_results[[length(variable_results) + 1L]] <- result$variables
  metric_results[[length(metric_results) + 1L]] <- result$metrics
  status_results[[length(status_results) + 1L]] <- result$status
  utils::write.table(
    result$predictions, prediction_connection,
    sep = ",", row.names = FALSE, col.names = FALSE,
    quote = FALSE, na = "NA"
  )
  cat(
    "  selected=", result$status$n_train_selected,
    ", rho=", sprintf("%.4f", result$status$rho_estimate),
    ", sigma=", sprintf("%.4f", result$status$sigma_estimate),
    ", fit=", sprintf("%.3fs", result$status$fit_seconds),
    "\n", sep = ""
  )
}
close(prediction_connection)
parameter_raw <- if (length(parameter_results) > 0L) {
  do.call(rbind, parameter_results)
} else data.frame()
variable_raw <- if (length(variable_results) > 0L) {
  do.call(rbind, variable_results)
} else data.frame()
metric_raw <- if (length(metric_results) > 0L) {
  do.call(rbind, metric_results)
} else data.frame()
status_raw <- do.call(rbind, status_results)
if (nrow(parameter_raw) > 0L) {
  parameter_summary <- do.call(rbind, lapply(
    split(parameter_raw, parameter_raw$parameter),
    function(g) {
      is_comparable <- all(g$comparable_to_dgp)
      data.frame(
        method = METHOD_NAME,
        parameter = g$parameter[1],
        ComparableToDGP = is_comparable,
        Truth = if (is_comparable) {
          g$truth[1]
        } else {
          NA_real_
        },
        Mean = safe_mean(g$estimate),
        Bias = safe_mean(g$bias),
        RMSE = sqrt(safe_mean(g$squared_error)),
        ESD = safe_sd(g$estimate),
        ASD = safe_mean(g$standard_error),
        Coverage = safe_mean(g$covered_95),
        AverageLength = safe_mean(g$interval_length),
        NumberEstimated = sum(is.finite(g$estimate)),
        NumberComparable = sum(is.finite(g$bias)),
        stringsAsFactors = FALSE
      )
    }
  ))
  rownames(parameter_summary) <- NULL
} else {
  parameter_summary <- data.frame()
}
variable_summary <- data.frame(
  method = METHOD_NAME,
  applicable = FALSE,
  note = "Standard Heckman estimates all coefficients; variable-selection metrics are not applicable.",
  stringsAsFactors = FALSE
)
if (nrow(metric_raw) > 0L) {
  metric_names <- setdiff(
    names(metric_raw),
    c("method", "replication", "n_test", "n_test_selected")
  )
  prediction_summary <- do.call(rbind, lapply(metric_names, function(name) {
    x <- metric_raw[[name]]
    n_finite <- sum(is.finite(x))
    data.frame(
      method = METHOD_NAME,
      metric = name,
      Mean = safe_mean(x),
      SD = safe_sd(x),
      MonteCarloSE = if (n_finite >= 2L) safe_sd(x) / sqrt(n_finite) else NA_real_,
      Q025 = safe_quantile(x, 0.025),
      Q975 = safe_quantile(x, 0.975),
      NumberSuccessful = n_finite,
      stringsAsFactors = FALSE
    )
  }))
  rownames(prediction_summary) <- NULL
} else {
  prediction_summary <- data.frame()
}
utils::write.csv(
  parameter_raw,
  file.path(OUTPUT_DIR, "parameter_recovery_by_replication.csv"),
  row.names = FALSE, na = "NA"
)
utils::write.csv(
  parameter_summary,
  file.path(OUTPUT_DIR, "parameter_recovery_summary.csv"),
  row.names = FALSE, na = "NA"
)
utils::write.csv(
  variable_raw,
  file.path(OUTPUT_DIR, "variable_selection_by_replication.csv"),
  row.names = FALSE, na = "NA"
)
utils::write.csv(
  variable_summary,
  file.path(OUTPUT_DIR, "variable_selection_summary.csv"),
  row.names = FALSE, na = "NA"
)
utils::write.csv(
  metric_raw,
  file.path(OUTPUT_DIR, "prediction_metrics_by_replication.csv"),
  row.names = FALSE, na = "NA"
)
utils::write.csv(
  prediction_summary,
  file.path(OUTPUT_DIR, "prediction_summary.csv"),
  row.names = FALSE, na = "NA"
)
utils::write.csv(
  status_raw,
  file.path(OUTPUT_DIR, "replication_status.csv"),
  row.names = FALSE, na = "NA"
)
run_manifest <- list(
  method = METHOD_NAME,
  estimator = "sampleSelection::selection(method='2step')",
  model = paste(
    "Standard Heckman two-step estimator with a valid exclusion variable;",
    "lambda fixed at 1"
  ),
  sampleSelection_version = as.character(
    utils::packageVersion("sampleSelection")
  ),
  software_reference = paste(
    "Toomet, O. and Henningsen, A. (2008). Sample Selection Models in R:",
    "Package sampleSelection. Journal of Statistical Software, 27(7), 1-23."
  ),
  common_data_dir = normalizePath(COMMON_DATA_DIR),
  dataset_manifest_sha256 = sha256_file(manifest_path),
  simulation_config_sha256 = sha256_file(config_path),
  n_replications_requested = nrow(manifest),
  n_replications_successful = sum(status_raw$success),
  pilot_mode = !is.null(REPLICATION_LIMIT),
  lambda_fixed = 1.0,
  n_predictive_draws = N_PREDICTIVE_DRAWS,
  prediction_master_seed = PREDICTION_MASTER_SEED,
  selection_formula = paste(deparse(selection_formula), collapse = ""),
  outcome_formula = paste(deparse(outcome_formula), collapse = ""),
  transformed_scale_metrics = "NA because lambda=1 and lambda_true=0.5 are different scales.",
  variable_selection_metrics = "not applicable",
  two_step_covariance_note = paste(
    "sampleSelection does not provide covariance entries for derived sigma/rho",
    "or cross-stage blocks in the full two-step covariance matrix."
  ),
  prediction_interval_uncertainty = paste(
    "Intervals include separate alpha and beta asymptotic draws plus future",
    "outcome errors; sigma and rho are plug-in estimates."
  ),
  sigma_v2_interval = "not reported because its two-step standard error is unavailable",
  test_data_rule = paste(
    "Only x and w are used to construct predictions;",
    "d, y_star and t_star are used only for evaluation."
  )
)
jsonlite::write_json(
  run_manifest,
  file.path(OUTPUT_DIR, "run_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE, na = "null"
)
cat("\n完成：", sum(status_raw$success), "/", nrow(manifest),
    "组成功。\n", sep = "")
cat("结果文件夹：\n", normalizePath(OUTPUT_DIR), "\n", sep = "")
