# Fit the Iqbal et al. spike-and-slab benchmark in the primary simulation.

.command_args <- commandArgs(trailingOnly = FALSE)
.file_arg <- .command_args[grepl("^--file=", .command_args)]
.script_dir <- if (length(.file_arg) == 1L) {
  dirname(normalizePath(sub("^--file=", "", .file_arg)))
} else {
  normalizePath(getwd())
}
REPO_ROOT <- normalizePath(file.path(.script_dir, "..", ".."), winslash = "/", mustWork = FALSE)
rm(.command_args, .file_arg, .script_dir)

command_arguments <- commandArgs(trailingOnly = FALSE)
file_argument <- command_arguments[grepl("^--file=", command_arguments)]
SCRIPT_DIR <- if (length(file_argument) == 1L) {
  dirname(normalizePath(sub("^--file=", "", file_argument)))
} else {
  normalizePath(getwd())
}
COMMON_DATA_DIR <- file.path(REPO_ROOT, "data", "simulation")
AUTHOR_ROOT <- file.path(REPO_ROOT, "external", "selection-spike-slab")
OUTPUT_DIR <- file.path(REPO_ROOT, "results", "simulation", "iqbal_spike_slab")
REPLICATION_LIMIT <- NULL
N_MCMC <- 10000L
BURN_IN <- 1250L
N_PREDICTIVE_DRAWS <- 500L
PIP_THRESHOLD <- 0.5
MCMC_MASTER_SEED <- 20261013L
PREDICTION_MASTER_SEED <- 20261014L
METHOD_NAME <- "iqbal_spike_slab_normal_class_I"
ALLOW_OVERWRITE <- FALSE
packages <- c(
  "sampleSelection", "MASS", "truncnorm", "jsonlite",
  "digest", "matrixStats"
)
missing_packages <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "缺少R包：", paste(missing_packages, collapse = ", "),
    "\n请先运行 install.packages() 安装。"
  )
}
suppressPackageStartupMessages({
  library(MASS)
  library(truncnorm)
})
author_sampler <- file.path(AUTHOR_ROOT, "methods", "gibbs_spike_slab.R")
author_utils <- file.path(AUTHOR_ROOT, "methods", "utils.R")
if (!file.exists(author_sampler) || !file.exists(author_utils)) {
  stop("没有找到作者原始 Gibbs sampler。请检查 AUTHOR_ROOT。")
}
old_working_directory <- getwd()
setwd(file.path(AUTHOR_ROOT, "methods"))
source("gibbs_spike_slab.R")
setwd(old_working_directory)
sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}
auc_rank <- function(truth, probability) {
  truth <- as.integer(truth)
  n1 <- sum(truth == 1L)
  n0 <- sum(truth == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  ranks <- rank(probability, ties.method = "average")
  (sum(ranks[truth == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
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
safe_quantile <- function(x, probability) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  as.numeric(stats::quantile(x, probability, names = FALSE))
}
selection_metrics <- function(pip, truth_active) {
  selected <- pip > PIP_THRESHOLD
  truth <- seq_along(pip) %in% truth_active
  tp <- sum(selected & truth)
  fp <- sum(selected & !truth)
  tn <- sum(!selected & !truth)
  fn <- sum(!selected & truth)
  denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- if (denominator > 0) {
    (tp * tn - fp * fn) / denominator
  } else {
    NA_real_
  }
  c(
    TPR = tp / (tp + fn),
    FPR = fp / (fp + tn),
    FDR = if ((tp + fp) > 0) fp / (tp + fp) else 0,
    Specificity = tn / (tn + fp),
    MCC = mcc,
    SelectedSize = sum(selected),
    TMR = as.numeric(all(selected == truth))
  )
}
make_parameter_rows <- function(rep_id, fit, config, p) {
  rho_draws <- fit$other_params[, "rho"]
  sigma_draws <- fit$other_params[, "sigma"]
  sigma12_draws <- rho_draws * sigma_draws
  sigma_v2_draws <- sigma_draws^2 * (1 - rho_draws^2)
  draws <- cbind(
    fit$alpha,
    fit$beta,
    sigma12_draws,
    sigma_v2_draws
  )
  parameter_names <- c(
    paste0("alpha_", 0:p),
    paste0("beta_", 0:p),
    "sigma12",
    "sigma_v2"
  )
  reference_truth <- c(
    config$alpha_true,
    config$beta_true,
    config$sigma12_true,
    config$sigma_v2_true
  )
  comparable_to_dgp <- startsWith(parameter_names, "alpha_")
  rows <- lapply(seq_along(parameter_names), function(j) {
    parameter_draws <- draws[, j]
    estimate <- mean(parameter_draws)
    interval <- stats::quantile(
      parameter_draws,
      c(0.025, 0.975),
      names = FALSE
    )
    is_comparable <- comparable_to_dgp[j]
    data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      parameter = parameter_names[j],
      truth = reference_truth[j],
      comparable_to_dgp = is_comparable,
      estimate = estimate,
      posterior_sd = stats::sd(parameter_draws),
      lower_95 = interval[1],
      upper_95 = interval[2],
      bias = if (is_comparable) {
        estimate - reference_truth[j]
      } else {
        NA_real_
      },
      squared_error = if (is_comparable) {
        (estimate - reference_truth[j])^2
      } else {
        NA_real_
      },
      covered_95 = if (is_comparable) {
        as.integer(
          interval[1] <= reference_truth[j] &&
            reference_truth[j] <= interval[2]
        )
      } else {
        NA_integer_
      },
      interval_length = interval[2] - interval[1],
      stringsAsFactors = FALSE
    )
  })
  rows[[length(rows) + 1L]] <- data.frame(
    method = METHOD_NAME,
    replication = rep_id,
    parameter = "lambda_fixed",
    truth = config$lambda_true,
    comparable_to_dgp = FALSE,
    estimate = 1.0,
    posterior_sd = NA_real_,
    lower_95 = NA_real_,
    upper_95 = NA_real_,
    bias = NA_real_,
    squared_error = NA_real_,
    covered_95 = NA_integer_,
    interval_length = NA_real_,
    stringsAsFactors = FALSE
  )
  do.call(rbind, rows)
}
make_variable_rows <- function(rep_id, fit, p) {
  equations <- list(
    selection = fit$alpha_indicators,
    outcome = fit$beta_indicators
  )
  rows <- lapply(names(equations), function(equation) {
    pip <- colMeans(equations[[equation]])
    metrics <- selection_metrics(pip, truth_active = 1:3)
    row <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      equation = equation,
      applicable = TRUE,
      stringsAsFactors = FALSE
    )
    for (j in seq_len(p)) row[[paste0("PIP_", j)]] <- pip[j]
    for (name in names(metrics)) row[[name]] <- metrics[name]
    row
  })
  do.call(rbind, rows)
}
manifest_path <- file.path(COMMON_DATA_DIR, "dataset_manifest.csv")
config_path <- file.path(COMMON_DATA_DIR, "simulation_config.json")
if (!file.exists(manifest_path) || !file.exists(config_path)) {
  stop("没有找到公共数据清单或模拟设置。")
}
manifest <- utils::read.csv(
  manifest_path, stringsAsFactors = FALSE, check.names = FALSE
)
config <- jsonlite::fromJSON(config_path)
if (!identical(as.character(config$scenario_id), "baseline_valid_exclusion")) {
  stop("公共数据不是有效排除变量基准场景。")
}
if (!identical(
  as.character(config$exclusion_restriction$selection_only_column), "w3"
)) {
  stop("配置没有确认w3为有效排除变量。")
}
if (!is.null(REPLICATION_LIMIT)) {
  manifest <- manifest[seq_len(min(REPLICATION_LIMIT, nrow(manifest))), ]
}
p <- as.integer(config$p_without_intercept)
x_slope_columns <- paste0("x", seq_len(p))
w_slope_columns <- paste0("w", seq_len(p))
x_all_columns <- c("x0", x_slope_columns)
w_all_columns <- c("w0", w_slope_columns)
selection_formula <- stats::as.formula(
  paste("d ~", paste(w_slope_columns, collapse = " + "))
)
outcome_formula <- stats::as.formula(
  paste("y_observed ~", paste(x_slope_columns, collapse = " + "))
)
if (dir.exists(OUTPUT_DIR) && length(list.files(OUTPUT_DIR)) > 0L &&
    !ALLOW_OVERWRITE) {
  stop("结果文件夹不是空的。请更换OUTPUT_DIR，防止覆盖旧结果。")
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
set.seed(MCMC_MASTER_SEED)
mcmc_seeds <- sample.int(.Machine$integer.max, nrow(manifest))
set.seed(PREDICTION_MASTER_SEED)
prediction_seeds <- sample.int(.Machine$integer.max, nrow(manifest))
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
  selection_bias_t = numeric(), selection_bias_y = numeric(),
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
cat("Iqbal normal--normal Class I spike-and-slab 正式运行\n")
cat("重复次数：", nrow(manifest), "\n", sep = "")
cat("MCMC：", N_MCMC, "，burn-in：", BURN_IN, "\n", sep = "")
cat("结果路径：", OUTPUT_DIR, "\n", sep = "")
for (i in seq_len(nrow(manifest))) {
  rep_id <- as.integer(manifest$replication[i])
  total_start <- proc.time()[["elapsed"]]
  warning_messages <- character()
  cat("Replication ", i, "/", nrow(manifest), "\n", sep = "")
  result <- withCallingHandlers(tryCatch({
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
    required_train <- c(
      "row_id", "d", "y_observed", x_all_columns, w_all_columns
    )
    missing_train <- setdiff(required_train, names(train))
    if (length(missing_train) > 0L) {
      stop("训练集缺少列：", paste(missing_train, collapse = ", "))
    }
    x_train <- as.matrix(train[, x_slope_columns, drop = FALSE])
    w_train <- as.matrix(train[, w_slope_columns, drop = FALSE])
    y_train <- train$y_observed
    initial_fit <- tryCatch(
      suppressWarnings(sampleSelection::selection(
        selection = selection_formula,
        outcome = outcome_formula,
        data = train,
        method = "ml"
      )),
      error = function(e) NULL
    )
    initial_value_source <- "zeros_fallback"
    init_alpha <- rep(0, p + 1L)
    init_beta <- rep(0, p + 1L)
    init_covariance <- 0
    init_conditional_variance <- 1
    init_gamma_alpha <- rep(0, p)
    init_gamma_beta <- rep(0, p)
    if (!is.null(initial_fit) && all(is.finite(initial_fit$estimate))) {
      initial_estimate <- initial_fit$estimate
      init_alpha <- initial_estimate[1:(p + 1L)]
      init_beta <- initial_estimate[(p + 2L):(2L * (p + 1L))]
      init_sigma <- initial_estimate[2L * (p + 1L) + 1L]
      init_correlation <- initial_estimate[2L * (p + 1L) + 2L]
      init_covariance <- init_sigma * init_correlation
      init_conditional_variance <- init_sigma^2 - init_covariance^2
      if (!is.finite(init_conditional_variance) ||
          init_conditional_variance <= 1e-8) {
        init_covariance <- 0
        init_conditional_variance <- 1
      }
      initial_summary <- tryCatch(summary(initial_fit)$estimate,
                                  error = function(e) NULL)
      if (!is.null(initial_summary) && ncol(initial_summary) >= 4L) {
        excluded <- c(1L, p + 2L, 2L * p + 3L, 2L * p + 4L)
        p_values <- initial_summary[-excluded, 4L]
        if (length(p_values) == 2L * p && all(is.finite(p_values))) {
          initial_gamma <- as.integer(p_values < 0.05)
          init_gamma_alpha <- initial_gamma[1:p]
          init_gamma_beta <- initial_gamma[(p + 1L):(2L * p)]
        }
      }
      initial_value_source <- "sampleSelection_ml"
    }
    fit_start <- proc.time()[["elapsed"]]
    set.seed(mcmc_seeds[i])
    fit <- gibbs_spike_slab(
      n = N_MCMC,
      y = y_train,
      x_samp = x_train,
      w_samp = w_train,
      model_select = TRUE,
      burn_in = BURN_IN,
      init_alpha = init_alpha,
      init_beta = init_beta,
      init_rho = init_covariance,
      init_var = init_conditional_variance,
      init_gamma_alpha = init_gamma_alpha,
      init_gamma_beta = init_gamma_beta,
      init_r = 0.5,
      rho_param = 5,
      var_params = c(1, 1),
      r_params = c(1, 1),
      alpha_spike = "normal",
      alpha_slab = "normal",
      beta_spike = "normal",
      beta_slab = "normal",
      tau_0_alpha = 1 / sqrt(p * nrow(train)),
      tau_1_alpha = 0.5,
      tau_0_beta = 1 / sqrt(p * nrow(train)),
      tau_1_beta = 0.5,
      var_scaling = FALSE,
      alpha_intercept_sd = 0.5,
      beta_intercept_sd = 0.5,
      fixed_r = FALSE
    )
    fit_seconds <- proc.time()[["elapsed"]] - fit_start
    if (any(!is.finite(fit$alpha)) || any(!is.finite(fit$beta)) ||
        any(!is.finite(fit$other_params))) {
      stop("MCMC输出含非有限值。")
    }
    parameter_table <- make_parameter_rows(rep_id, fit, config, p)
    variable_table <- make_variable_rows(rep_id, fit, p)
    test_hash_ok <- identical(
      tolower(sha256_file(test_path)),
      tolower(manifest$test_sha256[i])
    )
    if (!test_hash_ok) stop("测试文件SHA-256不一致。")
    test <- utils::read.csv(
      test_path, na.strings = "NA", check.names = FALSE
    )
    required_test <- c(
      "row_id", "d", "y_star", "t_star", x_all_columns, w_all_columns
    )
    missing_test <- setdiff(required_test, names(test))
    if (length(missing_test) > 0L) {
      stop("测试集缺少列：", paste(missing_test, collapse = ", "))
    }
    x_test <- as.matrix(test[, x_all_columns, drop = FALSE])
    w_test <- as.matrix(test[, w_all_columns, drop = FALSE])
    prediction_start <- proc.time()[["elapsed"]]
    kept_draws <- nrow(fit$alpha)
    draw_indices <- unique(as.integer(round(seq(
      1, kept_draws, length.out = min(N_PREDICTIVE_DRAWS, kept_draws)
    ))))
    alpha_draws <- fit$alpha[draw_indices, , drop = FALSE]
    beta_draws <- fit$beta[draw_indices, , drop = FALSE]
    rho_draws <- fit$other_params[draw_indices, "rho"]
    sigma_draws <- fit$other_params[draw_indices, "sigma"]
    n_draws <- length(draw_indices)
    n_test <- nrow(test)
    eta_draws <- w_test %*% t(alpha_draws)
    mean_draws <- x_test %*% t(beta_draws)
    selection_probability_draws <- stats::pnorm(eta_draws)
    selection_probability <- rowMeans(selection_probability_draws)
    log_mills <- stats::dnorm(eta_draws, log = TRUE) -
      stats::pnorm(eta_draws, log.p = TRUE)
    mills_draws <- exp(pmin(log_mills, 700))
    sigma12_draws <- rho_draws * sigma_draws
    selected_mean_draws <- mean_draws + sweep(
      mills_draws, 2L, sigma12_draws, "*"
    )
    y_mean_unconditional <- rowMeans(mean_draws)
    y_mean_selected <- rowMeans(selected_mean_draws)
    selection_bias_y <- y_mean_selected - y_mean_unconditional
    set.seed(prediction_seeds[i])
    unconditional_error <- matrix(
      stats::rnorm(n_test * n_draws), nrow = n_test
    )
    y_draws_unconditional <- mean_draws + sweep(
      unconditional_error, 2L, sigma_draws, "*"
    )
    lower_probability <- stats::pnorm(-eta_draws)
    selected_probability <- stats::pnorm(eta_draws)
    uniforms <- matrix(stats::runif(n_test * n_draws), nrow = n_test)
    truncated_probability <- lower_probability +
      uniforms * selected_probability
    truncated_probability <- pmin(
      pmax(truncated_probability, 1e-12), 1 - 1e-12
    )
    epsilon_selected <- stats::qnorm(truncated_probability)
    independent_error <- matrix(
      stats::rnorm(n_test * n_draws), nrow = n_test
    )
    selected_error <- sweep(
      epsilon_selected, 2L, sigma12_draws, "*"
    ) + sweep(
      independent_error, 2L,
      sigma_draws * sqrt(pmax(1 - rho_draws^2, 0)), "*"
    )
    y_draws_selected <- mean_draws + selected_error
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
      t_mean_latent = NA_real_,
      t_mean_selected = NA_real_,
      selection_bias_t = NA_real_,
      selection_bias_y = selection_bias_y,
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
    safe_probability <- pmin(
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
        d_test * log(safe_probability) +
          (1 - d_test) * log(1 - safe_probability)
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
      mcmc_seed = mcmc_seeds[i],
      prediction_seed = prediction_seeds[i],
      n_train = nrow(train),
      n_train_selected = sum(train$d == 1L),
      n_test = nrow(test),
      observed_fraction = mean(train$d),
      initial_value_source = initial_value_source,
      posterior_mean_rho = mean(fit$other_params[, "rho"]),
      posterior_mean_sigma = mean(fit$other_params[, "sigma"]),
      posterior_mean_r = mean(fit$other_params[, "r"]),
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
  }, error = function(e) e), warning = function(w) {
    warning_messages <<- c(warning_messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  if (inherits(result, "error")) {
    status_results[[length(status_results) + 1L]] <- data.frame(
      method = METHOD_NAME,
      replication = rep_id,
      mcmc_seed = mcmc_seeds[i],
      prediction_seed = prediction_seeds[i],
      n_train = manifest$n_train[i],
      n_train_selected = NA_integer_,
      n_test = manifest$n_test[i],
      observed_fraction = manifest$train_observed_fraction[i],
      initial_value_source = NA_character_,
      posterior_mean_rho = NA_real_,
      posterior_mean_sigma = NA_real_,
      posterior_mean_r = NA_real_,
      fit_seconds = NA_real_,
      prediction_seconds = NA_real_,
      total_seconds = proc.time()[["elapsed"]] - total_start,
      train_sha256_verified = FALSE,
      test_sha256_verified = FALSE,
      success = FALSE,
      error_type = class(result)[1],
      error_message = conditionMessage(result),
      warning_count = length(unique(warning_messages)),
      warning_messages = paste(unique(warning_messages), collapse = " | "),
      stringsAsFactors = FALSE
    )
    cat("  FAILED: ", conditionMessage(result), "\n", sep = "")
    next
  }
  result$status$warning_count <- length(unique(warning_messages))
  result$status$warning_messages <- paste(
    unique(warning_messages), collapse = " | "
  )
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
    ", rho=", sprintf("%.4f", result$status$posterior_mean_rho),
    ", sigma=", sprintf("%.4f", result$status$posterior_mean_sigma),
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
parameter_summary <- if (nrow(parameter_raw) > 0L) {
  out <- do.call(rbind, lapply(
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
        ASD = safe_mean(g$posterior_sd),
        Coverage = safe_mean(g$covered_95),
        AverageLength = safe_mean(g$interval_length),
        NumberEstimated = sum(is.finite(g$estimate)),
        NumberComparable = sum(is.finite(g$bias)),
        stringsAsFactors = FALSE
      )
    }
  ))
  rownames(out) <- NULL
  out
} else {
  data.frame()
}
variable_summary <- if (nrow(variable_raw) > 0L) {
  metric_names <- c(
    "TPR", "FPR", "FDR", "Specificity", "MCC", "SelectedSize", "TMR"
  )
  out <- do.call(rbind, lapply(
    split(variable_raw, variable_raw$equation),
    function(g) {
      row <- data.frame(
        method = METHOD_NAME,
        equation = g$equation[1],
        NumberSuccessful = nrow(g),
        stringsAsFactors = FALSE
      )
      for (name in metric_names) row[[name]] <- safe_mean(g[[name]])
      row
    }
  ))
  rownames(out) <- NULL
  out
} else data.frame()
prediction_summary <- if (nrow(metric_raw) > 0L) {
  metric_names <- setdiff(
    names(metric_raw),
    c("method", "replication", "n_test", "n_test_selected")
  )
  out <- do.call(rbind, lapply(metric_names, function(name) {
    values <- metric_raw[[name]]
    n_finite <- sum(is.finite(values))
    data.frame(
      method = METHOD_NAME,
      metric = name,
      Mean = safe_mean(values),
      SD = safe_sd(values),
      MonteCarloSE = if (n_finite >= 2L) {
        safe_sd(values) / sqrt(n_finite)
      } else {
        NA_real_
      },
      Q025 = safe_quantile(values, 0.025),
      Q975 = safe_quantile(values, 0.975),
      NumberSuccessful = n_finite,
      stringsAsFactors = FALSE
    )
  }))
  rownames(out) <- NULL
  out
} else data.frame()
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
  reference = paste(
    "Iqbal, Ogundimu and Rubio (2026), Computational Statistics,",
    "doi:10.1007/s00180-026-01745-3"
  ),
  author_sampler_file = normalizePath(author_sampler),
  author_sampler_sha256 = sha256_file(author_sampler),
  author_utils_sha256 = sha256_file(author_utils),
  author_code_modified = FALSE,
  common_data_dir = normalizePath(COMMON_DATA_DIR),
  dataset_manifest_sha256 = sha256_file(manifest_path),
  simulation_config_sha256 = sha256_file(config_path),
  scenario_id = config$scenario_id,
  n_replications_requested = nrow(manifest),
  n_replications_successful = sum(status_raw$success),
  pilot_mode = !is.null(REPLICATION_LIMIT),
  n_mcmc = N_MCMC,
  burn_in = BURN_IN,
  retained_draws = N_MCMC - BURN_IN,
  n_predictive_draws = N_PREDICTIVE_DRAWS,
  mcmc_master_seed = MCMC_MASTER_SEED,
  prediction_master_seed = PREDICTION_MASTER_SEED,
  prior = list(
    family = "normal-normal Class I",
    var_scaling = FALSE,
    tau_0_alpha = 1 / sqrt(p * config$n_train_per_replication),
    tau_1_alpha = 0.5,
    tau_0_beta = 1 / sqrt(p * config$n_train_per_replication),
    tau_1_beta = 0.5,
    alpha_intercept_sd = 0.5,
    beta_intercept_sd = 0.5,
    beta_binomial = c(1, 1),
    rho_param = 5,
    variance_inverse_gamma = c(1, 1)
  ),
  pip_threshold = PIP_THRESHOLD,
  parameter_point_estimator = "posterior mean",
  transformed_scale_metrics = paste(
    "NA because the comparator has no lambda parameter and its identity outcome",
    "scale is not the true lambda=0.5 transformed scale."
  ),
  fitting_rule = paste(
    "Only y_observed, d, x1:x10 and w1:w10 from each training file",
    "are used for fitting and variable selection."
  ),
  prediction_rule = paste(
    "Only row_id, x0:x10 and w0:w10 enter prediction;",
    "d, y_star and t_star are used only after predictions are fixed."
  ),
  package_versions = lapply(packages, function(package) {
    as.character(utils::packageVersion(package))
  })
)
names(run_manifest$package_versions) <- packages
jsonlite::write_json(
  run_manifest,
  file.path(OUTPUT_DIR, "run_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE, na = "null"
)
cat("\n完成：", sum(status_raw$success), "/", nrow(manifest),
    "组成功。\n", sep = "")
cat("结果文件夹：\n", normalizePath(OUTPUT_DIR), "\n", sep = "")
