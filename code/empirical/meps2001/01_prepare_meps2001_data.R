# Prepare the MEPS2001 data and construct both analysis specifications.

.command_args <- commandArgs(trailingOnly = FALSE)
.file_arg <- .command_args[grepl("^--file=", .command_args)]
.script_dir <- if (length(.file_arg) == 1L) {
  dirname(normalizePath(sub("^--file=", "", .file_arg)))
} else {
  normalizePath(getwd())
}
REPO_ROOT <- normalizePath(file.path(.script_dir, "..", "..", ".."), winslash = "/", mustWork = FALSE)
rm(.command_args, .file_arg, .script_dir)

OUTPUT_DIR <- file.path(REPO_ROOT, "data", "processed", "meps2001")
needed <- c("ssmrob", "jsonlite", "digest")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(
    "请先安装：", paste(missing, collapse = ", "),
    "\n运行 install.packages(c(\"ssmrob\", \"jsonlite\", \"digest\"))"
  )
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
data("MEPS2001", package = "ssmrob")
needed_columns <- c(
  "ambexp", "dambexp", "age", "female", "educ",
  "blhisp", "totchr", "ins", "income"
)
if (!all(needed_columns %in% names(MEPS2001))) {
  stop("MEPS2001 的列名与程序要求不一致。")
}
dat <- data.frame(
  row_id = seq_len(nrow(MEPS2001)),
  ambexp = as.numeric(MEPS2001$ambexp),
  d_original = as.integer(MEPS2001$dambexp),
  age = as.numeric(MEPS2001$age),
  female = as.integer(MEPS2001$female),
  educ = as.numeric(MEPS2001$educ),
  blhisp = as.integer(MEPS2001$blhisp),
  totchr = as.numeric(MEPS2001$totchr),
  ins = as.integer(MEPS2001$ins),
  income = as.numeric(MEPS2001$income)
)
if (anyNA(dat)) stop("必要变量中存在缺失值。")
if (any(dat$ambexp < 0)) stop("ambexp 中出现负值。")
dat$d <- as.integer(dat$ambexp > 0)
if (!all(dat$d_original == dat$d)) {
  stop("dambexp 与 I(ambexp > 0) 不完全一致。")
}
dat$y_observed <- ifelse(dat$d == 1, dat$ambexp, NA_real_)
n_total <- nrow(dat)
n_zero <- sum(dat$d == 0)
n_positive <- sum(dat$d == 1)
if (n_total != 3328) stop("总样本量不是 3328。")
if (n_zero != 526) stop("零支出人数不是 526。")
if (n_positive != 2802) stop("正支出人数不是 2802。")
if (any(dat$y_observed[dat$d == 1] <= 0)) stop("d=1 中存在非正支出。")
continuous <- c("age", "educ", "totchr", "income")
scaling <- data.frame(
  variable = continuous,
  method = "(value - mean) / sample SD",
  center = vapply(dat[continuous], mean, numeric(1)),
  scale = vapply(dat[continuous], sd, numeric(1)),
  minimum = vapply(dat[continuous], min, numeric(1)),
  maximum = vapply(dat[continuous], max, numeric(1)),
  row.names = NULL
)
for (j in seq_along(continuous)) {
  variable <- continuous[j]
  dat[[paste0(variable, "_z")]] <-
    (dat[[variable]] - scaling$center[j]) / scaling$scale[j]
}
positive_ambexp <- dat$ambexp[dat$d == 1]
ambexp_geometric_mean <- exp(mean(log(positive_ambexp)))
dat$y_scaled <- ifelse(
  dat$d == 1,
  dat$ambexp / ambexp_geometric_mean,
  NA_real_
)
scaling <- rbind(
  scaling,
  data.frame(
    variable = "ambexp_positive",
    method = "positive ambexp / geometric mean",
    center = NA_real_,
    scale = ambexp_geometric_mean,
    minimum = min(positive_ambexp),
    maximum = max(positive_ambexp)
  )
)
common <- data.frame(
  row_id = dat$row_id,
  d = dat$d,
  ambexp = dat$ambexp,
  y_observed = dat$y_observed,
  y_scaled = dat$y_scaled
)
X <- data.frame(
  x0 = 1,
  x_age = dat$age_z,
  x_female = dat$female,
  x_educ = dat$educ_z,
  x_blhisp = dat$blhisp,
  x_totchr = dat$totchr_z,
  x_ins = dat$ins
)
W <- data.frame(
  w0 = 1,
  w_age = dat$age_z,
  w_female = dat$female,
  w_educ = dat$educ_z,
  w_blhisp = dat$blhisp,
  w_totchr = dat$totchr_z,
  w_ins = dat$ins
)
design_with_exclusion <- cbind(common, X, W, w_income = dat$income_z)
design_no_exclusion <- cbind(common, X, W)
summary_variables <- c(
  "ambexp", "d", "age", "female", "educ",
  "blhisp", "totchr", "ins", "income"
)
data_summary <- data.frame(
  variable = c("sample", summary_variables),
  n = c(n_total, vapply(dat[summary_variables], length, integer(1))),
  missing = c(0, vapply(dat[summary_variables], function(x) sum(is.na(x)), integer(1))),
  mean = c(NA, vapply(dat[summary_variables], mean, numeric(1))),
  sd = c(NA, vapply(dat[summary_variables], sd, numeric(1))),
  minimum = c(NA, vapply(dat[summary_variables], min, numeric(1))),
  maximum = c(NA, vapply(dat[summary_variables], max, numeric(1))),
  note = c(
    paste0("zero expenditure=", n_zero,
           "; positive expenditure=", n_positive),
    rep("original scale", length(summary_variables))
  ),
  row.names = NULL
)
variable_dictionary <- data.frame(
  source_column = c(
    "ambexp", "dambexp", "age", "female", "educ",
    "blhisp", "totchr", "ins", "income"
  ),
  model_role = c(
    "outcome", "selection indicator", "both equations",
    "both equations", "both equations", "both equations",
    "both equations", "both equations", "selection equation only"
  ),
  definition = c(
    "Original ambulatory expenditure; not log transformed",
    "1 if ambexp > 0; 0 otherwise",
    "Age", "Female indicator", "Years of education",
    "Black or Hispanic indicator", "Number of chronic conditions",
    "Insurance indicator", "Income; candidate exclusion variable"
  ),
  processing = c(
    "retain original and divide positive values by geometric mean",
    "convert to integer 0/1 and verify against ambexp",
    "standardize", "retain 0/1", "standardize",
    "retain 0/1", "standardize", "retain 0/1", "standardize"
  )
)
files <- c(
  clean = file.path(OUTPUT_DIR, "meps2001_clean.csv"),
  with_exclusion = file.path(OUTPUT_DIR, "meps2001_design_with_exclusion.csv"),
  no_exclusion = file.path(OUTPUT_DIR, "meps2001_design_no_exclusion.csv"),
  summary = file.path(OUTPUT_DIR, "meps2001_data_summary.csv"),
  dictionary = file.path(OUTPUT_DIR, "meps2001_variable_dictionary.csv"),
  scaling = file.path(OUTPUT_DIR, "meps2001_scaling_parameters.csv")
)
write.csv(dat, files["clean"], row.names = FALSE, na = "")
write.csv(design_with_exclusion, files["with_exclusion"], row.names = FALSE, na = "")
write.csv(design_no_exclusion, files["no_exclusion"], row.names = FALSE, na = "")
write.csv(data_summary, files["summary"], row.names = FALSE, na = "")
write.csv(variable_dictionary, files["dictionary"], row.names = FALSE, na = "")
write.csv(scaling, files["scaling"], row.names = FALSE, na = "")
file_hashes <- lapply(files, function(x) {
  digest::digest(file = x, algo = "sha256", serialize = FALSE)
})
manifest <- list(
  script = "prepare_meps2001_data.R",
  model_fitting_performed = FALSE,
  mcmc_performed = FALSE,
  source = list(
    dataset = "ssmrob::MEPS2001",
    package_version = as.character(utils::packageVersion("ssmrob")),
    survey = "2001 Medical Expenditure Panel Survey"
  ),
  checks = list(
    total = n_total,
    zero = n_zero,
    positive = n_positive,
    positive_outcomes_strictly_positive = TRUE,
    dambexp_matches_ambexp = TRUE
  ),
  outcome = list(
    original = "ambexp",
    observed = "y_observed",
    scaled = "y_scaled",
    geometric_mean = ambexp_geometric_mean,
    unused_precomputed_columns = c("lambexp", "lnambx")
  ),
  with_exclusion = list(
    X = c("age", "female", "educ", "blhisp", "totchr", "ins"),
    W = c("age", "female", "educ", "blhisp", "totchr", "ins", "income"),
    exclusion_variable = "income"
  ),
  no_exclusion = list(
    X = c("age", "female", "educ", "blhisp", "totchr", "ins"),
    W = c("age", "female", "educ", "blhisp", "totchr", "ins")
  ),
  clean_data_sha256 = digest::digest(dat, algo = "sha256", serialize = TRUE),
  output_file_sha256 = file_hashes,
  r_version = R.version.string
)
jsonlite::write_json(
  manifest,
  file.path(OUTPUT_DIR, "data_manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = NA,
  null = "null"
)
cat("MEPS2001 preparation completed.\n")
cat("  total:", n_total, "\n")
cat("  zero expenditures:", n_zero, "\n")
cat("  positive expenditures:", n_positive, "\n")
cat("  geometric mean:", ambexp_geometric_mean, "\n")
cat("  output:", normalizePath(OUTPUT_DIR, winslash = "\\", mustWork = FALSE), "\n")
cat("  no model and no MCMC were run.\n")
