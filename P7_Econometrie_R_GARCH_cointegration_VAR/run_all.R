#!/usr/bin/env Rscript
# ============================================================
# run_all.R
# =========
# Lance tous les scripts d'econometrie et regenere les graphiques.
# ============================================================

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grep("^--file=", script_args)]
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  script_dir <- getwd()
}

script_dir <- normalizePath(script_dir, winslash = "/", mustWork = TRUE)
outputs_dir <- normalizePath(file.path(script_dir, "outputs"),
                             winslash = "/", mustWork = FALSE)
dir.create(outputs_dir, showWarnings = FALSE, recursive = TRUE)

scripts <- c(
  "garch.R",
  "cointegration.R",
  "var_model.R"
)

# Chaque script recree ces fichiers. On retire l'ancienne version avant
# l'execution afin de ne jamais conserver une image devenue obsolète.
output_files <- c(
  "garch_diagnostics.png",
  "cointegration.png",
  "var_irf.png"
)
unlink(file.path(outputs_dir, output_files), force = TRUE)

cat("=======================================================\n")
cat(" Execution de tous les scripts R\n")
cat("=======================================================\n\n")

for (script_name in scripts) {
  script_path <- file.path(script_dir, script_name)
  if (!file.exists(script_path)) {
    stop(sprintf("Script introuvable : %s", script_path))
  }

  cat(sprintf("\n>>> Execution de %s\n", script_name))
  result <- system2("Rscript", args = shQuote(script_path))
  if (result != 0) {
    stop(sprintf("Echec pendant l'execution de %s (code %s).",
                script_name, result))
  }
}

missing_outputs <- output_files[!file.exists(file.path(outputs_dir, output_files))]
if (length(missing_outputs) > 0) {
  stop(sprintf("Images non generees : %s", paste(missing_outputs, collapse = ", ")))
}

cat("\n=======================================================\n")
cat("Execution terminee avec succes. Images regenerees dans :\n")
cat(sprintf("%s\n", outputs_dir))
cat("=======================================================\n")
