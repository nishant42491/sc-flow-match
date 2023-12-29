library(DrImpute)

library(DrImpute)

names_list <- c("klein", "ziesel")
task_list <- c("zero_one_dropout", "zero_two_dropout", "zero_four_dropout")

for (name in names_list) {
  for (task in task_list) {
    # Define file paths
    input_csv <- paste0("D:/Nishant/cfm_impute/flow-matching-single cell/outputs/", name, "/", task, "/corrupted.csv")
    output_csv <- paste0("D:/Nishant/cfm_impute/flow-matching-single cell/baseline_outputs/drimpute/", name, "/", task, "/", name, "_imputed.csv")
    par_dir <- paste0("D:/Nishant/cfm_impute/flow-matching-single cell/baseline_outputs/drimpute/", name, "/", task)
    if (!file.exists(output_csv)) dir.create(par_dir, recursive = TRUE)
    # Load CSV file
    data <- as.matrix(read.csv(input_csv, header = FALSE, sep = ","))
    data <- t(data)

    # Run DrImpute
    data_imputed <- DrImpute(data, k = 10)

    # Save imputed data
    write.csv(data_imputed, output_csv, row.names = FALSE, col.names = FALSE)
  }
}

