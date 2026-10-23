# Runtime inputs and state names; no scientific operations.
CONFIG <- list("methods" = list("GARQ" = list("pt" = "${GARQ_INPUT_ROOT}/inputs/3dc44b753f01_tumor_slingshot_pseudotime.csv"),
"MetaQ" = list("pt" = "${GARQ_INPUT_ROOT}/inputs/bf34f91805c5_metaq_tra_pseudotime.csv"),
"SEACells" = list("pt" = "${GARQ_INPUT_ROOT}/inputs/811bd0d81f8d_sea_tra_pseudotime.csv"),
"MetaCellV2" = list("pt" = "${GARQ_INPUT_ROOT}/inputs/3f188cfb6303_mc_slingshot_pseudotime.csv"),
"SuperCell" = list("pt" = "${GARQ_INPUT_ROOT}/inputs/488e957bbd70_supercell_tra_pseudotime.csv")),
"states" = list("Tumor Epithelial", "Tumor Cells (Dedifferentiated)", "Stressed Tumor (p53+)"))
