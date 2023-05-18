
# load libraries and functions
library(ez)
library(dplyr)
library(jsonlite)

options(width = 160)  # print wide table without wrapping

# load data
paths <- fromJSON(txt = "../path.json")
data <- read.csv(
    file.path(
        paths["data"],
        "processed",
        "gait_params_per_person.csv"
    )
)

# add "condition" column to label st or dt in the "run" column values
data$condition <- ifelse(
    grepl("st", data$run),
    "st",
    "dt"
)

# add "fatigue" column to label pre or post in the "run" column values
data$fatigue <- ifelse(
    grepl("control", data$run),
    "control",
    "fatigue"
)

# select list of features / gait parameters
features_list <- c(
    "stride_lengths_avg",
    "speed_avg"
)

# two-way repeated measures ANOVAA
results_2_way_anova <- data.frame()
for (var_name in features_list) {
    dat_df <- select(data, "var" = var_name, sub, condition, fatigue)

    res_aov <- ezANOVA(
        data = dat_df,
        dv = .(var),
        wid = .(sub),
        within = .(condition, fatigue),
        detailed = TRUE,
        type = 2
    )

    anova_res <- data.frame(res_aov)
    anova_res$variable <- var_name
    results_2_way_anova <- dplyr::bind_rows(results_2_way_anova, anova_res)
}
print("ANOVA Summary:")
print(results_2_way_anova)
