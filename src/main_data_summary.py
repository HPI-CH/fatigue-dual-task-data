"""
Summarize data.
"""

import os
import json
import pandas as pd
import numpy as np
from data.SubjectInfo import *

with open("path.json") as f:
    paths = json.load(f)
processed_base_path = paths["processed_data"]

runs = [
    "OG_st_control",
    "OG_dt_control",
    "OG_st_fatigue",
    "OG_dt_fatigue" 
]

sub_list = [
    "sub_01",
    "sub_02",
    "sub_03",
    "sub_05",
    "sub_06",
    "sub_07",
    "sub_08",
    "sub_09",
    "sub_10",
    "sub_11",
    "sub_12",
    "sub_13",
    "sub_14",
    "sub_15",
    "sub_17",
    "sub_18"
]

parameter_list = [
    "stride_lengths_avg",
    "speed_avg"
]

#### gait parameters summary ####
df_list = []
summary_data = {}
for run in runs:
    for sub in sub_list:
        folder_path = os.path.join(
            processed_base_path,
            run,
            sub
        )
        stride_count = 0
        for foot in ["left", "right"]:
            core_df = pd.read_csv(os.path.join(folder_path, f"{foot}_foot_core_params.csv"))
            valid_df = core_df[core_df["is_outlier"] == False]
            stride_count += valid_df.shape[0]  # count number of valid strides
        agg_df = pd.read_csv(os.path.join(folder_path, "aggregate_params.csv"))
        agg_df['run'] = run
        agg_df['stride_count_avg'] = stride_count
        df_list.append(agg_df)

all_agg_df = pd.concat(df_list)[parameter_list + ["run", "stride_count_avg"]]
means = all_agg_df.groupby('run', sort=False).mean().round(2)
std = all_agg_df.groupby('run', sort=False).std().round(2)
std.columns = std.columns.str.replace("avg", "std")  # replace column names to std

# construct summary dataframe
summary_df = pd.merge(means, std, left_index=True, right_index=True)
summary_df.sort_index(axis=1, inplace=True)

print("Gait parameter summary:")
print(summary_df.to_string())


#### dual-task costs ####
# calculate dt costs for control and fatigue separately
dt_costs_ls = []
for condition in ["control", "fatigue"]:
    arr_st_list = []
    arr_dt_list = []
    for sub in sub_list:
        st_path = os.path.join(
                processed_base_path,
                "OG_st_" + condition,
                sub,
                "aggregate_params.csv"
            )

        dt_path = os.path.join(
            processed_base_path,
            "OG_dt_" + condition,
            sub,
            "aggregate_params.csv"
        )

        # read gait parameters and flatten into arrays
        arr_st = pd.read_csv(st_path)[parameter_list].to_numpy().flatten()
        arr_dt = pd.read_csv(dt_path)[parameter_list].to_numpy().flatten()

        costs = (arr_st - arr_dt)/arr_st*100
        df_costs = pd.DataFrame(columns=parameter_list)
        df_costs.loc[0] = costs
        df_costs['sub'] = sub
        df_costs['cond'] = condition
        dt_costs_ls.append(df_costs)

all_dt_costs_df = pd.concat(dt_costs_ls)
mean_dt_costs = all_dt_costs_df.groupby("cond").mean().round(2).values
std_dt_costs = all_dt_costs_df.groupby("cond").std().round(2).values

# construct summary dataframe
summary_dt_costs_df = pd.DataFrame(
    np.hstack((mean_dt_costs, std_dt_costs)),   # stack horizontally
    columns=parameter_list + [s + "_std" for s in parameter_list],
    index=["control", "fatigue"]
    )
summary_dt_costs_df.sort_index(axis=1, inplace=True)

print("\nDual-task cost (%) summary:")
print(summary_dt_costs_df.to_string())


#### study information ####
SubjectInfo.main(statistics=True)
