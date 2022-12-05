import json
import os
import pandas as pd
import numpy as np

dataset = 'fatigue_dual_task'
with open('../path.json') as f:
    paths = json.load(f)
processed_base_path = paths['processed_data']
interim_base_path = paths['interim_data']

save_folder = os.path.join(processed_base_path, 'dt_costs')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

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

dt_cost_summary_ls = []
for run in ['control', 'fatigue']: 
    arr_st_list = []
    arr_dt_list = []
    dt_costs_ls = []
    for sub in sub_list:
        arr_st_LR_list = []
        arr_dt_LR_list = []
        for foot in ['left', 'right']:
            st_path = os.path.join(
                processed_base_path,
                'OG_st_' + run,
                sub,
                foot + '_foot_aggregate_params.csv'
            )

            dt_path = os.path.join(
                processed_base_path,
                'OG_dt_' + run,
                sub,
                foot + '_foot_aggregate_params.csv'
            )

            df_st = pd.read_csv(st_path)
            df_dt = pd.read_csv(dt_path)

            col_names = df_st.columns.values

            arr_st = df_st.to_numpy().flatten()
            arr_dt = df_dt.to_numpy().flatten()

            costs = (arr_st - arr_dt)/arr_st*100
            df_costs = pd.DataFrame(columns=col_names)
            df_costs.loc[0] = costs
            df_costs['sub'] = sub
            df_costs['cond'] = run

            file_path = os.path.join(save_folder, foot + '.csv')
            if not os.path.isfile(file_path):
                df_costs.to_csv(file_path, index=False)
            else:
                df_costs.to_csv(file_path, mode='a', header=False, index=False)


        LR_st_path = os.path.join(
            processed_base_path,
            'OG_st_' + run,
            sub,
            'aggregate_params.csv'
        )

        LR_dt_path = os.path.join(
            processed_base_path,
            'OG_dt_' + run,
            sub,
            'aggregate_params.csv'
        )

        LR_df_st = pd.read_csv(LR_st_path)
        LR_df_dt = pd.read_csv(LR_dt_path)

        LR_col_names = LR_df_st.columns.values

        LR_arr_st = LR_df_st.to_numpy().flatten()
        LR_arr_dt = LR_df_dt.to_numpy().flatten()

        LR_costs = (LR_arr_st - LR_arr_dt) / LR_arr_st * 100
        LR_df_costs = pd.DataFrame(columns=LR_col_names)
        LR_df_costs.loc[0] = LR_costs
        LR_df_costs['sub'] = sub
        LR_df_costs['cond'] = run

        dt_costs_ls.append(LR_df_costs)

    all_dt_costs_df = pd.concat(dt_costs_ls)

    # save mean for selected gait parameters
    parameter_list = [
        "stride_lengths_avg",
        "stride_lengths_CV",
        "stride_lengths_SI",
        "stride_times_avg",
        "stride_times_CV",
        "stride_times_SI",
        "speed_avg",
        "speed_CV",
        "speed_SI"
    ]
    
    params_df = all_dt_costs_df[parameter_list].copy()
    means = params_df.mean().to_frame()
    means.rename({0: 'mean'}, axis=1, inplace=True)
    std = params_df.std().to_frame()
    std.rename({0: 'std'}, axis=1, inplace=True)
    dt_cost_summary = pd.concat([means, std], axis=1)
    dt_cost_summary = dt_cost_summary.round(3)
    dt_cost_summary['mean_std'] = '$' + dt_cost_summary['mean'].astype(str) + ' \pm ' + dt_cost_summary['std'].astype(str) + '$'
    dt_cost_summary["cond"] = run
    dt_cost_summary_ls.append(dt_cost_summary)

all_dt_cost_summary = pd.concat(dt_cost_summary_ls)  # concat control and fatigue

# reformat the gait paramter names for publication
all_dt_cost_summary.index = all_dt_cost_summary.index.str.title()
all_dt_cost_summary.index = all_dt_cost_summary.index.str.replace('_',' ',regex=True)
all_dt_cost_summary.index = all_dt_cost_summary.index.str.replace('Cv','CV',regex=True)
all_dt_cost_summary.index = all_dt_cost_summary.index.str.replace('Si','SI',regex=True)
all_dt_cost_summary.to_csv(os.path.join(save_folder, 'LR_dt_cost_summary.csv'))
