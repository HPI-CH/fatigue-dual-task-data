import os
import json
import sys

import pandas as pd
import numpy as np
from scipy.stats import variation

def aggregate_parameters_from_df(df, select_strides=False, onesided=False):

    print(".", end="")
    df.reset_index(inplace=True)
    # load data and filter out all outliers
    df = df[df.is_outlier != 1]  # filter outliers
    if "turning_interval" in df.columns:
        df = df[df.turning_interval != 1]  # filter turning strides
    if "interrupted" in df.columns:
        df = df[df.interrupted != 1]  # filter interrupted strides

    if select_strides:
        # select only first n stride to be aggregated
        df = df.iloc[0:30]

    # get gait parameters from df
    df_param = df.filter(items=[
        'stride_lengths',
        'clearances_min',
        'clearances_max',
        'stride_times',
        'swing_times',
        'stance_times',
        'stance_ratios'
    ])

    # calculate cadence and speed for single foot
    df_param['cadence'] = 120 / df_param['stride_times']  # cadence in (step / min)
    df_param['speed'] = df_param['stride_lengths'] / df_param['stride_times']

    avg_list = df_param.mean().tolist()
    CV_list = variation(df_param, axis=0).tolist()
    aggregate_list = avg_list + CV_list
    aggregate_params = pd.DataFrame(
        columns=[
            'stride_lengths_avg',
            'clearances_min_avg',
            'clearances_max_avg',
            'stride_times_avg',
            'swing_times_avg',
            'stance_times_avg',
            'stance_ratios_avg',
            'cadence_avg',
            'speed_avg',
            'stride_lengths_CV',
            'clearances_min_CV',
            'clearances_max_CV',
            'stride_times_CV',
            'swing_times_CV',
            'stance_times_CV',
            'stance_ratios_CV',
            'cadence_CV',
            'speed_CV',
        ])
    aggregate_params.loc[0] = aggregate_list

    if onesided:
        all_aggregate_params = aggregate_params
    else:
        # calculate symmetry index (SI) if data is not one-sided
        idx_left = df.index[df['foot'] == "left"]
        idx_right = df.index[df['foot'] == "right"]
        avg_list_left = df_param.loc[idx_left].mean().tolist()
        avg_list_right = df_param.loc[idx_right].mean().tolist()

        SI_list = calculate_SI(avg_list_left, avg_list_right)

        SI_df = pd.DataFrame(get_SI_series(SI_list)).transpose()

        all_aggregate_params = pd.concat([aggregate_params, SI_df], axis=1)
    
    # carry over information if they exist in the input dataframe
    if "fatigue" in df.columns:
        all_aggregate_params["fatigue"] = df["fatigue"].values[0]
    if "condition" in df.columns:
        all_aggregate_params["condition"] = df["condition"].values[0]
    if "sub" in df.columns:
        all_aggregate_params["sub"] = df["sub"].values[0]

    return all_aggregate_params


def calculate_SI(avg_list_left, avg_list_right):
    """
    calculate symmetry index
    """
    diff_avg = [abs(j - i) for i, j in zip(avg_list_left, avg_list_right)]
    sum_avg = [sum(x) for x in zip(avg_list_left, avg_list_right)]
    SI_list = [x / (0.5 * y) for x, y in zip(diff_avg, sum_avg)]
    return SI_list


def aggregate_parameters(save_path, prefix="", select_strides=False, save=True):
    """
        aggregate stride by stride gait parameters
        Parameters
        ----------
        save_path : directory to save aggregated results
        prefix : string in front of common file names, e.g. 'group_0_'
        select_strides: Boolean, whether to select a sub-section of the strides for aggregation
        save : Boolean, save file in .csv

        Returns
        -------
        aggregated gait parameters from left-, right foot and both feet

        """

    # aggregate left- and right foot strides separately
    core_params = {}
    aggregate_params = {}
    for side in ['left', 'right']:
        # load stride-by-stride data into dataframe
        try:
            core_params[side] = pd.read_csv(os.path.join(save_path, prefix + side + "_foot_core_params.csv"), index_col=False)
            core_params[side]["foot"] = side
        except FileNotFoundError as e:
            print("Foot core data not found. For file with prefix, please create it first using the cutting function.")
            print(e)
            sys.exit(0)

        aggregate_params[side] = aggregate_parameters_from_df(core_params[side], select_strides=select_strides, onesided=True)

    # aggregate left- and right foot strides together, including symmetry indes (SI)
    core_params_LR = pd.concat(core_params.values(), ignore_index=True)
    aggregate_params_LR = aggregate_parameters_from_df(core_params_LR, select_strides=select_strides, onesided=False)

    if save:
        aggregate_params["left"].to_csv(os.path.join(save_path,
                                                     prefix + "left_foot_aggregate_params.csv"), index=False)
        aggregate_params["right"].to_csv(os.path.join(save_path,
                                                      prefix + "right_foot_aggregate_params.csv"), index=False)
        aggregate_params_LR.to_csv(os.path.join(save_path,
                                              prefix + "aggregate_params.csv"), index=False)
        # print("saved aggregated gait parameters")

    return aggregate_params, aggregate_params_LR


def get_SI_series(SI_list) -> pd.Series:
    return pd.Series(
        data=SI_list,
        index=[
            'stride_lengths_SI',
            'clearances_min_SI',
            'clearances_max_SI',
            'stride_times_SI',
            'swing_times_SI',
            'stance_times_SI',
            'stance_ratios_SI',
            'cadence_SI',
            'speed_SI',
        ]
    )


def main(runs, sub_list, processed_base_path):

    for run in runs:
        for sub in sub_list:

            ### aggregate entire recording session
            save_path = os.path.join(
                processed_base_path,
                run,
                sub
            )
            aggregate_params, aggregate_overall = aggregate_parameters(save_path, save=True)
            print('aggregate ' + run + ', ' + sub)

            # ###  aggregate by 2-min group
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'groups'
            # )
            # for group_num in ['0', '1', '2']:
            #     group_name = 'group_' + group_num + '_'
            #     aggregate_params, aggregate_overall = aggregate_parameters(save_path, group_name, save=True)
            #     print('aggregate ' + run + ', ' + sub + ', group ' + group_num)
            #
            # ### aggregate by first n strides
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub
            # )
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='first_30_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)

            ### aggregate by 50 to end (stable gait speed)
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'stable_gait_speeds'
            # )
            #
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='from_50_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)

            ### aggregate by 50 to 100 (stable gait speed)
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'stable_gait_speeds'
            # )
            #
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='50_up_to_100_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)

            ### aggregate by last 50 (stable gait speed)
            # save_path = os.path.join(
            #     processed_base_path,
            #     run,
            #     sub,
            #     'stable_gait_speeds'
            # )
            #
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # aggregate_params, aggregate_overall = aggregate_parameters(save_path,
            #                                                            prefix='last_50_',
            #                                                            select_strides=True,
            #                                                            save=True)
            # print('aggregate ' + run + ', ' + sub)
