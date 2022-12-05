import pandas as pd
import os
import numpy as np
import json
import sys
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from features.aggregate_gait_parameters import aggregate_parameters_from_df


def merge_df_lists(dat_left, dat_right):
    """
    Merges left and right foot dfs and sorts after timestamp column.

    Parameters
    ----------
    dat_left: list of left foot dfs
    dat_right: list of right foot dfs

    Returns: list of merged dfs.
    -------

    """

    merged = [pd.concat([df_left, df_right], axis=0).sort_values(by="timestamps") for df_left, df_right in
              zip(dat_left, dat_right)]
    return merged

def merge_dfs(df_left, df_right):
    merged = pd.concat([df_left, df_right], axis=0).sort_values(by="timestamps")
    return merged


def get_valid_strides(dat_path, sub, fatigue, foot, condition, add_static_data=False, remove_turning_intervals=False):
    """

    Parameters
    ----------
    condition: the condition (st or dt)
    dat_path: the base path of the data folder
    sub: the subject
    fatigue: "fatigue" or "control"
    foot: "left" or "right" foot
    add_static_data: True if subject specific data (i.e., height and leg length) should be added
    add_sub: if sub column should be added

    Returns list of dfs with valid strides
    -------

    """

    # build the path
    path = os.path.join(dat_path, "processed", f'OG_{condition}_{fatigue}', f'{sub}',
                            f'{foot}_foot_core_params.csv')

    # read the data
    df = pd.read_csv(path)

    # insert subject info
    df["sub"] = sub
    df["foot"] = foot

    if add_static_data:
        # read sub_info
        sub_info = pd.read_csv(os.path.join(dat_path, "raw", "subject_info.csv"))
        df["sub_height"] = sub_info[sub_info["sub"] == sub]["height(cm)"].item()
        df["sub_leg_length"] = sub_info[sub_info["sub"] == sub]["leg_length(cm)"].item()

    if fatigue == "fatigue":
        fatigue = 1
    elif fatigue == "control":
        fatigue = 0
    else:
        raise ValueError("Fatigue parameter has to be either fatigue or control.")
    df["fatigue"] = fatigue


    # cut df into the steps until turning steps
    turning_idx = df.index[df['turning_step'] == True].tolist()

    df_list = []
    prev_end = -10
    for idx, turning_idx in enumerate(turning_idx):
        if turning_idx - prev_end == 1:
            prev_end += 1
            continue
        if idx == 0:
            df_list.append(df[:turning_idx])
            prev_end = turning_idx
            continue
        if len(df[
               prev_end + 1:turning_idx]) <= 6:  # drop walking sequences that are too short between the turning steps
            prev_end = turning_idx
            continue
        df_list.append(df[prev_end + 1:turning_idx])
        prev_end = turning_idx


    # drop outliers if present
    dfs_filtered = []
    for df in df_list:
        df = df[df["is_outlier"] == False]  # this also removes na value rows because they have been marked as outliers before
        if remove_turning_intervals:
            df = df[df["turning_interval"] == False]
        dfs_filtered.append(df)

    # df = pd.concat(df_list, axis=0)
    # df.reset_index(inplace=True)
    return dfs_filtered


def get_valid_strides_df(dat_path, sub, fatigue, foot, condition, add_static_data=False, remove_turning_intervals=True):
    # build the path
    path = os.path.join(dat_path, "processed", f'OG_{condition}_{fatigue}', f'{sub}',
                        f'{foot}_foot_core_params.csv')

    # read the data
    df = pd.read_csv(path)

    # insert subject info
    df["sub"] = sub
    df["foot"] = foot

    if add_static_data:
        # read sub_info
        sub_info = pd.read_csv(os.path.join(dat_path, "raw", "subject_info.csv"))
        df["sub_height"] = sub_info[sub_info["sub"] == sub]["height(cm)"].item()
        df["sub_leg_length"] = sub_info[sub_info["sub"] == sub]["leg_length(cm)"].item()

    df["fatigue"] = fatigue
    df["condition"] = condition

    # drop outliers if present
    df = df[
        df["is_outlier"] == False]  # this also removes na value rows because they have been marked as outliers before
    if remove_turning_intervals:
        df = df[df["turning_interval"] == False]
    return df


def construct_windows(df, window_sz=10, window_slide=2):
    """

    Parameters
    ----------
    df: the dataframe
    window_sz: the window size

    Returns list of windowed dataframes
    -------

    """
    windowed_dfs = []
    start_idx = 0
    end_idx = window_sz
    while end_idx <= len(df) - 1:
        windowed_dfs.append(df[start_idx:end_idx])
        start_idx += window_slide
        end_idx += window_slide

    return windowed_dfs


def build_features(sub_list, base_path, test: list, conditions: list, window_sz, window_slide,
                   aggregate_windows=False, add_static_features=True, save_unwindowed_df=True):
    df_list = []
    for sub in sub_list:
        for fatigue in test:
            for cond in conditions:
                df_left = get_valid_strides_df(base_path, sub=sub, foot="left", fatigue=fatigue, condition=cond, add_static_data=True,
                                             remove_turning_intervals=True)
                df_right = get_valid_strides_df(base_path, sub=sub, foot="right", fatigue=fatigue, condition=cond, add_static_data=True,
                                             remove_turning_intervals=True)
                merged = merge_dfs(df_left, df_right)
                merged['fatigue'] = fatigue
                merged['condition'] = cond
                df_list.append(merged)

    if save_unwindowed_df:
        df_all = pd.concat(df_list)
        if add_static_features:
            df_static = pd.read_csv(os.path.join(base_path, "raw", "subject_info.csv"))
            df_all = pd.merge(df_all, df_static, on="sub")
            save_feature_path = os.path.join(base_path, "processed", "features")
            if not os.path.exists(save_feature_path):
                os.makedirs(save_feature_path)
            df_all.to_csv(os.path.join(save_feature_path, "df_all.csv"), index=False)

    # construct windows of the sequences
    dat = []
    for df in df_list:
        windows = construct_windows(df, window_sz=window_sz, window_slide=window_slide)
        dat.extend(windows)
    # aggregate parameters and save it to csv for other methods such as SVM
    if aggregate_windows:
        agg_dat = [aggregate_parameters_from_df(df) for df in dat]
        all_windows_df = pd.concat(agg_dat)
        all_windows_df.reset_index(drop=True, inplace=True)
        all_windows_df.dropna(inplace=True)  # in case SI parameters are NaN
        all_windows_df.drop(all_windows_df[all_windows_df.clearances_min_CV > 50000].index, inplace=True)  # remove the extreme outlier

        # # remove outliers based on z-score
        # z_idx = np.ones(all_windows_df.shape[0], dtype=bool)
        # for parameter in all_windows_df.columns[0:27]:
        #     z_threshold = 3
        #     z_scores_bool = np.array(np.abs(zscore(all_windows_df[parameter])) < z_threshold, dtype=bool)
        #     z_idx = z_idx & z_scores_bool  # keep removing outliers for each parameters
        # all_windows_df = all_windows_df.loc[z_idx]

        if add_static_features:
            sub_info_df = pd.read_csv(os.path.join(base_path, "raw", "subject_info.csv"))
            sub_info_df = sub_info_df.filter(items=[
                "sub",
                "sex",
                "age",
                "height(cm)",
                "leg_length(cm)",
                "weight(kg)"
            ])
            all_windows_df = pd.merge(all_windows_df, sub_info_df, on="sub")

        # build file name based on parameters
        test_names = "_".join(test)
        condition_names = "_".join(conditions)
        fname = f"agg_windows_{test_names}_{condition_names}_{window_sz}_{window_slide}.csv"
        path = os.path.join(base_path, "processed", "features", fname)
        all_windows_df.to_csv(path, index=False)
        print(f"Saved aggregated data to {path}, terminating program...")
        sys.exit()


if __name__ == '__main__':

    ### PARAMS ###
    sub_list = ["sub_01", "sub_02", "sub_03", "sub_05", "sub_06", "sub_07", "sub_08", "sub_09", "sub_10",
                "sub_11", "sub_12", "sub_13", "sub_14", "sub_15", "sub_17", "sub_18"]
    test = ["control", "fatigue"]  # "control", "fatigue"
    conditions = ["st", "dt"]  # "st", "dt"
    with open(os.path.dirname(__file__) + '/../../path.json') as f:
        paths = json.load(f)
    base_path = paths["data_pub"]

    window_sz = 10
    window_slide = 2

    build_features(sub_list, base_path, test, conditions, window_sz, window_slide, 
                   aggregate_windows=True, add_static_features=True, save_unwindowed_df=True)
