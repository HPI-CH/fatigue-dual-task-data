import json
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
)  # src folder
if module_path not in sys.path:
    sys.path.append(module_path)
from data.imu import IMU

"""
Functions to select a subset of the strides recorded in each 6-minute session.
"""


def mark_turning_interval(df, interval_size):
    """
    Parameters
    ----------
    df : pandas DataFrame containing core gait parameters from each foot
    interval_size : num. of strides to be identified as 'turning interval' surrounding the explicit turning steps

    Returns
    -------
    pandas DataFrame with added 'turning_interval' column to mark a larger selection of turning steps
    """
    ts = df.index[df["turning_step"] == True].tolist()
    turning_idx = []
    for x in ts:
        turning_idx.extend(np.arange(x - interval_size, x + interval_size + 1))

    turning_idx = np.array(sorted(set(turning_idx)))
    # select only indices within all strides
    all_idx = np.array(range(df.shape[0]))
    turning_idx = turning_idx[np.isin(turning_idx, all_idx)]

    df["turning_interval"] = False
    df.loc[turning_idx, "turning_interval"] = True

    # remove first and last strides in the session as well
    head_tail_slice = list(range(interval_size)) + list(range(-1 * interval_size, 0))
    df.iloc[head_tail_slice, df.columns.get_loc("turning_interval")] = True

    return df


def mark_interrupted_strides(df, sub, run, interrupt_path):
    """
    Mark strides where the participant was interrupted while walking

    Parameters
    ----------
    df : pandas DataFrame containing core gait parameters
    sub: subject number
    run: run (recording session) number
    interrupt_path : path to interruption time range csv file

    Returns
    -------

    """
    interrupt_df = pd.read_csv(interrupt_path)
    df["interrupted"] = False

    for index, row in interrupt_df.iterrows():
        if np.logical_and(
            interrupt_df.loc[index, "sub"] == sub,
            interrupt_df.loc[index, "run"]
            == run[3:],  # remove prefix 'OG_' from column name
        ):
            # mark interrupted strides within the manually documented time interval
            true_idx = df[
                np.logical_and(
                    df["timestamps"] >= interrupt_df.loc[index, "start(s)"],
                    df["timestamps"] <= interrupt_df.loc[index, "end(s)"],
                )
            ].index
            df.loc[true_idx, "interrupted"] = True

    return df


def mark_processed_data(runs, sub_list, processed_base_path, interim_base_path):
    for run in runs:
        for sub in sub_list:
            for foot in ["left", "right"]:  # comment out for cut_by_stride
                folder_path = os.path.join(processed_base_path, run, sub)
                df_path = os.path.join(folder_path, foot + "_foot_core_params.csv")
                params_df = pd.read_csv(df_path)
                ### mark turning intervals
                params_df = mark_turning_interval(
                    params_df, 2
                )  # add 'turning interval' column

                ## mark strides that are interrupted during the recording
                params_df = mark_interrupted_strides(
                    params_df,
                    sub,
                    run,
                    os.path.join(interim_base_path, "interruptions.csv"),
                )

                # save updated df
                print(f"Marked processed data from {df_path}.")
                params_df.to_csv(df_path, index=False)


if __name__ == "__main__":
    # params
    # dataset = 'fatigue_dual_task'
    # with open('../../path.json') as f:
    with open(os.path.dirname(__file__) + "/../../path.json") as f:
        paths = json.load(f)
    processed_base_path = paths["processed_pub"]
    interim_base_path = paths["interim_pub"]

    runs = [
        # 'OG_st_control',
        # 'OG_st_fatigue',
        # 'OG_dt_control',
        "OG_dt_fatigue",
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
        "sub_18",
    ]

    mark_processed_data(runs, sub_list, processed_base_path, interim_base_path)
