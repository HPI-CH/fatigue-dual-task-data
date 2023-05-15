"""
Summarize data.
"""

import os
import json
import pandas as pd
import numpy as np
from data.SubjectInfo import *
from data.imu import IMU

class DataSummary():
    def __init__(self, sub_list, run_list, parameter_list):
        self.sub_list = sub_list
        self.run_list = run_list
        self.parameter_list = parameter_list

        # get paths to data
        with open("path.json") as f:
            paths = json.load(f)
        self.raw_base_path = paths["raw_data"]
        self.interim_base_path = paths["interim_data"]
        self.processed_base_path = paths["processed_data"]

    #### IMU raw data summary ####
    def append_imu_to_df(self, df, sub, run, imu_loc, imu):
        """append imu statistical data to dataframe at the last row

        Args:
            df (DataFrame): the dataframe to be appended
            imu (IMU object): the imu whose information is to be appended
        """
        # get duration of recording
        duration = imu.time()[-1]  # time is zero-based already when reading the data
        # get acceleration magnitude
        imu.acc_to_meter_per_square_sec()
        acc_mag = imu.accel_mag()
        # get gyro magnitude
        gyro_mag = imu.gyro_mag()

        # append data to dataframe
        df.loc[len(df)] = [
            sub,
            imu_loc,
            run,
            duration,
            acc_mag.mean(),
            gyro_mag.mean(),
        ]
    
    def imu_raw_data_summary(self):
        # load imu data from the 4 walks and the fatigue exercise
        cols = ["sub", "imu_loc", "run", "duration", "acc_mag_mean", "gyro_mag_mean"]
        walk_imu_df = pd.DataFrame(columns=cols)
        exercise_imu_df = pd.DataFrame(columns=cols)
        entire_session_imu_df = pd.DataFrame(columns=cols)

        for sub in self.sub_list:
            for location in ["LF", "SA"]:

                # load walking data
                for run in self.run_list:
                    imu_walk = IMU(os.path.join(self.interim_base_path, run, sub, f"{location}.csv"))
                    self.append_imu_to_df(walk_imu_df, sub, run, location, imu_walk)

                for cond in ["st", "dt"]:
                    # load exercise data
                    imu_exercise = IMU(
                        os.path.join(
                            self.interim_base_path, f"OG_{cond}_sit_to_stand", sub, f"{location}.csv"
                        )
                    )
                    self.append_imu_to_df(
                        exercise_imu_df, sub, f"{cond}_exercise", location, imu_exercise
                    )

                    # load entire recording session data
                    imu_all = IMU(
                        os.path.join(
                            self.interim_base_path, f"OG_{cond}_all", sub, f"{location}.csv"
                        )
                    )
                    self.append_imu_to_df(entire_session_imu_df, sub, cond, location, imu_all)

        # concat all dataframes and save to csv
        imu_stats_df = pd.concat([walk_imu_df, exercise_imu_df, entire_session_imu_df])
        imu_stats_df.to_csv(os.path.join(self.processed_base_path, "imu_stats.csv"), index=False)

        # aggregate imu data across participants
        imu_stats_mean_df = (
            imu_stats_df.groupby(["run", "imu_loc"], sort=False)
            .mean()
            .round(2)
            .add_suffix("_mean")
            .reset_index()
        )
        imu_stats_std_df = (
            imu_stats_df.groupby(["run", "imu_loc"], sort=False)
            .std()
            .round(2)
            .add_suffix("_std")
            .reset_index()
        )

        # merge mean and std dataframes
        imu_stats_summary_df = pd.merge(
            imu_stats_mean_df, imu_stats_std_df, on=["run", "imu_loc"], how="inner"
        )
        # sort columns
        imu_stats_summary_df = imu_stats_summary_df[
            [
                "run",
                "imu_loc",
                "duration_mean",
                "duration_std",
                "acc_mag_mean_mean",
                "acc_mag_mean_std",
                "gyro_mag_mean_mean",
                "gyro_mag_mean_std",
            ]
        ]

        # save to csv
        imu_stats_summary_df.to_csv(
            os.path.join(self.processed_base_path, "imu_stats_summary.csv"), index=False
        )

        # print to console
        print("IMU raw data summary:")
        print(imu_stats_summary_df.to_string(index=False))

    #### gait parameters summary ####
    def gait_params_summary(self):
        df_list = []
        summary_data = {}
        for run in self.run_list:
            for sub in self.sub_list:
                folder_path = os.path.join(self.processed_base_path, run, sub)
                stride_count = 0
                for foot in ["left", "right"]:
                    core_df = pd.read_csv(
                        os.path.join(folder_path, f"{foot}_foot_core_params.csv")
                    )
                    valid_df = core_df[core_df["is_outlier"] == False]
                    stride_count += valid_df.shape[0]  # count number of valid strides
                agg_df = pd.read_csv(os.path.join(folder_path, "aggregate_params.csv"))
                agg_df["run"] = run
                agg_df["stride_count_avg"] = stride_count
                df_list.append(agg_df)

        all_agg_df = pd.concat(df_list)[self.parameter_list + ["run", "stride_count_avg"]]
        means = all_agg_df.groupby("run", sort=False).mean().round(2)
        std = all_agg_df.groupby("run", sort=False).std().round(2)
        std.columns = std.columns.str.replace("avg", "std")  # replace column names to std

        # construct summary dataframe
        summary_df = pd.merge(means, std, left_index=True, right_index=True)
        summary_df.sort_index(axis=1, inplace=True)

        # save summary dataframe to csv
        summary_df.to_csv(os.path.join(self.processed_base_path, "gait_params_summary.csv"))

        print("Gait parameter summary:")
        print(summary_df.to_string())


    #### dual-task costs ####
    # calculate dt costs for control and fatigue separately
    def dt_costs(self):
        dt_costs_ls = []
        for condition in ["control", "fatigue"]:
            arr_st_list = []
            arr_dt_list = []
            for sub in self.sub_list:
                st_path = os.path.join(
                    self.processed_base_path, "OG_st_" + condition, sub, "aggregate_params.csv"
                )

                dt_path = os.path.join(
                    self.processed_base_path, "OG_dt_" + condition, sub, "aggregate_params.csv"
                )

                # read gait parameters and flatten into arrays
                arr_st = pd.read_csv(st_path)[self.parameter_list].to_numpy().flatten()
                arr_dt = pd.read_csv(dt_path)[self.parameter_list].to_numpy().flatten()

                costs = (arr_st - arr_dt) / arr_st * 100
                df_costs = pd.DataFrame(columns=self.parameter_list)
                df_costs.loc[0] = costs
                df_costs["sub"] = sub
                df_costs["cond"] = condition
                dt_costs_ls.append(df_costs)

        all_dt_costs_df = pd.concat(dt_costs_ls)
        mean_dt_costs = all_dt_costs_df.groupby("cond").mean().round(2).values
        std_dt_costs = all_dt_costs_df.groupby("cond").std().round(2).values

        # construct summary dataframe
        summary_dt_costs_df = pd.DataFrame(
            np.hstack((mean_dt_costs, std_dt_costs)),  # stack horizontally
            columns=self.parameter_list + [s + "_std" for s in self.parameter_list],
            index=["control", "fatigue"],
        )
        summary_dt_costs_df.sort_index(axis=1, inplace=True)

        # save summary dataframe to csv
        summary_dt_costs_df.to_csv(os.path.join(self.processed_base_path, "dt_costs_summary.csv"))

        print("\nDual-task cost (%) summary:")
        print(summary_dt_costs_df.to_string())

#### main ####
if __name__ == "__main__":

    run_list = ["OG_st_control", "OG_dt_control", "OG_st_fatigue", "OG_dt_fatigue"]

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

    parameter_list = ["stride_lengths_avg", "speed_avg"]

    data_summary = DataSummary(sub_list, run_list, parameter_list)
    data_summary.imu_raw_data_summary()  # summarize raw IMU data
    data_summary.gait_params_summary()      # summarize gait parameters
    data_summary.dt_costs()            # summarize dual-task costs

    SubjectInfo.main(statistics=True)   # summarize study-related information