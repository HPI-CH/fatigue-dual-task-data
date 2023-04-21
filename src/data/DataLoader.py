import pandas as pd
import numpy as np
import json
import os.path
import fnmatch


class DataLoader:
    """
    load raw IMU data & save as .csv file
    """

    def __init__(self, data_path, location_kw):
        self.data_path = data_path
        self.location_kw = location_kw
        self.data_df = None

    def load_GaitUp_data(self):
        """load and reformat data for gait analysis script."""

        # find file containing key word for left or right foot
        no_file = True
        for file in os.listdir(self.data_path):
            if fnmatch.fnmatch(file, "*" + self.location_kw + ".csv"):
                file_name = file
                print(file_name)
                no_file = False

        if no_file:
            print("No file for " + self.location_kw + " found.")
            return

        # extract data columns
        raw_data_df = pd.read_csv(
            os.path.join(self.data_path, file_name), skiprows=5, low_memory=False
        )
        self.data_df = raw_data_df.filter(
            ["Time", "Gyro X", "Gyro Y", "Gyro Z", "Accel X", "Accel Y", "Accel Z"],
            axis=1,
        )
        self.data_df = self.data_df.rename(
            columns={
                "Time": "timestamp",
                "Accel X": "AccX",
                "Accel Y": "AccY",
                "Accel Z": "AccZ",
                "Gyro X": "GyrX",
                "Gyro Y": "GyrY",
                "Gyro Z": "GyrZ",
            }
        )

        self.data_df = self.data_df.drop(0)
        self.data_df = self.data_df.apply(
            pd.to_numeric
        )  # convert all columns of DataFrame to numbers

        return self.data_df

    def cut_data(self, start_cut, end_cut):
        try:
            self.data_df = self.data_df[
                (self.data_df.index >= start_cut) & (self.data_df.index < end_cut)
            ]
            return self.data_df
        except AttributeError:
            print("Could not cut: Data not loaded yet.")
        else:
            print("Data successfully cut.")

    def save_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        try:
            self.data_df.to_csv(os.path.join(save_path, f"{self.location_kw}.csv"))
        except AttributeError:
            print("Could not save to csv: Data not loaded yet.")
        else:
            print("IMU data loaded and saved.")
