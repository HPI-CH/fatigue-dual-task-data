import json
import os
import pandas as pd
import matplotlib

# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from LFRF_parameters.preprocessing.plot_raw_xyz import plot_acc_gyr
from LFRF_parameters.preprocessing.get_imu_gyro_thresholds import GyroPlot
from data.DataLoader import DataLoader

#### PARAMS START ####
load_raw = True  # load (and plot) raw IMU data into interim data
get_stance_threshold = False  # determine stance threshold

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

raw_folders = [
    "OG_st_raw",
    "OG_dt_raw",
]

all_folders = [
    "OG_st_all",
    "OG_dt_all",
]

runs = [
    "OG_st_control",
    "OG_st_fatigue",
    "OG_dt_control",
    "OG_dt_fatigue",
]

with open(os.path.join(os.path.dirname(__file__), "..", "path.json")) as f:
    paths = json.loads(f.read())
raw_base_path = os.path.join(paths["raw_data"])
interim_base_path = os.path.join(paths["interim_data"])
#### PARAMS END ####


#### plot and load raw data ####
if load_raw:
    sub = sub_list[0]
    read_folder = raw_folders[1]
    save_entire_recording_folder = all_folders[1]
    save_folder = runs[0]

    from_interim = False
    read_folder_path = os.path.join(raw_base_path, read_folder, sub)
    save_entire_recording_path = os.path.join(
        interim_base_path, save_entire_recording_folder, sub
    )  # save entire recording to interim folder
    save_folder_path = os.path.join(interim_base_path, save_folder, sub)

    # select IMU locations to load
    IMU_loc_list = ["LF", "SA"]
    for loc in IMU_loc_list:
        if from_interim:  # load interim data
            df_loc = pd.read_csv(os.path.join(read_folder_path, loc + ".csv"))
        else:  # load raw data (& save file to the interim folder)
            data_loader = DataLoader(read_folder_path, loc)
            df_loc = data_loader.load_GaitUp_data()
            data_loader.save_data(
                save_entire_recording_path
            )  # save re-formatted entire data into /interim folder
            # df_loc = data_loader.cut_data(322000, 372000)  # (if necessary: segment data)
            data_loader.save_data(save_folder_path)  # save re-formatted data into /interim folder

        if df_loc is not None:  # if the IMU data is loaded, plot the signals
            plot_acc_gyr(df_loc, ["AccX", "AccY", "AccZ"], "raw_Acc_" + loc, save_folder_path)
            plot_acc_gyr(df_loc, ["GyrX", "GyrY", "GyrZ"], "raw_Gyr_" + loc, save_folder_path)

    plt.show()


#### get gyro stance threshold ####
if get_stance_threshold:
    overwrite = False  # if False: append to existing file

    # if no file, create one. Otherwise append to the existing file
    if (
        not os.path.isfile(
            os.path.join(interim_base_path, "stance_magnitude_thresholds.csv")
        )
        or overwrite
    ):
        pd.DataFrame(
            columns=[
                "subject",
                "run",
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
                "stance_count_threshold_left",
                "stance_count_threshold_right",
            ],
        ).to_csv(
            os.path.join(interim_base_path, "stance_magnitude_thresholds.csv"),
            index=False,
        )

    for subject_id, subject in enumerate(sub_list):
        for run_id, run in enumerate(runs):
            file_directory = os.path.join(interim_base_path, run, subject)
            print(
                "subject",
                subject_id + 1,
                "/",
                len(sub_list),
                "/",
                "run",
                run_id + 1,
                "/",
                len(runs),
            )

            # run interactive gyro stance phase detection
            gp = GyroPlot(file_directory, interim_base_path, subject, run)
            gp.gyro_threshold_slider()
