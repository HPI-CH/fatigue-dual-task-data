""" This script anonymizes raw data by removing sensitive data """

import os
import fnmatch

if __name__ == "__main__":  # noqa: C901
    directory = "./data/raw/"

    for root, d_names, f_names in os.walk(directory):
        for f in f_names:
            file_path = os.path.join(root, f)
            # print(file_path)
            if f in [
                "IPAQ.csv", 
                "subject_info.csv", 
                "transcript_control.csv", 
                "transcript_fatigue.csv"
                ]:
                # print(f)
                # no modifications
                pass

            elif fnmatch.fnmatch(f.lower(), "*.BIN"):
                # remove binary files
                os.remove(file_path)

            elif fnmatch.fnmatch(f.lower(), "*.wav"):
                # remove audio files
                os.remove(file_path)

            elif fnmatch.fnmatch(f.lower(), "heart_rate.csv"):
                # remove metadata
                lines = open(file_path, "r").readlines()
                lines[1] = "fatigue_dual_task\n"
                open(file_path, "w").writelines(lines)

            elif fnmatch.fnmatch(f.lower(), "*.csv"):
                # remove date and time from IMU data
                lines = open(file_path, "r").readlines()
                lines[1] = "Created on: YYYY-MM-DD hh:mm:ss\n"
                open(file_path, "w").writelines(lines)

