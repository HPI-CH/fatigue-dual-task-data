"""
Calculate stride-by-stride and aggregated gait parameters.
"""

import json
import os
from features.build_features import build_features
from features.postprocessing import mark_processed_data
from LFRF_parameters import pipeline_playground
from features import aggregate_gait_parameters

### PARAMS START ###
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
test = ["control", "fatigue"]  # "control", "fatigue"
conditions = ["st", "dt"]  # "st", "dt"
runs = [
    f"OG_{conditions[0]}_{test[0]}",
    f"OG_{conditions[0]}_{test[1]}",
    f"OG_{conditions[1]}_{test[0]}",
    f"OG_{conditions[1]}_{test[1]}",
]
dataset = "fatigue_dual_task"
with open(os.path.join(os.path.dirname(__file__), "..", "path.json")) as f:
    paths = json.load(f)
interim_base_path = paths["interim_data"]
processed_base_path = paths["processed_data"]
### PARAMS END ###

### Execute the Gait Analysis Pipeline ###
pipeline_playground.execute(sub_list, runs, dataset, paths)

### Mark outliers strides (turning intervals, interrupted strides) ###
mark_processed_data(runs, sub_list, processed_base_path, interim_base_path)

### Aggregate Gait Parameters over all recording session ###
aggregate_gait_parameters.main(runs, sub_list, processed_base_path)

### Build windows ###
base_path = paths["data"]
window_sz = 10
window_slide = 2
build_features(
    sub_list,
    base_path,
    test,
    conditions,
    window_sz,
    window_slide,
    aggregate_windows=True,
    add_static_features=True,
    save_unwindowed_df=True,
)
