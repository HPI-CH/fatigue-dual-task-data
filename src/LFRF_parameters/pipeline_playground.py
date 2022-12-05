import json
import os
import sys

from LFRF_parameters.pipeline.data_loader import *
from LFRF_parameters.pipeline.event_detector import TuncaEventDetector
from LFRF_parameters.pipeline.trajectory_estimator import TuncaTrajectoryEstimator
from LFRF_parameters.pipeline.pipeline import Pipeline

def execute(sub_list, runs, dataset, paths):
    """
    Executes the Playground pipeline.
    Returns
    -------

    """
    #configure the pipeline
    pipeline_config = {
                        'raw_base_path': paths['raw_data'],
                        'interim_base_path': paths['interim_data'],
                        'processed_base_path': paths['processed_data'],
                        'overwrite': False,  # overwrite the trajectory estimations
                        'show_figures': 0,  # show figures from intermediate steps. 2: figures are saved; 1: figures are shown; 0: no figures plotted
                        'location_kws': ['LF', 'RF'],
                        'data_loader': PhysilogDataLoader,
                        'trajectory_estimator': TuncaTrajectoryEstimator,
                        'gait_event_detector': TuncaEventDetector,
                        'dataset': dataset,
                        'runs': runs,
                        'subjects': sub_list,
    }

    # create the pipeline
    pipeline = Pipeline(pipeline_config)

    # list of tuples (run number, subject number)
    everything = [(x, y) for x in range(0, len(pipeline_config["runs"])) for y in range(0, len(pipeline_config["subjects"]))]
    # analyze = [(1, 0), (1, 1), (1, 2)]

    analyze = everything
    pipeline.execute(analyze)

