# DUO-GAIT: A Gait Dataset for Walking under Dual-Task and Fatigue Conditions with Inertial Measurement Units

## About
This repository contains scripts used to process the dataset DUO-GAIT (**Gait** Dataset for **D**ual-task and Fatig**u**e C**o**nditions). The publication about this dataset will soon be online.

The scripts achieves the following main functions:
* Segment the IMU recordings into walking sessions and fatigue exercise
* Calculate spatio-temporal gait parameters from the IMU signals
* Summarize gait parameters and other study-related information

---------
## Getting started
Clone repository \
```git clone https://github.com/HPI-CH/fatigue-dual-task-data.git``` \
```cd fatigue-dual-task-data```

Create virtual environment with Python version 3.7 (for example using Conda) \
```conda create --name name_of_your_choice python=3.7``` \
```conda activate name_of_your_choice```

Install requirements \
```pip install -r requirements.txt```

Download data for scientific purposes [here](TBD).

Save data to ```./data/```

Create ```./path.json``` file to store the absolute paths to the ```./data/``` folder. See example in ```./example_path.json```

---------
## Usage
### **Segmenting and loading the raw data into interim data**
*This step is not necessary if you have already downloaded the interim data.*\
The raw data of a specific run (for example sub_03 dual task) is transformed to the interim folder (which contains the raw data but cut into the sections control, sit_to_stand, and fatigue) in the following way:

1. Make sure the paths.json file has the correct paths to your data.
2. Configure read_folder and save_folder.
3. Run ```python ./src/main_LFRF_preprocessing.py```
4. The raw IMU data will be loaded and plotted. Manually inspect the plots to identify the start and end cutting sample numbers for the condition (e.g., control) using an appropriate sensor such as LF for the walking and SA for sit-to-stand. Segment the data in the ```data_loader.cut_data()``` line.
5. Re-execute the script, this time with cutting and and saving the data to the specified save_folder.

### **Calculate gait parameters using the interim data**
*This step is not necessary if you have already downloaded the processed data.*\
This part is based on the TRIPOD pipeline described in [this publication](https://doi.org/10.3390/data6090095), the scripts can be found [here](https://github.com/HPI-CH/TRIPOD). If you are using the scripts in this repository for gait parameter calculation, please cite [this publication](https://doi.org/10.3390/data6090095).
1. Make sure the paths.json file has the correct paths to your data.
2. Run ```python ./src/main_gait_parameters.py```

The calculated gait parameters can be found at data/processed/.../sub_XY/ and contains:
- left/right_foot_core_params.csv: containing the gait parameters for each stride
- left/right_foot_aggregate_params.csv: containing aggregated gait parameters of that foot (average, CV)
- aggregate_params.csv: containing aggregated gait parameters of both feet and additionally symmetry features

### **Summarize gait parameters and other study-related information**
1. Configure the data to be summarized:
   - gait parameters: parameter_list in file ```./src/main_data_summary.py```
2. Run ```python ./src/main_data_summary.py```

### **Perform two-way repeated measures ANOVA for example gait parameters**
1. Navigate to directory containing the R project:
    ```cd src_R```
2. Install the required packages using renv
3. Run ```Rscript anova.R```

--------
## Authors
* Lin Zhou
* Eric Fischer

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to process data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── LFRF_parameters  <- Scripts to calculate gait parameters from the left and right foot sensors.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
