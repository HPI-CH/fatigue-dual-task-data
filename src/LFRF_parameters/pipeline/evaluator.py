import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Evaluator:

    def __init__(self):
        self.data = {}
        self.merged = {}

    def add_data(self, subject_num, run_num, data, reference_data):
        self.data[(subject_num, run_num)] = {"data" : data, "reference_data" : reference_data, "merged" : {}} 
        self.match_timestamps()

    def match_timestamps(self):
        for data_key, data in self.data.items():
            for side in ["left", "right"]:
                data["merged"][side] = pd.merge_asof(left=data["data"][side], right=data["reference_data"][side], on='timestamp', direction='nearest', tolerance=400, allow_exact_matches=True)
        
    def detect_outliers(self, column):
        z_threshold = 1
        # Compute z scores
        reference_column = column + "_ref"
        for data in self.data.values():
            for side in ["left", "right"]:
                data["merged"][side]["z_scores_data"] = np.abs(stats.zscore(data["merged"][side][column]))
                data["merged"][side]["z_scores_reference"] = np.abs(stats.zscore(data["merged"][side][reference_column]))
                data["merged"][side]["outlier"] = np.logical_or(data["merged"][side]["z_scores_data"] > z_threshold, data["merged"][side]["z_scores_reference"] > z_threshold)

    def reg_line(self, x, y):
        gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
        # get p values and CI for the gradient and intercept
        X = sm.add_constant(x)
        model = sm.OLS(y,X)
        results = model.fit()
        pvalues = results.pvalues
        conf_interval = results.conf_int(alpha=0.05, cols=None)
        #print('p values:')
        #print(results.pvalues)
        #print('confidence intervals:')
        #print(conf_interval)
        
        # calculate RMSE (root mean squared error)
        y_pred = gradient * x + intercept
        rmse = np.sqrt(np.mean((y_pred - y)**2))
        
        # make a regression line
        mn = np.min(x)
        mx = np.max(x + 0.5)
        mn = 0
        x1 = np.linspace(mn,mx,500)
        y1 = gradient * x1 + intercept
        
        # summary line info
        line_info = [round(gradient, 4), 
                     round(intercept, 4), 
                     round(r_value, 4), 
                     round(p_value, 4),
                     round(std_err, 4), 
                     round(rmse, 4)]

        return x1, y1, line_info, pvalues, conf_interval

    def plot_correlation(self, column, subject_run_nums):
        self.detect_outliers(column)

        merged = {}
        for side in ["left", "right"]:
            merged[side] = pd.concat([self.data[subject_run_num]["merged"][side] for subject_run_num in subject_run_nums])

        reference_column = column + "_ref"
        axes_min = np.inf
        axes_max = - np.inf
        for side in ["left", "right"]:
            x = merged[side][column][merged[side]["outlier"] == False]
            y = merged[side][reference_column][merged[side]["outlier"] == False]
            axes_min = np.minimum(np.minimum(np.min(x), np.min(y)), axes_min)
            axes_max = np.maximum(np.maximum(np.max(x), np.max(y)), axes_max)
            x_line, y_line, info, pvalues, conf_interval = self.reg_line(x, y)
            print("r=", info[2])
            print(f"y={info[0]}x + {info[1]}")
            plt.plot(x_line, y_line)
            plt.plot(x_line, x_line, color='0.75')
            plt.scatter(x, y)

        plt.xlim((axes_min - 0.05, axes_max + 0.05))
        plt.ylim((axes_min - 0.05, axes_max + 0.05))
        
        plt.show()
