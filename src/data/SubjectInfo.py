#### imports ####
import pandas as pd
import glob
import numpy as np
import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import json
import os.path
import fnmatch
import seaborn as sns

sns.set()

from data.statistics import *


class SubjectInfo:
    """
    takes information in the subject_info.csv file
    visualizes the columns
    """

    def __init__(self):
        # get path of the subject_info file
        with open("path.json") as f:
            paths = json.load(f)
        read_path = os.path.join(paths["raw_data"], "subject_info.csv")

        self.data_df = pd.read_csv(read_path)  # dataframe with original subject info
        self.plot_df = {}  # dataframe for plotting
        self.stats_df = {}  # dataframe for plotting
        self.col_names = {
            "lac": "lac",
            "HR": "HR",
            "time": "time_to_fatigue(min)",
            "borg": "borg_scale",
        }

        # add average lactate baseline from the two measurements
        if "st_lac_baseline" not in self.data_df:
            self.data_df["st_lac_baseline"] = self.data_df[
                ["st_lac_baseline1", "st_lac_baseline2"]
            ].mean(axis=1)
        if "dt_lac_baseline" not in self.data_df:
            self.data_df["dt_lac_baseline"] = self.data_df[
                ["dt_lac_baseline1", "dt_lac_baseline2"]
            ].mean(axis=1)

    def get_data(self):
        return self.data_df

    def anthropometrics(self):
        """
        summary of subjects' age, height, weight, leg length
        @return: print out mean +- standard deviation
        @rtype:
        """
        for item in ["age", "weight(kg)", "height(cm)", "leg_length(cm)"]:
            mean = round(self.data_df[item].values.mean(), 1)
            std = round(self.data_df[item].values.std(), 1)
            min = round(self.data_df[item].values.min(), 1)
            max = round(self.data_df[item].values.max(), 1)

            print(f"{item}: {mean} +- {std}, min: {min}, max: {max}")

    def append_visits_lac(self):
        """
        append column to self.data_df that marks 1st and 2nd visit for lactate value and heart rate
        (where there are before- and after values)
        @return:
        @rtype:
        """
        self.plot_df.loc[
            np.logical_or(
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 1,
                    self.plot_df["condition"].str.contains("dt.*baseline"),
                ),
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 2,
                    self.plot_df["condition"].str.contains("st.*baseline"),
                ),
            ),
            "visit",
        ] = "1st Visit Control"

        self.plot_df.loc[
            np.logical_or(
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 1,
                    self.plot_df["condition"].str.contains("st.*baseline"),
                ),
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 2,
                    self.plot_df["condition"].str.contains("dt.*baseline"),
                ),
            ),
            "visit",
        ] = "2nd Visit Control"

        self.plot_df.loc[
            np.logical_or(
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 1,
                    self.plot_df["condition"].str.contains("dt.*fatigue"),
                ),
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 2,
                    self.plot_df["condition"].str.contains("st.*fatigue"),
                ),
            ),
            "visit",
        ] = "1st Visit Fatigue"

        self.plot_df.loc[
            np.logical_or(
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 1,
                    self.plot_df["condition"].str.contains("st.*fatigue"),
                ),
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 2,
                    self.plot_df["condition"].str.contains("dt.*fatigue"),
                ),
            ),
            "visit",
        ] = "2nd Visit Fatigue"

    def append_visits(self):
        """
        append column to self.data_df that marks 1st & 2nd visit (where there's only one value for each visit)
        @return:
        @rtype:
        """
        self.plot_df.loc[
            np.logical_or(
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 1,
                    self.plot_df["condition"].str.contains("dt"),
                ),
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 2,
                    self.plot_df["condition"].str.contains("st"),
                ),
            ),
            "visit",
        ] = "1st Visit"

        self.plot_df.loc[
            np.logical_or(
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 1,
                    self.plot_df["condition"].str.contains("st"),
                ),
                np.logical_and(
                    self.plot_df["dual_task_visit"] == 2,
                    self.plot_df["condition"].str.contains("dt"),
                ),
            ),
            "visit",
        ] = "2nd Visit"

    def get_plot_df(self, data_type, group_by):
        """
        re-structure the dataframe for boxplots
        @param data_type: lac (lactate), HR (heart rate), time (time to fatigue), borg
        @type data_type: string
        @param group_by: condition (single- vs. dual-task), visit (1st visit vs. 2nd visit)
        @type group_by: string
        @return: dataframe
        @rtype: dataframe
        """

        # reshape the dataframe for plotting
        if data_type in [
            "lac",
            "HR",
        ]:  # if two data points per visit (baseline & fatigue)
            if "lac" in data_type:
                value_vars_kw = "lac"
            elif "HR" in data_type:
                value_vars_kw = "HR"

            self.plot_df = pd.melt(
                self.data_df,
                id_vars=["sub", "sex", "dual_task_visit"],
                value_vars=[
                    "st_" + value_vars_kw + "_baseline",
                    "dt_" + value_vars_kw + "_baseline",
                    "st_" + value_vars_kw + "_fatigue",
                    "dt_" + value_vars_kw + "_fatigue",
                ],
                var_name="condition",
                value_name="value",
            )

            if "visit" in group_by:
                self.append_visits_lac()

        if data_type in ["time", "borg"]:  # if one data points per visit
            if "time" in data_type:
                value_vars_kw = "time_to_fatigue(min)"
            elif "borg" in data_type:
                value_vars_kw = "borg_scale"

            self.plot_df = pd.melt(
                self.data_df,
                id_vars=["sub", "sex", "dual_task_visit"],
                value_vars=["st_" + value_vars_kw, "dt_" + value_vars_kw],
                var_name="condition",
                value_name="value",
            )

            if "visit" in group_by:
                self.append_visits()

        data_points = self.plot_df.count()
        print("num. data points: \n" + str(data_points))

        # reformat colums for publication
        self.plot_df["condition"] = self.plot_df["condition"].str.replace("_", " ")
        self.plot_df["condition"] = self.plot_df["condition"].str.title()
        self.plot_df["condition"] = self.plot_df["condition"].str.replace("St", "ST")
        self.plot_df["condition"] = self.plot_df["condition"].str.replace("Dt", "DT")

        return self.plot_df

    def get_stats_df(self, data_type, group_by):
        """
        re-structure the dataframe for statistics (make paired analysis easier)
        @param data_type: lac (lactate), HR (heart rate), time (time to fatigue), borg
        @type data_type: string
        @param group_by: condition (single- vs. dual-task), visit (1st visit vs. 2nd visit)
        @type group_by: string
        @return: dataframe
        @rtype: dataframe
        """
        # find all columns containing data_type
        data_cols = [col for col in self.data_df.columns if data_type in col]
        data_cols = ["sub", "sex", "dual_task_visit"] + data_cols
        # only keep the averaged baselines
        data_cols[:] = [
            x
            for x in data_cols
            if np.logical_and("baseline1" not in x, "baseline2" not in x)
        ]
        self.stats_df = self.data_df[data_cols]

        if group_by == "visit":
            # rename columns, default: st is 1st visit
            self.stats_df.columns = self.stats_df.columns.str.replace(
                "st_", "1st_visit_"
            )
            self.stats_df.columns = self.stats_df.columns.str.replace(
                "dt_", "2nd_visit_"
            )

            # now correct the rows where dt is 1st visit
            mask = self.stats_df["dual_task_visit"] == 1
            if data_type in ["time", "borg"]:
                self.stats_df.loc[
                    mask,
                    [
                        "1st_visit_" + self.col_names[data_type],
                        "2nd_visit_" + self.col_names[data_type],
                    ],
                ] = self.stats_df.loc[
                    mask,
                    [
                        "2nd_visit_" + self.col_names[data_type],
                        "1st_visit_" + self.col_names[data_type],
                    ],
                ].values
                print("")
            elif data_type in ["lac", "HR"]:
                self.stats_df.loc[
                    mask,
                    [
                        "1st_visit_" + self.col_names[data_type] + "_baseline",
                        "2nd_visit_" + self.col_names[data_type] + "_baseline",
                        "1st_visit_" + self.col_names[data_type] + "_fatigue",
                        "2nd_visit_" + self.col_names[data_type] + "_fatigue",
                    ],
                ] = self.stats_df.loc[
                    mask,
                    [
                        "2nd_visit_" + self.col_names[data_type] + "_baseline",
                        "1st_visit_" + self.col_names[data_type] + "_baseline",
                        "2nd_visit_" + self.col_names[data_type] + "_fatigue",
                        "1st_visit_" + self.col_names[data_type] + "_fatigue",
                    ],
                ].values

        return self.stats_df

    def boxplot_info(self, group, title, save_fig_path):
        """
        make boxplots
        @param group: 'visit' (1st vs. 2nd visit), 'condition' (control vs. fatigue)
        @type group: string
        @param title: title of the figure
        @type title: string
        @param save_fig_path: folder
        @type save_fig_path:
        @return:
        @rtype:
        """
        print("make boxplot")
        if "Lactate" in title:
            # h_position = 2
            y_label = "Blood Lactate Concentration (mmol/L)"
        elif "Heart" in title:
            y_label = "Heart Rate (bpm)"
        elif "Borg" in title:
            y_label = "Borg Scale"
        elif "Time" in title:
            y_label = "Time (min)"

        sns.set_style("whitegrid", {"axes.edgecolor": "black"})
        sns.set_context("paper", font_scale=1.8)

        fig1 = plt.figure(figsize=(5, 6))
        ax = sns.boxplot(
            x=group, y="value", data=self.plot_df
        )  # color=plt.cm.winter_r(100), fliersize=2)  # , #palette='Paired')
        plt.setp(ax.artists, edgecolor="0.4", facecolor="w")
        plt.setp(ax.lines, color="0.4")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.label.set_visible(False)
        plt.ylabel(y_label)
        plt.tight_layout(h_pad=0.2)
        plt.title(title)

        # if a reference line exists
        if "h_position" in locals():
            plt.axhline(
                y=h_position, linewidth=2, color=plt.cm.winter_r(100), alpha=0.5
            )

        plt.savefig(
            os.path.join(save_fig_path, str(title + ".pdf")), bbox_inches="tight"
        )
        # plt.show()

    def main(boxplot=False, statistics=False):
        """
        main function to execute selected summaries
        """

        # create subject info object
        sub_info = SubjectInfo()

        # print anthropometrics
        sub_info.anthropometrics()

        if boxplot:
            with open("path.json") as f:
                paths = json.load(f)
            save_fig_path = os.path.join(paths["processed_data"], "all_subjects_info")

            if not os.path.exists(save_fig_path):
                os.makedirs(save_fig_path)

            # make boxplots
            dat_type = (
                "lac"  # lac (lactate), HR (heart rate), time (time to fatigue), borg
            )
            grou_by = "condition"  # visit, condition
            sub_info.get_plot_df(dat_type, grou_by)
            sub_info.get_stats_df(dat_type, grou_by)
            sub_info.boxplot_info(
                grou_by, "Lactate Single Task vs. Dual Task", save_fig_path
            )
            # sub_info.boxplot_info(grou_by, 'Heart Rate Single Task vs. Dual Task', save_fig_path)
            # sub_info.boxplot_info(grou_by, 'Time to Fatigue Single Task vs. Dual Task', save_fig_path)
            # sub_info.boxplot_info(grou_by, 'Borg Scale Single Task vs. Dual Task', save_fig_path)

            # sub_info.boxplot_info(grou_by, 'Lactate 1st vs. 2nd Visit', save_fig_path)
            # sub_info.boxplot_info(grou_by, 'Heart Rate 1st vs. 2nd Visit', save_fig_path)
            # sub_info.boxplot_info(grou_by, 'Time to Fatigue 1st vs. 2nd Visit', save_fig_path)
            # sub_info.boxplot_info(grou_by, 'Borg Scale 1st vs. 2nd Visit', save_fig_path)

        if statistics:
            dat_type = "dt_lac"  
            # lac (lactate), HR (heart rate), time (time to fatigue), borg, days_recovery
            # st_lac: lactate under single task before and after fatigue
            # dt_lac: lactate under dual task before and after fatigue
            grou_by = "condition"  # visit, condition
            # plot_df = sub_info.get_plot_df(dat_type, grou_by)
            stats_df = sub_info.get_stats_df(dat_type, grou_by)
            stats_df = stats_df.dropna()

            # counting = Counting('summary')
            # dat_type = 'accuracy'  # response_rate, accuracy, n_overall, n_correct, duration(s), CRR
            # grou_by = 'condition'  # can only be condition here
            # stats_df = counting.get_stats_df()

            # check_normal_dist(plot_df, dat_type, grou_by)
            test_mean(stats_df, dat_type, grou_by)


if __name__ == "__main__":
    SubjectInfo.main(statistics=True)
