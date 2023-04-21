import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class Counting:
    """
    data from the number counting task, analysis methods
    """

    def __init__(self, data_type):
        """
        read in the counting data from all subjects
        @param data_type: 'raw': with all numbers, 'summary': summary of count, duration, correct response rate, etc.
        @type data_type: string
        """
        # get path of the subject_info file
        with open("path.json") as f:
            paths = json.load(f)
            self.save_path = os.path.join(paths["processed_data"], "all_counting_task")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        if data_type == "raw":
            read_path = os.path.join(paths["raw_data"], "OG_dt_BINs")
            self.sub_list = [
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
            self.cond_list = ["control", "fatigue"]
            self.data_raw = pd.DataFrame()
            for sub in self.sub_list:
                for cond in self.cond_list:
                    data_df = pd.read_csv(
                        os.path.join(read_path, sub, "transcript_" + cond + ".csv")
                    )
                    # append information
                    data_df["subject"] = sub
                    data_df["condition"] = cond
                    # append to the end of data_raw
                    self.data_raw = self.data_raw.append(data_df, ignore_index=True)
            print("Loaded raw counting transcripts from all subjects.")

        elif data_type == "summary":
            read_path = self.save_path
            self.data_summary = pd.read_csv(
                os.path.join(read_path, "summary_counting.csv")
            )

        self.stats_df = pd.DataFrame()

    def get_raw_df(self):
        return self.data_raw

    def get_summary_df(self):
        """
        summarize the data by subject and condition, can also be used for plotting i.e. same as "plot_df"
        @return:
        @rtype:
        """
        return self.data_summary

    def get_stats_df(self):
        """
        reshape the dataframe for statistical analysis
        @return: dataframe
        @rtype:
        """
        df_list = []
        for cond in ["control", "fatigue"]:
            cond_df = self.data_summary[self.data_summary["condition"] == cond]
            cond_df = cond_df.add_suffix("_" + cond)
            cond_df.rename(columns={"subject_" + cond: "subject"}, inplace=True)
            df_list.append(cond_df)

        self.stats_df = pd.merge(left=df_list[0], right=df_list[1], on="subject")
        return self.stats_df

    def process_transcript(self):
        """
        extract information from transcript from one subject, save summary to processed data folder
        this method only has to be executed once
        @param read_path: folder path
        @type read_path: string
        @param condition: 'control' or 'fatigue'
        @type condition: string
        @return:
        @rtype:
        """

        data = []  # list to save summary data
        for sub in self.sub_list:
            for cond in self.cond_list:
                num_df = self.data_raw[
                    np.logical_and(
                        self.data_raw["subject"] == sub,
                        self.data_raw["condition"] == cond,
                    )
                ]
                num_df.reset_index(inplace=True)
                diff_df = num_df["number"].diff()
                n_overall = num_df["number"].count()
                n_diff = diff_df.value_counts()

                try:
                    n_correct = n_diff.at[-7.0]
                except KeyError:
                    n_correct = 0

                # find duration of the recording
                duration = num_df["duration(s)"][0]

                # append data
                data.append([sub, cond, n_overall, n_correct, duration])

        # make dataframe for summary from all subjects and conditions
        sum_df = pd.DataFrame(
            data,
            columns=["subject", "condition", "n_overall", "n_correct", "duration(s)"],
        )
        num_df.reset_index(inplace=True)
        # save to .csv
        sum_df.to_csv(os.path.join(self.save_path, "summary_counting.csv"), index=False)
        print("Summary of transcripts from all subjects saved.")

    def calculate_crr(self, n_correct, n_overall, time):
        """
        Calculates the Correct Response Rate.

        :param n_correct: number of correct subtractions
        :param n_overall: overall number of subtractions
        :param time: elapsed time of the recording in seconds
        :return: the CRR
        """

        response_rate = n_overall / time
        accuracy = n_correct / n_overall
        crr = response_rate * accuracy

        return crr

    def append_crr(self):
        """
        add correct resonse rate and other columns to the summary data
        @return:
        @rtype:
        """
        self.data_summary["CRR"] = self.calculate_crr(
            self.data_summary["n_correct"],
            self.data_summary["n_overall"],
            self.data_summary["duration(s)"],
        )
        # append other calculations
        self.data_summary["response_rate"] = (
            self.data_summary["n_overall"] / self.data_summary["duration(s)"]
        )
        self.data_summary["accuracy"] = (
            self.data_summary["n_correct"] / self.data_summary["n_overall"]
        )

        # self.data_summary.reset_index(inplace=True)
        self.data_summary.to_csv(
            os.path.join(self.save_path, "summary_counting.csv"), index=False
        )
        print("Correct response rates from all subjects saved.")

    def histogram(self):
        fig, ax = plt.subplots(1, 3, sharex="col", sharey="row")
        col_list = ["response_rate", "accuracy", "CRR"]
        for i in range(3):
            print(i)
            self.data_summary.hist(
                column=col_list[i], bins=10, ax=ax[i], figsize=(10, 30)
            )

        # self.data_summary[column].hist(alpha=0.5)
        plt.suptitle("Number Counting Task")
        plt.show()

    def boxplot(self):
        self.data_summary.boxplot(
            column=["response_rate", "accuracy", "CRR"], by="condition", layout=(1, 3)
        )
        plt.suptitle("Number Counting Task")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # add space around super title
        plt.show()


if __name__ == "__main__":

    # process the raw transcripts (only run this once)
    counting = Counting("raw")
    counting.process_transcript()

    counting = Counting("summary")
    counting.append_crr()
    counting.boxplot()
    counting.histogram()

    summary_df = counting.get_summary_df()
    stats_df = counting.get_stats_df()

    # get mean values
    df_mean = counting.get_summary_df().groupby(by="condition").mean()
    print(df_mean)
