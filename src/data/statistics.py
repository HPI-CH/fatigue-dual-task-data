import numpy as np
from data.SubjectInfo import *
from data.Counting import *
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats


#### functions ####
def check_normal_dist(df, data_type, group_by, var="value"):
    """
    make plots to check for normal distribution
    @return: plots
    @rtype:
    """
    # histograms
    df[var].hist(by=df[group_by], bins=16, alpha=0.5)
    plt.suptitle("Histogram " + data_type)
    plt.show()

    # Empirical cumulative distribution
    df[var].hist(by=df[group_by], cumulative=True, density=1, bins=16, alpha=0.5)
    plt.suptitle("Empirical Cumulative Distribution " + data_type)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # add space around super title
    plt.show()

    # Q-Q plots
    # print(df[data_type].quantile([.1, .25, .5, .75]))
    stats.probplot(df[var], dist="norm", plot=pylab)
    plt.suptitle("Q-Q plot " + data_type)
    plt.show()


def test_mean(df, data_type, group_by, norm_dist=True, var="value"):
    """
    significance test for the means
    @param df: input dataframe with the variable keyword in TWO column names
    @type df:
    @param norm_dist:
    @type norm_dist:
    @param var:
    @type var:
    @return:
    @rtype:
    """

    # Shapiro test for normal distribution
    norm_dist = True
    for col in df.columns:
        if data_type in col:
            shap_stat, shapiro_p = stats.shapiro(df[col])
            print("Shapiro test for " + col)
            print(shapiro_p)
            if shapiro_p <= 0.05:
                norm_dist = False

    # t test
    if data_type in ["lac", "HR"]:
        cols = [
            col
            for col in df.columns
            if np.logical_and(data_type in col, "fatigue" in col)
        ]
    elif data_type in ["st_lac", "dt_lac"]:
        cols = [
            col
            for col in df.columns
            if data_type in col
        ]
    else:
        cols = [col for col in df.columns if data_type in col]

    # check if two columns are selected
    if len(cols) == 2:
        if norm_dist:
            # one-sided t test
            t, p_value_t = stats.ttest_rel(df[cols[0]], df[cols[1]])
            print("two-sided p value = " + str(p_value_t))
            print("one-sided p value = " + str(p_value_t / 2))
            print(f"H0: there is no significant difference between {cols[0]} and {cols[1]}")
            if t > 0:
                print("H1: " + cols[0] + " is greater than " + cols[1])
            elif t < 0:
                print("H1: " + cols[0] + " is less than " + cols[1])
        elif not norm_dist:
            # two-sided Wolcox test
            w, p_value_w = stats.wilcoxon(
                df[cols[0]], df[cols[1]], alternative="two-sided"
            )
            # one-sided Wilcox test
            w1, p_value_w1 = stats.wilcoxon(
                df[cols[0]], df[cols[1]], alternative="greater"
            )
            w2, p_value_w2 = stats.wilcoxon(
                df[cols[0]], df[cols[1]], alternative="less"
            )
            print("Wilcoxon test")
            print(f"two-sided, p = {p_value_w}")
            print(
                "H1: "
                + cols[0]
                + " is greater than "
                + cols[1]
                + ", p = "
                + str(p_value_w1)
            )
            print(
                "H1: "
                + cols[0]
                + " is less than "
                + cols[1]
                + ", p = "
                + str(p_value_w2)
            )

    else:
        print("Check if the correct columns are selected:")
        print(cols)


#### main ####
if __name__ == "__main__":
    sub_info = SubjectInfo()
    dat_type = "lac"  # lac (lactate), HR (heart rate), time (time to fatigue), borg, days_recovery
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
