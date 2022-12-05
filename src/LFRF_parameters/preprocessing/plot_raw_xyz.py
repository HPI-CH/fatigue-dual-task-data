#### imports ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import seaborn as sns
sns.set()

#### functions ####
def get_acc_gyr(df, sensor):
    """
    extract acc and gyr columns and add sensor name to the column headers
    :param df: the raw data
    :param sensor: sensor name to be added to the column headers
    :return: extracted dataframe
    """
    df = df[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]  # get acc and gyr columns
    df = df.add_suffix('_' + sensor)
    return df


def plot_acc_gyr(df, columns, title, save_fig_path):
    """
    plot raw sensor data from xyz axis
    :param df: dataframe contatining data, column names are used as legends
    :param columns: selsect columns to be plotted. e.g. ['AccX', 'AccY', 'AccZ']
    :param title: title of the figure
    :param save_fig_path: folder path for saving the figure
    :return: saves the figure in .png
    """

    sns.set_style("whitegrid", {'axes.edgecolor': 'black'})
    sns.set_context("paper", font_scale=1.8)

    df = df[columns]
    df.plot(figsize=(15, 5),
            color=[plt.cm.winter_r(0), plt.cm.winter_r(100), plt.cm.winter_r(200)])  # cmap=plt.cm.winter_r)
    plt.xlabel('Sample')
    # plt.ylim(-10, 10)

    if 'Acc' in title:
        plt.ylabel('Acceleration (g)')
        acc_mag = np.linalg.norm(df.values, axis=-1)
        plt.title(title + '\n Acc_mag = ' +
                  '{:.2f}'.format(round(np.mean(acc_mag), 2)) + '    '
                                                                'num. samples = ' + str(len(df.index)))
        # print('acc mag: ' + str(np.mean(acc_mag)))
        print('num. acc samples = ' + str(len(df.index)))
    elif 'Gyr' in title:
        plt.ylabel('Angular Velocity (Degrees/s)')
        gyro_mag = np.linalg.norm(df.values, axis=-1)
        plt.title(title + '\n Gyro_mag = ' +
                  '{:.2f}'.format(round(np.mean(gyro_mag), 2)) + '    '
                                                                 'num. samples = ' + str(len(df.index)))
        # print('gyro mag: ' + str(np.mean(gyro_mag)))
        print('num. gyro samples = ' + str(len(df.index)) + '\n')
    else:
        plt.ylabel('Data')

    fig = plt.gcf()

    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    plt.savefig(os.path.join(save_fig_path, str(title + '.png')), bbox_inches='tight')
    # plt.show()
