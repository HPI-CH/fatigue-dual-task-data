"""
===========
Slider for gyro stance threshold
===========

"""
from curses import raw
import sys,os
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append("./src")
import csv
import numpy as np
import pandas as pd
from data.imu import IMU
from LFRF_parameters.event_detection.imu_event_detection import gyro_threshold_stance
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class GyroPlot():
    def __init__(self, read_path, save_path, sub, run):
        self.read_path = read_path
        self.save_path = save_path
        self.sub = sub
        self.run = run

    def update(self, val):
        gyro_threshold = self.sgyro_mag.val
        count = int(self.scount.val)
        stance = gyro_threshold_stance(
            self.imu,
            stance_magnitude_threshold=gyro_threshold,
            stance_count_threshold=count,
        )
        self.hline.set_ydata(gyro_threshold)
        # delete old stance_shadows for re-plot
        for collection in self.ax.collections:
            if str(collection.get_label()) == "Stance":
                collection.remove()
        self.stance_shadows = self.ax.fill_between(self.samples, 0, 1, where=stance, alpha=0.4,
                                        facecolor='skyblue',
                                        transform=self.ax.get_xaxis_transform(), label='Stance')
        self.fig.canvas.draw_idle()

    def reset(self, event):
        self.scount.reset()
        self.sgyro_mag.reset()

    def save(self, event):
        self.stance_magnitude_thresholds[self.foot] = self.sgyro_mag.val
        self.stance_count_thresholds[self.foot] = self.scount.val

        plt.savefig(os.path.join(self.save_path,
                                str('stance_threshold_' + self.foot + '_.png')),
                    bbox_inches='tight')

    def change_color(self, event):
        self.button_save.color = '0.7'

    def check_duplicates(self, file_path):
        df = pd.read_csv(file_path)
        dup = df.groupby(['subject', 'run']).size() > 1
        if dup.any():
            print('!!!!!!!!!! duplicate entries !!!!!!!!!! see below:')
            print(dup[dup != 0])
        else:
            print("======== No duplicate entries for gyro stance threshold. ========")

    def gyro_threshold_slider(self):
        c0 = 8  # initial stance count threshold
        g0 = 0.7  # initial gyro magnitude threshold
        # setup
        delta_c = 1  # stance count resolution
        delta_g = 0.1  # gyro magnitude resolution

        # get the data file
        gyro_thresholds = []  # save one subject at a time
        self.stance_magnitude_thresholds = {"LF": None, "RF": None}
        self.stance_count_thresholds = {"LF": None, "RF": None}

        for self.foot in ["LF", "RF"]:
            imu_path = os.path.join(self.read_path, self.foot + ".csv")
            self.imu = IMU(imu_path)
            self.imu.gyro_to_rad()
            gyro_mag = np.linalg.norm(self.imu.gyro(), axis=1)
            self.samples = np.arange(len(gyro_mag))

            stance = gyro_threshold_stance(
                self.imu,
                stance_magnitude_threshold=g0,
                stance_count_threshold=c0,
            ).astype(bool)

            # make plot
            self.fig, self.ax = plt.subplots(figsize=(12, 5))
            plt.subplots_adjust(left=0.12, bottom=0.3)
            l, = plt.plot(self.samples, gyro_mag, lw=1)
            self.hline = self.ax.axhline(y=g0, xmin=0.0, xmax=1.0, color='coral', label='Gyro Magnitude Threshold')
            self.stance_shadows = self.ax.fill_between(self.samples, 0, 1, where=stance, alpha=0.4,
                                                facecolor='skyblue',
                                                transform=self.ax.get_xaxis_transform(), label='Stance')
            plt.title('Manual Stance Detection' +
                        '\n' + self.sub + '  ' + self.run + '  ' + self.foot)
            plt.legend()
            plt.ylabel('Gyro Magnitude (rad/s)')
            plt.xlabel('Sample Number')
            self.ax.margins(x=0)

            axcolor = '0.9'
            axcount = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            axgyro_mag = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

            self.scount = Slider(axcount, 'Count Threshold', 3, 30, valinit=c0, valstep=delta_c)
            self.sgyro_mag = Slider(axgyro_mag, 'Gyro Mag. Threshold', 0.1, 2.5, valinit=g0, valstep=delta_g)
            # The vline attribute controls the initial value line
            self.scount.vline.set_color('coral')
            self.sgyro_mag.vline.set_color('coral')

            self.scount.on_changed(self.update)
            self.sgyro_mag.on_changed(self.update)

            resetax = plt.axes([0.65, 0.025, 0.1, 0.04])
            self.button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.7')
            self.button_reset.on_clicked(self.reset)

            saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
            self.button_save = Button(saveax, 'Save', color='coral')
            self.button_save.on_clicked(self.save)
            self.button_save.on_clicked(self.change_color)

            plt.show()

        # add data to csv file
        gyro_thresholds.append(
            [
                self.sub,
                self.run,
                round(self.stance_magnitude_thresholds["LF"], 2),
                round(self.stance_magnitude_thresholds["RF"], 2),
                round(self.stance_count_thresholds["LF"], 2),
                round(self.stance_count_thresholds["RF"], 2)
            ]
        )

        with open(os.path.join(self.save_path, 'stance_magnitude_thresholds.csv'), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(gyro_thresholds)

        self.check_duplicates(os.path.join(self.save_path, 'stance_magnitude_thresholds.csv'))
