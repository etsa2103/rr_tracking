#!/usr/bin/env python3
import sys, time, rospy, cv2
import pandas as pd, numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import pyqtgraph as pg


class RosGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS Video + BPM + Graph")
        self.resize(1400, 1000)

        self.bpm_label = QLabel("BPM: --")
        self.bpm_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.bpm_label.setAlignment(Qt.AlignCenter)

        self.image1_label = QLabel(alignment=Qt.AlignCenter)
        self.image2_label = QLabel(alignment=Qt.AlignCenter)

        video_layout = QHBoxLayout()
        video_layout.addStretch(1)
        video_layout.addWidget(self.image1_label)
        video_layout.addWidget(self.image2_label)
        video_layout.addStretch(1)

        self.plot_widget = pg.PlotWidget(title="Ground Truth vs. Raw Signal")
        self.plot_widget.addLegend()
        self.csv_curve = self.plot_widget.plot(pen='y', name="CSV Force (centered)")
        self.raw_curve = self.plot_widget.plot(pen='c', name="Raw Signal")

        layout = QVBoxLayout()
        layout.addWidget(self.bpm_label)
        layout.addLayout(video_layout)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        rospy.init_node("gui_node", anonymous=True)
        self.bridge = CvBridge()

        rospy.Subscriber("/boson/image_annotated", Image, self.image_callback1)
        rospy.Subscriber("/boson/image_roi", Image, self.image_callback2)
        rospy.Subscriber("/rr_tracking/rr_inst", Float32, self.bpm_callback)
        rospy.Subscriber("/rr_tracking/raw_signal", Float32, self.raw_signal_callback)

        self.img1 = self.img2 = None
        self.csv_data = self.time_data = []
        self.csv_index = 0
        self.raw_data, self.raw_times = [], []
        self.start_time = time.time()

        self.csv_loaded = False
        self.duration_sec = 90  # window size when CSV is not loaded

        csv_path = "/home/etsa/boson_recordings/new_tests/slow_shallow/slow_shallow3.csv"
        if csv_path:
            self.load_csv(csv_path)
        else:
            self.plot_widget.setXRange(0, self.duration_sec)
            self.plot_widget.setYRange(-25, 25)

        self.gui_timer = QTimer(timeout=self.update_gui)
        self.gui_timer.start(30)

        self.graph_timer = QTimer(timeout=self.update_plot)
        self.graph_timer.setSingleShot(True)
        self.schedule_next_plot()

    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            times = df["Data Set 1:Time(s)"].dropna().tolist()
            forces = df["Data Set 1:Force(N)"].dropna().tolist()
            mid = (min(forces) + max(forces)) / 2
            self.time_data = times
            self.csv_data = [f - mid for f in forces]
            self.plot_widget.setXRange(min(times), max(times))
            self.plot_widget.setYRange(-25, 25)
            self.csv_loaded = True
            print(f"Loaded {len(self.csv_data)} CSV points.")
        except Exception as e:
            print(f"CSV Error: {e}")
            self.csv_loaded = False

    def bpm_callback(self, msg):
        self.bpm_label.setText(f"BPM: {msg.data:.1f}")

    def raw_signal_callback(self, msg):
        t = time.time() - self.start_time
        self.raw_times.append(t)
        self.raw_data.append(msg.data)
        self.raw_data = self.raw_data[-2000:]
        self.raw_times = self.raw_times[-2000:]

    def image_callback1(self, msg):
        try:
            self.img1 = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(f"Image 1 Error: {e}")

    def image_callback2(self, msg):
        try:
            mono16 = self.bridge.imgmsg_to_cv2(msg, 'mono16')
            norm = ((mono16 - mono16.min()) / (mono16.ptp()) * 255).astype(np.uint8) if mono16.ptp() > 0 else np.zeros_like(mono16, dtype=np.uint8)
            self.img2 = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"Image 2 Error: {e}")

    def update_gui(self):
        self.show_image(self.img1, self.image1_label)
        if self.img1 is not None and self.img2 is not None:
            resized = cv2.resize(self.img2, self.img1.shape[1::-1])
            self.show_image(resized, self.image2_label)
        else:
            self.show_image(self.img2, self.image2_label)

    def show_image(self, img, label):
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg))

    def update_plot(self):
        if self.csv_loaded and self.csv_index < len(self.csv_data):
            self.csv_curve.setData(self.time_data[:self.csv_index], self.csv_data[:self.csv_index])
            self.raw_curve.setData(self.raw_times, self.raw_data)
            self.csv_index += 1
            self.schedule_next_plot()
        elif self.csv_loaded:
            self.reset_graph()
        else:
            if self.raw_times:
                x_start = max(0, self.raw_times[-1] - self.duration_sec)
                x_end = self.raw_times[-1]
                self.plot_widget.setXRange(x_start, x_end)
            else:
                self.plot_widget.setXRange(0, self.duration_sec)

            self.plot_widget.setYRange(-25, 25)
            self.raw_curve.setData(self.raw_times, self.raw_data)
            self.schedule_next_plot()


    def schedule_next_plot(self):
        if self.csv_loaded and self.csv_index < len(self.time_data) - 1:
            dt = (self.time_data[self.csv_index + 1] - self.time_data[self.csv_index]) * 1000
            self.graph_timer.start(max(1, int(dt)))
        else:
            self.graph_timer.start(1000 // 30)

    def reset_graph(self):
        self.csv_index = 0
        self.start_time = time.time()
        self.raw_data.clear()
        self.raw_times.clear()
        self.csv_curve.setData([], [])
        self.raw_curve.setData([], [])
        self.schedule_next_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RosGui()
    gui.show()
    app.aboutToQuit.connect(lambda: rospy.signal_shutdown("Closed"))
    sys.exit(app.exec_())
