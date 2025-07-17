#!/usr/bin/env python3
import sys, time, rospy, cv2
if not rospy.get_param("/enable_gui", False):
    rospy.loginfo("Running in headless mode, GUI disabled.")
    pass

import pandas as pd, numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool, Float64MultiArray
from cv_bridge import CvBridge
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

class RosGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resperatory Rate Tracking GUI")
        self.resize(1400, 1000)

        self.mask_checkbox = QCheckBox("Use Background Mask")
        self.mask_checkbox.setChecked(rospy.get_param("/rr_tracking/use_mask", False))
        self.mask_checkbox.stateChanged.connect(self.toggle_mask)

        self.bpm_label = QLabel("BPM: --   AVG: --")
        self.bpm_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.bpm_label.setAlignment(Qt.AlignCenter)

        self.image1_label = QLabel(alignment=Qt.AlignCenter)
        self.image2_label = QLabel(alignment=Qt.AlignCenter)
        self.image1_label.setMinimumSize(700, 520)
        self.image2_label.setMinimumSize(700, 520)

        self.label1_title = QLabel("TMOI", alignment=Qt.AlignCenter)
        self.label1_title.setFont(QFont("Arial", 18))
        self.label2_title = QLabel("MROI", alignment=Qt.AlignCenter)
        self.label2_title.setFont(QFont("Arial", 18))

        self.plot_widget = pg.PlotWidget(title="Ground Truth vs. Raw Signal")
        self.plot_widget.addLegend()
        self.csv_curve = self.plot_widget.plot(pen='y', name="CSV Force (centered)")
        self.raw_curve = self.plot_widget.plot(pen='c', name="Raw Signal")
        self.peak_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush('r'), size=8, name="Detected Peaks")
        self.plot_widget.addItem(self.peak_scatter)
        self.segment_lines = []

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.label1_title)
        vbox1.addWidget(self.image1_label)
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.label2_title)
        vbox2.addWidget(self.image2_label)
        video_layout = QHBoxLayout()
        video_layout.addStretch(1)
        video_layout.addLayout(vbox1)
        video_layout.addLayout(vbox2)
        video_layout.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.bpm_label)
        layout.addLayout(video_layout)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.mask_checkbox)
        self.setLayout(layout)

        rospy.init_node("gui_node", anonymous=True)
        self.bridge = CvBridge()

        self.start_time = rospy.get_time()
        self.csv_data = self.time_data = []
        self.csv_index = 0
        self.csv_loaded = False
        self.duration_sec = 90
        
        self.img_raw = None
        self.img_annotated = None


        self.plot_y_range = (-100, 100)
        csv_path = ""  # Leave blank unless loading CSV
        if csv_path:
            self.load_csv(csv_path)
        else:
            self.plot_widget.setXRange(0, self.duration_sec)
            self.plot_widget.setYRange(*self.plot_y_range)

        self.gui_timer = QTimer(timeout=self.update_gui)
        self.gui_timer.start(30)
        self.graph_timer = QTimer(timeout=self.update_plot)
        self.graph_timer.setSingleShot(True)
        self.schedule_next_plot()

    # ====================================================== ROS Callbacks ======================================================

    def image_raw_cb(self, msg):
        try:
            self.img_raw = self.bridge.imgmsg_to_cv2(msg, 'mono16')
        except Exception as e:
            print(f"Raw Image Error: {e}")

    def image_annotated_cb(self, msg):
        try:
            self.img_annotated = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(f"annotated image Error: {e}")

    def image_roi_cb(self, msg):
        try:
            mask_enabled = rospy.get_param("/rr_tracking/use_mask", False)
            mono16 = self.bridge.imgmsg_to_cv2(msg, 'mono16')
            ptp = mono16.ptp()
            norm = ((mono16 - mono16.min()) / ptp * 255).astype(np.uint8) if ptp > 0 else np.zeros_like(mono16, dtype=np.uint8)
            self.img_roi = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
            if mask_enabled and self.img_raw is not None:
                threshold = 30400
                warm_mask = mono16 < threshold
                self.img_roi[warm_mask] = [0, 0, 255]
        except Exception as e:
            print(f"roi image Error: {e}")

    # ======================================================= GUI Updates =======================================================
    def update_bpm_label(self):
        if self.recording:
            self.bpm_label.setText("Recording... Please wait.")
        else:
            inst = f"{self.rr_inst:.1f}" if self.rr_inst is not None else "--"
            avg = f"{self.rr_avg:.1f}" if self.rr_avg is not None else "--"
            tracking = f"{self.tracking_stable:.1f}" if self.tracking_stable is not None else "--"
            self.bpm_label.setText(f"BPM: {inst}   AVG: {avg} Tracking: {tracking}")

    def toggle_mask(self, state):
        rospy.set_param("/rr_tracking/use_mask", state == Qt.Checked)

    def update_gui(self):
        self.show_image(self.img_annotated, self.image1_label)
        if self.img_annotated is not None and self.img_roi is not None:
            resized = cv2.resize(self.img_roi, self.img_annotated.shape[1::-1])
            self.show_image(resized, self.image2_label)
        else:
            self.show_image(self.img_roi, self.image2_label)

    def show_image(self, img, label):
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_plot(self):
        try:
            # Always draw the raw curve
            if self.raw_times and self.raw_data:
                x_start = max(0, self.raw_times[-1] - self.duration_sec)
                x_end = self.raw_times[-1]
                self.plot_widget.setXRange(x_start, x_end)
                self.plot_widget.setYRange(*self.plot_y_range)
                self.raw_curve.setData(self.raw_times, self.raw_data)

            # If CSV is loaded, update that too
            if self.csv_loaded:
                if self.csv_index < len(self.csv_data):
                    self.csv_curve.setData(self.time_data[:self.csv_index], self.csv_data[:self.csv_index])
                    self.csv_index += 1
                    self.schedule_next_plot()
                else:
                    self.csv_curve.setData(self.time_data, self.csv_data)
                    self.graph_timer.stop()

            # Draw peaks (only if both peak data and raw data exist)
            if self.latest_peaks and self.raw_times:
                spots = []
                for t in self.latest_peaks:
                    t=t-self.start_time
                    closest = min(self.raw_times, key=lambda rt: abs(rt - t))
                    if abs(closest - t) < 1:
                        idx = self.raw_times.index(closest)
                        y = self.raw_data[idx]
                        spots.append({'pos': (closest, y), 'symbol': 'x'})
                self.peak_scatter.setData(spots)
            else:
                self.peak_scatter.setData([])

            # Draw segment bounds
            for line in self.segment_lines:
                self.plot_widget.removeItem(line)
            self.segment_lines.clear()
            if self.latest_segment_bounds:
                for seg_start, seg_end in self.latest_segment_bounds:
                    seg_start = seg_start-self.start_time
                    seg_end = seg_start-self.start_time
                    line_start = pg.InfiniteLine(pos=seg_start, angle=90, pen=pg.mkPen('g', style=Qt.DashLine))
                    line_end = pg.InfiniteLine(pos=seg_end, angle=90, pen=pg.mkPen('g', style=Qt.DashLine))
                    self.plot_widget.addItem(line_start)
                    self.plot_widget.addItem(line_end)
                    self.segment_lines.extend([line_start, line_end])

            self.schedule_next_plot()

        except Exception as e:
            rospy.logerr(f"[rr_tracking/gui] error in update_plot: {e}")


    def schedule_next_plot(self):
        try:
            if self.csv_loaded and self.csv_index < len(self.time_data) - 1:
                dt = (self.time_data[self.csv_index + 1] - self.time_data[self.csv_index]) * 1000
                self.graph_timer.start(max(1, int(dt)))
            else:
                self.graph_timer.start(1000 // 30)
        except Exception as e:
            rospy.logerr(f"[rr_tracking/gui] error: {e}")

    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            times = df["Data Set 1:Time(s)"].dropna().tolist()
            forces = df["Data Set 1:Force(N)"].dropna().tolist()
            mid = (min(forces) + max(forces)) / 2
            self.time_data = times
            self.csv_data = [f - mid for f in forces]
            self.plot_widget.setXRange(min(times), max(times))
            self.plot_widget.setYRange(*self.plot_y_range)
            self.csv_loaded = True
            print(f"Loaded {len(self.csv_data)} CSV points.")
        except Exception as e:
            print(f"CSV Error: {e}")
            self.csv_loaded = False

    def reset_graph(self):
        try:
            self.csv_index = 0
            self.start_time = rospy.get_time()
            self.raw_data.clear()
            self.raw_times.clear()
            self.csv_curve.setData([], [])
            self.raw_curve.setData([], [])
            self.schedule_next_plot()
        except Exception as e:
            rospy.logerr(f"[rr_tracking/gui] error: {e}")


if __name__ == "__main__":
    if not rospy.get_param("/enable_gui", False):
        rospy.loginfo("GUI is disabled via parameter.")
        sys.exit(0)

    app = QApplication(sys.argv)
    gui = RosGui()
    gui.show()
    app.aboutToQuit.connect(lambda: rospy.signal_shutdown("Closed"))

    sys.exit(app.exec_())
