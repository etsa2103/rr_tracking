#!/usr/bin/env python3
import sys, time, rospy, cv2
import pandas as pd, numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import pyqtgraph as pg


class RosGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resperatory Rate Tracking GUI")
        self.resize(1400, 1000)
        
        self.mask_checkbox = QCheckBox("Use Background Mask")
        self.mask_checkbox.setChecked(rospy.get_param("/rr_tracking/use_mask", False))
        self.mask_checkbox.stateChanged.connect(self.toggle_mask)

        # === BPM Display ===
        self.bpm_label = QLabel("BPM: --   AVG: --")
        self.bpm_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.bpm_label.setAlignment(Qt.AlignCenter)

        # === Video Display ===
        self.image1_label = QLabel(alignment=Qt.AlignCenter)
        self.image2_label = QLabel(alignment=Qt.AlignCenter)
        self.image1_label.setMinimumSize(700, 520)
        self.image2_label.setMinimumSize(700, 520)

        self.label1_title = QLabel("TMOI", alignment=Qt.AlignCenter)
        self.label1_title.setFont(QFont("Arial", 18))
        self.label2_title = QLabel("MROI", alignment=Qt.AlignCenter)
        self.label2_title.setFont(QFont("Arial", 18))

        # === Graph ===
        self.plot_widget = pg.PlotWidget(title="Ground Truth vs. Raw Signal")
        self.plot_widget.addLegend()
        self.csv_curve = self.plot_widget.plot(pen='y', name="CSV Force (centered)")
        self.raw_curve = self.plot_widget.plot(pen='c', name="Raw Signal")
        
        # === Layouts for video display ===
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
        # === Main Layout ===
        layout = QVBoxLayout()
        layout.addWidget(self.bpm_label)
        layout.addLayout(video_layout)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.mask_checkbox)
        self.setLayout(layout)

        # === Initialize ROS and Subscribers ===
        rospy.init_node("gui_node", anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber("/boson/image_raw", Image, self.image_raw_cb)
        rospy.Subscriber("/rr_tracking/image_annotated", Image, self.image_annotated_cb)
        rospy.Subscriber("/rr_tracking/image_roi", Image, self.image_roi_cb)
        
        rospy.Subscriber("/rr_tracking/raw_signal", Float32, self.raw_signal_callback)
        rospy.Subscriber("/rr_tracking/tracking_stable", Bool, self.tracking_stable_callback)
        
        rospy.Subscriber("/rr_tracking/rr_inst", Float32, self.rr_inst_callback)
        rospy.Subscriber("/rr_tracking/rr_avg", Float32, self.rr_avg_callback)

        # === Initialize Variables ===
        self.img_raw = self.img_annotated = self.img_roi = None
        self.rr_inst = None
        self.rr_avg = None
        self.csv_data = self.time_data = []
        self.csv_index = 0
        self.raw_data, self.raw_times = [], []
        self.start_time = time.time()
        self.csv_loaded = False
        self.duration_sec = 90
        self.tracking_stable = True
        
        # === Initialize Plot ===
        self.plot_y_range = (-100, 100)
        csv_path = ""#"/home/etsa/boson_recordings/new_tests/fast_shallow/fast_shallow.csv" 
        if csv_path:
            self.load_csv(csv_path)
        else:
            self.plot_widget.setXRange(0, self.duration_sec)
            self.plot_widget.setYRange(self.plot_y_range[0], self.plot_y_range[1])

        # === Initialize Timers ===
        self.gui_timer = QTimer(timeout=self.update_gui)
        self.gui_timer.start(30)
        self.graph_timer = QTimer(timeout=self.update_plot)
        self.graph_timer.setSingleShot(True)
        self.schedule_next_plot()
    def toggle_mask(self, state):
        mask_enabled = state == Qt.Checked
        rospy.set_param("/rr_tracking/use_mask", mask_enabled)

    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            times = df["Data Set 1:Time(s)"].dropna().tolist()
            forces = df["Data Set 1:Force(N)"].dropna().tolist()
            mid = (min(forces) + max(forces)) / 2
            self.time_data = times
            self.csv_data = [f - mid for f in forces]
            self.plot_widget.setXRange(min(times), max(times))
            self.plot_widget.setYRange(self.plot_y_range[0], self.plot_y_range[1])
            self.csv_loaded = True
            print(f"Loaded {len(self.csv_data)} CSV points.")
        except Exception as e:
            print(f"CSV Error: {e}")
            self.csv_loaded = False

    def tracking_stable_callback(self, msg):
        self.tracking_stable = msg.data
        self.update_bpm_label()
        
    def rr_inst_callback(self, msg):
        self.rr_inst = msg.data
        self.update_bpm_label()

    def rr_avg_callback(self, msg):
        self.rr_avg = msg.data
        self.update_bpm_label()

    def update_bpm_label(self):
        inst = f"{self.rr_inst:.1f}" if self.rr_inst is not None else "--"
        avg = f"{self.rr_avg:.1f}" if self.rr_avg is not None else "--"
        tracking = f"{self.tracking_stable:.1f}" if self.tracking_stable is not None else "--"
        self.bpm_label.setText(f"BPM: {inst}   AVG: {avg} Tracking: {tracking}")

    def raw_signal_callback(self, msg):
        t = time.time() - self.start_time
        self.raw_times.append(t)
        self.raw_data.append(msg.data)
        self.raw_data = self.raw_data[-2000:]
        self.raw_times = self.raw_times[-2000:]
        
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
            mask_enabled = rospy.get_param("/rr_tracking/use_mask", True)
            if mask_enabled:
                mono16 = self.bridge.imgmsg_to_cv2(msg, 'mono16')
        
                # Use grayscale for mask
                #threshold = np.percentile(self.img_raw, 70)
                threshold = 13000  # Adjusted threshold for warm pixel detection
                warm_mask = mono16 < threshold  # Shape: (H, W)
                ptp = mono16.ptp()
                norm = ((mono16 - mono16.min()) / ptp * 255).astype(np.uint8) if ptp > 0 else np.zeros_like(mono16, dtype=np.uint8)

                # Convert to RGB for display
                self.img_roi = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
                
                # Apply red color to RGB image where mask is true
                self.img_roi[warm_mask] = [0, 0, 255]
            else:
                mono16 = self.bridge.imgmsg_to_cv2(msg, 'mono16')
                ptp = mono16.ptp()
                norm = ((mono16 - mono16.min()) / ptp * 255).astype(np.uint8) if ptp > 0 else np.zeros_like(mono16, dtype=np.uint8)
                self.img_roi = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

        except Exception as e:
            print(f"roi image Error: {e}")
            
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
            if self.csv_loaded and self.csv_index < len(self.csv_data):
                self.csv_curve.setData(self.time_data[:self.csv_index], self.csv_data[:self.csv_index])
                self.raw_curve.setData(self.raw_times, self.raw_data)
                self.csv_index += 1
                self.schedule_next_plot()
            elif self.csv_loaded:
                # Stop updating when done
                self.csv_curve.setData(self.time_data, self.csv_data)
                self.raw_curve.setData(self.raw_times, self.raw_data)
                self.graph_timer.stop()
            else:
                if self.raw_times:
                    x_start = max(0, self.raw_times[-1] - self.duration_sec)
                    x_end = self.raw_times[-1]
                    self.plot_widget.setXRange(x_start, x_end)
                else:
                    self.plot_widget.setXRange(0, self.duration_sec)

                self.plot_widget.setYRange(self.plot_y_range[0], self.plot_y_range[1])
                self.raw_curve.setData(self.raw_times, self.raw_data)
                self.schedule_next_plot()
        except Exception as e:
            rospy.logerr(f"[GUI] error: {e}")

    def schedule_next_plot(self):
        try:
            if self.csv_loaded and self.csv_index < len(self.time_data) - 1:
                dt = (self.time_data[self.csv_index + 1] - self.time_data[self.csv_index]) * 1000
                self.graph_timer.start(max(1, int(dt)))
            else:
                self.graph_timer.start(1000 // 30)
        except Exception as e:
            rospy.logerr(f"[GUI] error: {e}")

    def reset_graph(self):
        try:
            self.csv_index = 0
            self.start_time = time.time()
            self.raw_data.clear()
            self.raw_times.clear()
            self.csv_curve.setData([], [])
            self.raw_curve.setData([], [])
            self.schedule_next_plot()
        except Exception as e:
            rospy.logerr(f"[GUI] error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RosGui()
    gui.show()
    app.aboutToQuit.connect(lambda: rospy.signal_shutdown("Closed"))

    sys.exit(app.exec_())
