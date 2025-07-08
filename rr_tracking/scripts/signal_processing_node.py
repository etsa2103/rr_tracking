#!/usr/bin/env python3
import rospy
import time
import numpy as np
from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool, String
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

def bandpass(data, fs, low=0.1, high=0.7):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

class SimpleBreathTracker:
    def __init__(self):
        rospy.init_node("simple_rr_tracker")
        self.bridge = CvBridge()

        # === Subscribers & Publishers ===
        rospy.Subscriber("/boson/image_raw", Image, self.image_raw_cb)
        rospy.Subscriber("/rr_tracking/image_roi", Image, self.image_cb)
        rospy.Subscriber("/rr_tracking/tracking_stable", Bool, self.stability_cb)
        rospy.Subscriber("/rr_tracking/pose_type", String, self.pose_type_cb)

        self.raw_pub = rospy.Publisher("/rr_tracking/raw_signal", Float32, queue_size=1)
        self.rr_inst_pub = rospy.Publisher("/rr_tracking/rr_inst", Float32, queue_size=1)
        self.rr_avg_pub = rospy.Publisher("/rr_tracking/rr_avg", Float32, queue_size=1)

        # === Runtime State ===
        self.image_raw = None
        self.image_roi = None
        self.frame_rate = 30
        self.min_peak_distance = 0.8
        self.tracking_stable = True
        self.pose_type = "unknown"

        self.signal_buffer = deque(maxlen=self.frame_rate*10) # 10 seconds of signal data
        self.time_buffer = deque(maxlen=self.frame_rate*45) # 45 seconds of timestamps
        self.peak_buffer = deque(maxlen=self.frame_rate*45) # 45 seconds of peaks

    def stability_cb(self, msg): self.tracking_stable = msg.data
    def pose_type_cb(self, msg): self.pose_type = msg.data
    
    def image_raw_cb(self, msg): 
        try:
            self.image_raw = self.bridge.imgmsg_to_cv2(msg, 'mono16')
        except Exception as e:
            print(f"Image 1 Error: {e}")
            
    def image_cb(self, msg):
        # === Decode & Preprocess ROI ===
        
        self.image_roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        if self.image_roi is None or self.image_roi.size == 0:
            rospy.logwarn_throttle(1.0, "[rr_tracking] Empty ROI image received.")
            return
        if self.image_raw is None or self.image_raw.size == 0:
            rospy.logwarn_throttle(1.0, "[rr_tracking] Empty raw image received.")
            return

        threshold = np.percentile(self.image_raw, 70)
        warm_pixels = self.image_roi[self.image_roi > threshold]

        # === Edge Cases & Frame Skipping ===
        if not self.tracking_stable or warm_pixels.size == 0 or warm_pixels.mean() < 0.3:
            rospy.logwarn_throttle(1.0, "[rr_tracking] Unstable or empty ROI — skipping frame.")
            return

        # === Signal Buffering ===
        mean_val = np.mean(warm_pixels)
        now = time.time()
        self.signal_buffer.append(mean_val)
        self.time_buffer.append(now)

        if len(self.signal_buffer) < self.signal_buffer.maxlen:
            rospy.logwarn_throttle(5.0, "[rr_tracking] Not enough data to process yet.")
            return

        # === Signal Normalization & Filtering ===
        signal = np.array(self.signal_buffer)
        baseline = uniform_filter1d(signal, size=self.frame_rate*4)  # 4 seconds baseline
        normalized = signal - baseline
        test = (-100, 100)
        inverted = np.clip(-normalized, *test)  # Clip to avoid extreme values
        self.raw_pub.publish(Float32(inverted[-1]))   # Publish the latest raw signal value
        self.peak_buffer.append(inverted[-1])         # Store the latest value for peak detection

        # === Adaptive Peak Detection & Respiration Rate Estimation ===
        peak_signal = np.array(self.peak_buffer)
        min_samples = int(self.min_peak_distance * self.frame_rate)
        min_prominence = 15 if self.pose_type in ["left", "right"] else 8
        peaks, _ = find_peaks(peak_signal, distance=min_samples, prominence=min_prominence)

        if len(peaks) >= 2 and len(self.time_buffer) >= len(peak_signal):
            peak_times = np.array(self.time_buffer)[-len(peak_signal):][peaks]
            rr_inst = 60.0 / np.diff(peak_times[-2:])[0]
            rr_avg = 60.0 / np.mean(np.diff(peak_times))

            self.rr_inst_pub.publish(Float32(rr_inst))
            self.rr_avg_pub.publish(Float32(rr_avg))
        else:
            self.rr_inst_pub.publish(Float32(0.0))
            self.rr_avg_pub.publish(Float32(0.0))

if __name__ == "__main__":
    try:
        SimpleBreathTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
