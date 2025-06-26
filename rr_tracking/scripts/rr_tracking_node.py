#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from cv_bridge import CvBridge

import time
import numpy as np
from collections import deque


from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, detrend, butter, filtfilt
from scipy.ndimage import uniform_filter1d

def bandpass(data, fs, low=0.1, high=0.7):
            nyq = 0.5 * fs
            b, a = butter(4, [low / nyq, high / nyq], btype='band')
            return filtfilt(b, a, data)

class SimpleBreathTracker:
    def __init__(self):
        rospy.init_node("simple_rr_tracker")
        self.bridge = CvBridge()

        self.frame_rate = 30                  # Hz
        self.buffer_size = 450                # ~10 seconds
        self.min_peak_distance = 2            # seconds

        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)

        rospy.Subscriber("/boson/image_roi", Image, self.image_cb)
        self.raw_pub = rospy.Publisher("/rr_tracking/raw_signal", Float32, queue_size=1)
        self.filtered_pub = rospy.Publisher("/rr_tracking/filtered_signal", Float32, queue_size=1)
        self.bpm_pub = rospy.Publisher("/rr_tracking/breath_rate", Float32, queue_size=1)
        self.bpm_timestamp_pub = rospy.Publisher("/rr_tracking/bpm_timestamp", Float32, queue_size=1)
        

    def image_cb(self, msg):
        # Get mean pixel value from image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        if img is None or img.size == 0:
            return
        mean_val = np.mean(img)
        now = time.time()
        # Add to buffer
        self.signal_buffer.append(mean_val)
        self.time_buffer.append(now)
        if len(self.signal_buffer) < self.buffer_size:
            return
        signal = np.array(self.signal_buffer)
        times = np.array(self.time_buffer)

        # publish the latest raw signal value
        baseline = uniform_filter1d(signal, size=90)  # ~2 seconds at 30 Hz
        # Subtract baseline (preserves dips!)
        normalized = signal - baseline
        self.raw_pub.publish(Float32(normalized[-1]))
        
        # ====== Filter signal ======
        signal = gaussian_filter1d(signal, sigma=3)
        # Estimate local baseline using sliding window
        baseline = uniform_filter1d(signal, size=90)  # ~2 seconds at 30 Hz
        # Subtract baseline (preserves dips!)
        normalized = signal - baseline
        filtered = bandpass(normalized, fs=self.frame_rate)
        filtered = filtered * 3
        # publish the latest raw signal value
        self.filtered_pub.publish(Float32(filtered[-1]))
        
        # ====== Peak detection ======
        min_samples = int(self.min_peak_distance * self.frame_rate)
        peaks, _ = find_peaks(filtered, distance=min_samples)
        if len(peaks) >= 2:
            intervals = np.diff(times[peaks])
            avg_interval = np.mean(intervals)
            bpm = 60.0 / avg_interval
            # publish the calculated BPM
            self.bpm_pub.publish(Float32(bpm))
            self.bpm_timestamp_pub.publish(Float32(rospy.Time.now().to_sec()))

if __name__ == "__main__":
    try:
        SimpleBreathTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
