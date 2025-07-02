#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool

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
        self.buffer_size = 300                # ~10 seconds
        self.min_peak_distance = 2            # seconds

        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=1350)
        self.peak_buffer = deque(maxlen=1350)
        
        self.tracking_stable = True

        rospy.Subscriber("/boson/image_roi", Image, self.image_cb)
        rospy.Subscriber("/rr_tracking/tracking_stable", Bool, self.stability_cb)
        self.raw_pub = rospy.Publisher("/rr_tracking/raw_signal", Float32, queue_size=1)
        self.filtered_pub = rospy.Publisher("/rr_tracking/filtered_signal", Float32, queue_size=1)
        self.rr_inst_pub = rospy.Publisher("/rr_tracking/rr_inst", Float32, queue_size=1)
        self.rr_inst_timestamp_pub = rospy.Publisher("/rr_tracking/rr_inst_timestamp", Float32, queue_size=1)
        self.rr_avg_pub = rospy.Publisher("/rr_tracking/rr_avg", Float32, queue_size=1)

    def stability_cb(self, msg):
        self.tracking_stable = msg.data
        
    def image_cb(self, msg):
        # if(not self.tracking_stable):
        #     # If tracking is not stable, do not process the image
        #     self.raw_pub.publish(Float32(0.0))
        #     self.filtered_pub.publish(Float32(0.0))
        #     self.rr_inst_pub.publish(Float32(0.0))
        #     self.rr_inst_timestamp_pub.publish(Float32(0.0))
        #     self.rr_avg_pub.publish(Float32(0.0))
        #     return
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

        # ====== Clean up raw signal ======
        # Normalize signal
        baseline = uniform_filter1d(signal, size=90)  # ~3 seconds at 30 Hz
        normalized = signal - baseline
        # clip and Invert signal for peak detection
        inverted = np.clip(-normalized,-100.0, 100.0)
        # Publish the latest raw signal value
        self.raw_pub.publish(Float32(inverted[-1]))
        self.peak_buffer.append(inverted[-1])
        
        # ====== Filter signal ======
        filtered = bandpass(inverted, fs=self.frame_rate)
        # Publish the latest filtered signal value
        self.filtered_pub.publish(Float32(filtered[-1]))
        
        # ====== Peak detection ======
        min_samples = int(self.min_peak_distance * self.frame_rate)
        peak_signal = np.array(self.peak_buffer)
        peaks, _ = find_peaks(peak_signal, distance=min_samples, prominence=15, height=7)
        if len(peaks) >= 2:
            # Calculate instantaneous respiration rate
            rr_inst = 60.0/ np.diff(times[peaks][-2:])
            # Publish the instantaneous respiration rate
            self.rr_inst_pub.publish(Float32(rr_inst))
            self.rr_inst_timestamp_pub.publish(Float32(rospy.Time.now().to_sec()))
            
            # Calculate average respiration rate
            intervals = np.diff(times[peaks])
            avg_interval = np.mean(intervals)
            rr_avg = 60.0 / avg_interval
            # Publish the average respiration rate
            self.rr_avg_pub.publish(Float32(rr_avg))
        else:
            # Not enough peaks detected, reset BPM
            self.rr_inst_pub.publish(Float32(0.0))
            self.rr_inst_timestamp_pub.publish(Float32(0.0))
            self.rr_avg_pub.publish(Float32(0.0))

if __name__ == "__main__":
    try:
        SimpleBreathTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass