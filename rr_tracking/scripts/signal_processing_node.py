#!/usr/bin/env python3
import rospy
import time
import threading
import numpy as np
from collections import deque
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool, String, UInt8
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# === Main Signal Processing Class ===
class SignalProcessingNode:
    def __init__(self):
        rospy.init_node("simple_rr_tracker")
        self.bridge = CvBridge()

        # Subscribers & Publishers
        rospy.Subscriber("/facial_tracking/image_roi", Image, self.image_roi_cb)
        rospy.Subscriber("/facial_tracking/tracking_stable", Bool, self.stability_cb)
        rospy.Subscriber("/facial_tracking/pose_type", String, self.pose_type_cb)
        rospy.Subscriber('/jackal_teleop/trigger', UInt8, self.trigger_cb)

        self.raw_pub = rospy.Publisher("/rr_tracking/raw_signal", Float32, queue_size=1)
        self.rr_inst_pub = rospy.Publisher("/rr_tracking/rr_inst", Float32, queue_size=1)
        self.rr_avg_pub = rospy.Publisher("/rr_tracking/rr_avg", Float32, queue_size=1)
        self.rr_final_pub = rospy.Publisher("/rr_tracking/rr_final", Float32, queue_size=1)
        self.recording_pub = rospy.Publisher("/rr_tracking/recording", Bool, queue_size=1)

        # Runtime State
        self.recording = False
        self.recorded_raw_images = []
        self.recorded_roi_images = []
        self.record_start_time = None  
        
        self.image_raw = None
        self.image_roi = None
        self.frame_rate = 30
        self.min_peak_distance = 0.8
        self.tracking_stable = True
        self.pose_type = "unknown"

        self.signal_buffer = deque(maxlen=self.frame_rate*10) # 10 seconds of signal data
        self.time_buffer = deque(maxlen=self.frame_rate*45) # 45 seconds of timestamps
        self.peak_buffer = deque(maxlen=self.frame_rate*45) # 45 seconds of peaks
    
    # ================================================================================================================================      
    # ====================================================== Callback Functions ======================================================
    # ================================================================================================================================
    
    def trigger_cb(self, msg):
        if msg.data > 0 and not self.recording:
            self.recording = True
            self.recording_pub.publish(Bool(True))

            self.recorded_roi_images = []
            self.record_start_time = time.time()
            rospy.loginfo("Started recording 20 seconds of frames.")  
            
    def stability_cb(self, msg): self.tracking_stable = msg.data
    
    def pose_type_cb(self, msg): self.pose_type = msg.data

    def image_roi_cb(self, msg):
        # === Get the raw image from the topic ===
        self.image_roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        rospy.logdebug(f"[rr_tracking] Received ROI image of shape: {self.image_roi.shape}")
        # === Check if we are recording ===
        if self.recording:
            self.recorded_roi_images.append(self.image_roi.copy())
            if time.time() - self.record_start_time >= 20:
                self.recording = False
                rospy.loginfo("Finished recording. Processing...")
                if rospy.get_param("/enable_gui", False):
                    threading.Thread(target=self.process_recorded_data, daemon=True).start()
                else:
                    self.process_recorded_data()

        # Check if GUI is enabled via parameter
        if rospy.get_param("/enable_gui", False):
            self.process_live_data()
            
    # ================================================================================================================================      
    # ======================================================= Helper Functions =======================================================
    # ================================================================================================================================
    def process_live_data(self):
        # === Decode ROI ===
        rospy.logdebug(f"[rr_tracking] Received ROI image of shape: {self.image_roi.shape}")
        if self.image_roi is None or self.image_roi.size == 0:
            rospy.logwarn_throttle(1.0, "[rr_tracking] Empty ROI image received.")
            return
        
        # === ROI Masking & Warm Pixel Extraction ===
        use_mask = rospy.get_param("/rr_tracking/use_mask", True)
        if use_mask:
            #threshold = np.percentile(self.image_raw, 70)
            threshold = 13000  # Adjusted threshold for warm pixel detection
            warm_pixels = self.image_roi[self.image_roi > threshold]
        else:
            warm_pixels = self.image_roi.flatten()

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
            
    def process_recorded_data(self):
        if not self.recorded_roi_images:
            rospy.logwarn("No recorded frames to process.")
            return

        # === Step 1: Extract warm pixel signal from each frame ===
        use_mask = rospy.get_param("/rr_tracking/use_mask", True)
        signal = []
        timestamps = []

        for i, frame in enumerate(self.recorded_roi_images):
            if use_mask and self.image_raw is not None:
                #threshold = np.percentile(self.image_raw, 70)
                threshold = 13000  # Adjusted threshold for warm pixel detection
                warm_pixels = frame[frame > threshold]
            else:
                warm_pixels = frame.flatten()

            if warm_pixels.size == 0 or warm_pixels.mean() < 0.3:
                continue

            signal.append(np.mean(warm_pixels))
            timestamps.append(self.record_start_time + i / self.frame_rate)

        signal = np.array(signal)
        timestamps = np.array(timestamps)

        if len(signal) < 2:
            rospy.logwarn("Insufficient valid data in recorded buffer.")
            return

        # === Step 2: Filter + Normalize ===
        baseline = uniform_filter1d(signal, size=self.frame_rate*4)
        normalized = signal - baseline
        inverted = np.clip(-normalized, -100, 100)

        # === Step 3: Peak Detection ===
        min_samples = int(self.min_peak_distance * self.frame_rate)
        min_prominence = 18 if self.pose_type in ["left", "right"] else 9
        peaks, _ = find_peaks(inverted, distance=min_samples, prominence=min_prominence)

        # === Step 4: Respiration Rate Calculation ===
        if len(peaks) >= 2:
            peak_times = timestamps[peaks]
            rr_inst = 60.0 / np.diff(peak_times[-2:])[0]
            rr_avg = 60.0 / np.mean(np.diff(peak_times))

            rospy.loginfo(f"[Offline RR] BPM: {rr_inst:.1f} AVG: {rr_avg:.1f}")
            self.rr_inst_pub.publish(Float32(rr_inst))
            self.rr_avg_pub.publish(Float32(rr_avg))
            self.rr_final_pub.publish(Float32(rr_avg))  # ← Publish final average BPM
        else:
            rospy.logwarn("[Offline RR] Not enough peaks to calculate RR.")
            self.rr_inst_pub.publish(Float32(0.0))
            self.rr_avg_pub.publish(Float32(0.0))
            self.rr_final_pub.publish(Float32(0.0))  # ← Still publish a 0

        # === OPTIONAL: Plot the processed signal with peaks ===
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(timestamps, inverted, label="Filtered Signal", color="blue")
            plt.plot(np.array(timestamps)[peaks], inverted[peaks], "rx", label="Detected Peaks")
            plt.title("Recorded Breathing Signal (Processed)")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.grid(True)
            plt.show()  # or plt.savefig("/tmp/rr_plot.png") to save instead
        except Exception as e:
            rospy.logwarn(f"Failed to plot signal: {e}")
        self.recording_pub.publish(Bool(False))


# ========= Main =========
if __name__ == "__main__":
    try:
        SignalProcessingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

# ===================================
# ======== Utility Functions ========
# ===================================
def bandpass(data, fs, low=0.1, high=0.7):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)