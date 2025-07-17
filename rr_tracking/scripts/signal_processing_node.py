#!/usr/bin/env python3
import rospy
import time
# import threading
import numpy as np
from collections import deque, Counter
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool, String, UInt8, Float64MultiArray
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
# from rr_tracking.msg import TrackingState



class SignalProcessingNode:
    def __init__(self):
        rospy.init_node("simple_rr_tracker")
        self.bridge = CvBridge()

        # ROS I/O
        rospy.Subscriber("/facial_tracking/image_roi", Image, self.image_roi_cb)
        rospy.Subscriber("/facial_tracking/tracking_stable", Bool, self.tracking_stable_cb)
        rospy.Subscriber("/facial_tracking/pose_type", String, self.pose_type_cb)
        
        rospy.Subscriber('/jackal_teleop/trigger', UInt8, self.trigger_cb)

        self.peaks_pub = rospy.Publisher("/rr_tracking/peaks", Float64MultiArray, queue_size=1)
        self.segment_bounds_pub = rospy.Publisher("/rr_tracking/segment_bounds", Float64MultiArray, queue_size=1)
        
        self.raw_pub = rospy.Publisher("/rr_tracking/raw_signal", Float32, queue_size=1)
        self.rr_inst_pub = rospy.Publisher("/rr_tracking/rr_inst", Float32, queue_size=1)
        self.rr_avg_pub = rospy.Publisher("/rr_tracking/rr_avg", Float32, queue_size=1)
        self.rr_final_pub = rospy.Publisher("/rr_tracking/rr_final", Float32, queue_size=1)
        self.recording_pub = rospy.Publisher("/rr_tracking/recording", Bool, queue_size=1)

        # Variables
        self.frame_rate = 30
        self.min_peak_distance = 0.8
        
        self.recording = False
        self.recorded_data = []
        self.record_start_time = None
        
        self.image_roi = None
        self.tracking_stable = True
        self.pose_type = "unknown"

        self.signal_buffer = deque(maxlen=self.frame_rate * 10)
        self.stability_buffer = deque(maxlen=self.frame_rate * 45)
        self.pose_buffer = deque(maxlen=self.frame_rate * 45)
        self.time_buffer = deque(maxlen=self.frame_rate * 45)
        self.peak_buffer = deque(maxlen=self.frame_rate * 45)

    # ================================================================================================================================      
    # ====================================================== Callback Functions ======================================================
    # ================================================================================================================================
    def trigger_cb(self, msg):
        if msg.data > 0 and not self.recording:
            self.recording = True
            self.recording_pub.publish(Bool(True))
            self.record_start_time = rospy.get_time()
            rospy.loginfo("Started recording for 20 seconds")
            
            # maybe reset all the buffers for a new recording (or reset all except signal buffer)
            # self.signal_buffer.clear()
            # self.stability_buffer.clear()
            # self.time_buffer = deque(maxlen=self.frame_rate * 45)
            # self.peak_buffer = deque(maxlen=self.frame_rate * 45)
            
    def image_roi_cb(self, msg):
        self.image_roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        self.process_signal()
        rr_inst, rr_avg = self.analyze_signal()
        self.rr_inst_pub.publish(Float32(rr_inst))
        self.rr_avg_pub.publish(Float32(rr_avg))
        
        if self.recording and (rospy.get_time() - self.record_start_time >= 20): #maybe add a way to stop before 20 seconds
                self.recording = False
                rospy.loginfo("Finished recording. Processing...")
                self.rr_final_pub.publish(Float32(rr_avg))
                
    def tracking_stable_cb(self, msg): self.tracking_stable = msg.data
    def pose_type_cb(self, msg): self.pose_type = msg.data
    # ================================================================================================================================      
    # ===================================================== Signal Processing  ======================================================
    # ================================================================================================================================
    def extract_warm_pixels(self, frame, use_mask=True):
        threshold = 30400
        if use_mask:
            return frame[frame > threshold]  
        else:
            return frame.flatten()

    def process_signal(self):
        # Check edge case
        if self.image_roi is None or self.image_roi.size == 0:
            rospy.logwarn_throttle(1.0, "[rr_tracking] Empty ROI image.")
            return
        # Mask background pixels
        use_mask = rospy.get_param("/rr_tracking/use_mask", False)
        warm_pixels = self.extract_warm_pixels(self.image_roi, use_mask)
        # Check edge case
        if not self.tracking_stable or warm_pixels.size == 0 or warm_pixels.mean() < 0.3:
            rospy.logwarn_throttle(1.0, "[rr_tracking] Unstable or invalid ROI — skipping.")
            return
        # Update signal buffers
        mean_val = np.mean(warm_pixels)
        now = rospy.get_time()
        self.signal_buffer.append(mean_val)
        self.stability_buffer.append(self.tracking_stable)
        self.pose_buffer.append(self.pose_type)
        self.time_buffer.append(now)
        # Check edge case
        if len(self.signal_buffer) < self.signal_buffer.maxlen:
            rospy.logwarn_throttle(5.0, "[rr_tracking] Buffering live signal...")
            return
        # Process signal and update peak buffer
        signal = np.array(self.signal_buffer)
        baseline = uniform_filter1d(signal, size=self.frame_rate * 4)
        normalized = signal - baseline
        inverted = np.clip(-normalized, -100, 100)
        self.peak_buffer.append(inverted[-1])
        self.raw_pub.publish(inverted[-1])
        
    def analyze_signal(self):
        segments = []
        start_idx = None
        peak_signal = np.array(self.peak_buffer)
        stabilities = np.array(self.stability_buffer)
        poses = np.array(self.pose_buffer)
        timestamps = np.array(self.time_buffer)
        
        # Get indexes of stable signal segments
        for i, stable in enumerate(stabilities):
            if stable and start_idx is None:
                start_idx = i
            elif not stable and start_idx is not None:
                if timestamps[i - 1] - timestamps[start_idx] >= 5.0:
                    segments.append((start_idx, i))
                start_idx = None
        if start_idx is not None and timestamps[-1] - timestamps[start_idx] >= 5.0:
            segments.append((start_idx, len(stabilities)))
            
        # Check edge case
        if not segments:
            rospy.logwarn_throttle(1.0,"[rr_tracking] No valid stable segments found.")
            return 0.0, 0.0
        
        # For each stable segment get the time of all peaks and calculate rr
        rr_values = []
        segment_bounds_msg = Float64MultiArray()
        peaks_msg = Float64MultiArray()
        for start, end in segments:
            seg_signal = peak_signal[start:end]
            seg_poses = poses[start:end]
            seg_timestamps = timestamps[start:end]

            if len(seg_poses) == 0:
                continue
            most_common_pose = Counter(seg_poses).most_common(1)[0][0]
            min_prominence = 18 if most_common_pose in ["left", "right"] else 9
            min_samples = int(self.min_peak_distance * self.frame_rate)
            peaks, _ = find_peaks(seg_signal, distance=min_samples, prominence=min_prominence)
            peak_times = []
            for peak in peaks:
                peak_times.append(seg_timestamps[peak])
                peaks_msg.data.append(float(seg_timestamps[peak]))
                
            if(len(peak_times) >= 2):
                for i in range(len(peak_times)-1):
                    rr_values.append(60.0/(peak_times[i+1] - peak_times[i]))
            # Update messages for gui
            bounds = [float(timestamps[start]), float(timestamps[end - 1])]
            segment_bounds_msg.data.extend(bounds)  
                    
        # Publish segments and peaks for overlay
        if segment_bounds_msg.data:
            self.segment_bounds_pub.publish(segment_bounds_msg)
        if peaks_msg.data:
            self.peaks_pub.publish(peaks_msg)
            
        # Return resperation rate values across whole signal
        if not rr_values:
            rospy.logwarn_throttle(1.0,"[rr_tracking] Not enough peaks across all segments.")
            return 0.0, 0.0
        else:
            return rr_values[-1], np.mean(rr_values)

# =========================================================================================================
# =========================================== Utility Functions ===========================================
# =========================================================================================================
def bandpass(data, fs, low=0.1, high=0.7):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

# # Plot if GUI enabled
    # if result and rospy.get_param("/enable_gui", False):
    #     ts, inverted, peaks = result
    #     try:
    #         plt.figure(figsize=(10, 4))
    #         plt.plot(ts, inverted, label="Filtered Signal", color="blue")
    #         plt.plot(np.array(ts)[peaks], inverted[peaks], "rx", label="Detected Peaks")
    #         plt.title("Recorded Breathing Signal"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    #         plt.grid(True); plt.tight_layout(); plt.legend(); plt.show()
    #     except Exception as e:
    #         rospy.logwarn(f"[rr_tracking] Failed to plot: {e}")
    # self.recording_pub.publish(Bool(False))

# === Main Entry Point ===
if __name__ == "__main__":
    try:
        SignalProcessingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
