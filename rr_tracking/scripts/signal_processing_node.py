#!/usr/bin/env python3
import rospy
import time
# import threading
import numpy as np
from cv_bridge import CvBridge
from collections import deque, Counter
from std_msgs.msg import Float32, UInt8, Float64MultiArray
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

from rr_tracking.msg import TrackingState
from rr_tracking.msg import ProcessingState, StableRange

class SignalProcessingNode:
    def __init__(self):
        rospy.init_node("simple_rr_tracker")
        self.bridge = CvBridge()

        # ROS I/O
        rospy.Subscriber('/jackal_teleop/trigger', UInt8, self.trigger_cb)
        rospy.Subscriber("/facial_tracking/trackingState", TrackingState, self.tracking_state_cb)
        self.processing_state_pub = rospy.Publisher("/signal_processing/processingState", ProcessingState, queue_size=10)

        # Variables
        self.frame_rate = 30
        self.min_peak_distance = 0.8
        
        self.recording = False
        self.record_start_time = None
        
        self.image_roi = None
        self.tracking_stable = True
        self.pose_type = "unknown"
        
        self.start_time = time.time()

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
            # self.recording_pub.publish(Bool(True))
            self.record_start_time = rospy.get_time()
            recording_time = rospy.get_param("/recording_time", 20)
            rospy.loginfo(f"Started recording for {recording_time} seconds")
            
            # maybe reset all the buffers for a new recording (or reset all except signal buffer)
            # self.signal_buffer.clear()
            # self.stability_buffer.clear()
            # self.time_buffer = deque(maxlen=self.frame_rate * 45)
            # self.peak_buffer = deque(maxlen=self.frame_rate * 45)
            
    def tracking_state_cb(self, msg):
        self.image_roi = self.bridge.imgmsg_to_cv2(msg.image_roi, desired_encoding="mono16")
        self.process_signal()
        recording_time = rospy.get_param("/recording_time", 20)
        if self.recording and (rospy.get_time() - self.record_start_time >= recording_time): #maybe add a way to stop before 20 seconds
                self.recording = False
                rospy.loginfo("Finished recording. Processing...")
                # self.rr_final_pub.publish(Float32(rr_avg))
        self.tracking_stable = msg.tracking_stable.data
        self.pose_type = msg.pose_type.data
        
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
            rospy.logwarn_throttle(1.0, "[signal_processing] Empty ROI image.")
            return
        # Mask background pixels
        use_mask = rospy.get_param("/rr_tracking/use_mask", False)
        warm_pixels = self.extract_warm_pixels(self.image_roi, use_mask)
        # Check edge case
        if not self.tracking_stable or warm_pixels.size == 0 or warm_pixels.mean() < 0.3:
            rospy.logwarn_throttle(1.0, "[signal_processing] Unstable or invalid ROI — skipping.")
            return
        # Update signal buffers
        mean_val = np.mean(warm_pixels)
        now = rospy.get_time()
        self.signal_buffer.append(mean_val)
        # Check edge case
        if len(self.signal_buffer) < self.signal_buffer.maxlen:
            rospy.logwarn_throttle(5.0, "[signal_processing] Buffering live signal...")
            return
        self.stability_buffer.append(self.tracking_stable)
        self.pose_buffer.append(self.pose_type)
        self.time_buffer.append(now-self.start_time)
        
        # Process signal and update peak buffer
        signal = np.array(self.signal_buffer)
        baseline = uniform_filter1d(signal, size=self.frame_rate * 4)
        normalized = signal - baseline
        inverted = np.clip(-normalized, -100, 100)
        self.peak_buffer.append(inverted[-1])
        # self.raw_pub.publish(inverted[-1])
        
        # analysis
        segments = []
        start_idx = None
        peak_signal = np.array(self.peak_buffer)
        stabilities = np.array(self.stability_buffer)
        poses = np.array(self.pose_buffer)
        timestamps = np.array(self.time_buffer)
        
        # Get indexes of stable signal segments
        # peak_timestamps = timestamps[-len(peak_signal):]
        for i, stable in enumerate(stabilities):
            if stable and start_idx is None:
                start_idx = i
            elif not stable and start_idx is not None:
                if timestamps[i - 1] - timestamps[start_idx] >= 5.0:
                    segments.append((start_idx, i))
                start_idx = None
        if start_idx is not None and timestamps[-1] - timestamps[start_idx] >= 5.0:
            segments.append((start_idx, len(stabilities)))
            
        rr_values = []
        all_peak_times = []
        bounds = []
        # Check edge case
        if segments:
            # For each stable segment get the time of all peaks and calculate rr
            
            for start, end in segments:
                # Update messages for gui
                bounds.extend([float(timestamps[start]), float(timestamps[end - 1])]) 
                
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
                    all_peak_times.append(float(seg_timestamps[peak]-10))
                    
                if(len(peak_times) >= 2):
                    for i in range(len(peak_times)-1):
                        rr_values.append(60.0/(peak_times[i+1] - peak_times[i]))
                
        else:
            rospy.logwarn_throttle(1.0,"[signal_processing] No valid stable segments found.")
                    
        # Publish processing state
        processing_state_msg = ProcessingState()
        processing_state_msg.graph_data = Float64MultiArray(data=peak_signal.tolist())
        processing_state_msg.graph_times = Float64MultiArray(data=timestamps.tolist())
        if(rr_values):
            processing_state_msg.rr_inst = Float32(data=rr_values[-1])
            processing_state_msg.rr_avg = Float32(data=np.mean(rr_values))
        else:
            processing_state_msg.rr_inst = Float32(data=0.0)
            processing_state_msg.rr_avg = Float32(data=0.0)
            rospy.logwarn_throttle(1.0,"[signal_processing] Not enough peaks across all segments.")
        if bounds:
            processing_state_msg.stable_ranges = [
                StableRange(start_time=bounds[i], end_time=bounds[i+1]) 
                for i in range(0, len(bounds), 2)]
        if all_peak_times:
            all_peak_times = [t+10 for t in all_peak_times]
            processing_state_msg.peak_times = Float64MultiArray(data=all_peak_times)
        self.processing_state_pub.publish(processing_state_msg)

# =========================================================================================================
# =========================================== Utility Functions ===========================================
# =========================================================================================================
def bandpass(data, fs, low=0.1, high=0.7):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

# === Main Entry Point ===
if __name__ == "__main__":
    try:
        SignalProcessingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
