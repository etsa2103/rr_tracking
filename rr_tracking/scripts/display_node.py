#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import time
from collections import deque
import numpy as np

class DisplayNode:
    def __init__(self):
        rospy.init_node("display_node")
        self.bridge = CvBridge()

        # Display size
        self.disp_w, self.disp_h = 640, 480

        # Initialize black placeholders
        black = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
        self.latest_raw = black.copy()
        self.latest_annotated = black.copy()
        self.latest_roi = black.copy()

        # Simulated breath peak flag (optional visual flash)
        self.breath_peak = False

        # Latest matched BPM to show with annotated frame
        self.latest_bpm = None

        # Initialize BPM buffer: (timestamp, bpm)
        self.bpm_buffer = deque(maxlen=100)

        # Subscribers
        rospy.Subscriber("/boson/image_raw", Image, self.raw_callback)
        rospy.Subscriber("/boson/image_roi", Image, self.roi_callback)
        rospy.Subscriber("/boson/image_annotated", Image, self.annotated_callback)
        rospy.Subscriber("/rr_tracking/breath_rate", Float32, self.breath_rate_callback)

        # Display update timer (10 Hz)
        rospy.Timer(rospy.Duration(0.1), self.display_timer_callback)

    def mono16_to_rgb(self, image_raw):
        min_val, max_val = np.min(image_raw), np.max(image_raw)
        if max_val == min_val:
            rospy.logwarn("Image has no variation, using default color.")
            image_mono8 = np.zeros_like(image_raw, dtype=np.uint8)
            image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
        else:
            image_mono8 = ((image_raw - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_mono8, cv2.COLOR_GRAY2RGB)
        return image_rgb

    def raw_callback(self, msg):
        try:
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            image_rgb = self.mono16_to_rgb(image_raw)
            self.latest_raw = self.safe_resize(image_rgb)
        except Exception as e:
            rospy.logerr(f"[raw_callback] {e}")

    def annotated_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.latest_annotated = self.safe_resize(image)

            # Get timestamp from annotated image
            annotated_stamp = msg.header.stamp.to_sec()

            # Match closest BPM to this timestamp
            matched_bpm = None
            min_bpm_diff = float("inf")
            for bpm_stamp, bpm_val in self.bpm_buffer:
                diff = abs(bpm_stamp - annotated_stamp)
                if diff < min_bpm_diff:
                    matched_bpm = bpm_val
                    min_bpm_diff = diff

            self.latest_bpm = matched_bpm
        except Exception as e:
            rospy.logerr(f"[annotated_callback] {e}")

    def roi_callback(self, msg):
        try:
            image_roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            image_roi = self.mono16_to_rgb(image_roi)
            self.latest_roi = self.safe_resize(image_roi)
        except Exception as e:
            rospy.logerr(f"[roi_callback] {e}")

    def breath_rate_callback(self, msg):
        try:
            stamp = time.time()
            self.bpm_buffer.append((stamp, msg.data))
        except Exception as e:
            rospy.logerr(f"[bpm_callback] {e}")

    def safe_resize(self, img):
        try:
            return cv2.resize(img, (self.disp_w, self.disp_h))
        except Exception as e:
            rospy.logerr(f"[safe_resize] Failed to resize image: {e}")
            return np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

    def display_timer_callback(self, event):
        try:
            flash_display = self.latest_roi.copy()
            if self.breath_peak:
                flash_display[:] = [0, 255, 0]

            top_row = np.hstack((self.latest_raw, self.latest_annotated))
            bottom_row = np.hstack((self.latest_roi, flash_display))
            combined = np.vstack((top_row, bottom_row))

            if self.latest_bpm is not None:
                text = f"BPM: {self.latest_bpm:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                color = (0, 255, 0)
                cv2.putText(combined, text, (670, 50), font, font_scale, color, thickness, cv2.LINE_AA)

            cv2.imshow("Thermal Debug Views", combined)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"[display_timer_callback] {e}")

if __name__ == "__main__":
    try:
        DisplayNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
