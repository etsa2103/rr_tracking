#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from cv_bridge import CvBridge
import cv2

import time
from datetime import datetime
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

        # Timestamp of latest frames
        self.raw_timestamp = None
        self.annotated_timestamp = None
        self.roi_timestamp = None
        
        # Simulated breath peak flag (optional visual flash)
        self.breath_peak = False

        # Latest matched BPM to show with annotated frame
        self.latest_rr_inst = None
        self.latest_rr_avg = None
        self.bmp_timestamp = None

        # Subscribers
        rospy.Subscriber("/boson/image_raw", Image, self.raw_callback)
        rospy.Subscriber("/boson/image_roi", Image, self.roi_callback)
        rospy.Subscriber("/boson/image_annotated", Image, self.annotated_callback)
        rospy.Subscriber("/rr_tracking/rr_inst", Float32, self.rr_inst_callback)
        rospy.Subscriber("/rr_tracking/rr_avg", Float32, self.rr_avg_callback)

        # Display update timer (10 Hz)
        rospy.Timer(rospy.Duration(0.1), self.display_timer_callback)

    def raw_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            # Normalize and convert to RGB
            min_val, max_val = np.min(image_raw), np.max(image_raw)
            image_mono8 = ((image_raw - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_mono8, cv2.COLOR_GRAY2RGB)
            # Resize to display size
            self.latest_raw = self.safe_resize(image_rgb)
            # Get timestamp from raw image
            self.raw_timestamp = datetime.fromtimestamp(msg.header.stamp.to_sec()).strftime("%H:%M:%S")
            
        except Exception as e:
            rospy.logerr(f"[raw_callback] {e}")

    def annotated_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # Resize to display size
            self.latest_annotated = self.safe_resize(image)
            # Get timestamp from annotated image
            self.annotated_timestamp = datetime.fromtimestamp(msg.header.stamp.to_sec()).strftime("%H:%M:%S")

        except Exception as e:
            rospy.logerr(f"[annotated_callback] {e}")

    def roi_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            image_roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            # Normalize and convert to RGB to display
            min_val, max_val = np.min(image_roi), np.max(image_roi)
            image_mono8 = ((image_roi - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            image_roi = cv2.cvtColor(image_mono8, cv2.COLOR_GRAY2RGB)
            # Resize to display size
            self.latest_roi = self.safe_resize(image_roi)
            # Get timestamp from roi image
            self.roi_timestamp = datetime.fromtimestamp(msg.header.stamp.to_sec()).strftime("%H:%M:%S")
        except Exception as e:
            rospy.logerr(f"[roi_callback] {e}")

    def rr_inst_callback(self, msg):
        try:
            self.latest_rr_inst = msg.data
            #self.rr_inst_timestamp = datetime.fromtimestamp(msg.header.stamp.to_sec()).strftime("%H:%M:%S")
        except Exception as e:
            rospy.logerr(f"[rr_inst_callback] {e}")
            
    def rr_avg_callback(self, msg):
        try:
            self.latest_rr_avg = msg.data
            #self.rr_avg_timestamp = datetime.fromtimestamp(msg.header.stamp.to_sec()).strftime("%H:%M:%S")
        except Exception as e:
            rospy.logerr(f"[rr_avg_callback] {e}")

    def safe_resize(self, img):
        try:
            return cv2.resize(img, (self.disp_w, self.disp_h))
        except Exception as e:
            rospy.logerr(f"[safe_resize] Failed to resize image: {e}")
            return np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

    def display_timer_callback(self, event):
        try:
            # Flash display if breath peak detected
            flash_display = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
            # flash_display = self.latest_roi.copy()
            # if self.breath_peak:
            #     flash_display[:] = [0, 255, 0]

            # Combine all images into a single display
            top_row = np.hstack((self.latest_raw, self.latest_annotated))
            bottom_row = np.hstack((self.latest_roi, flash_display))
            combined = np.vstack((top_row, bottom_row))
                
            # Show timestamps if available
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.raw_timestamp is not None:
                timestamp_str = self.raw_timestamp
                cv2.putText(combined, timestamp_str, (20, 40), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            if self.annotated_timestamp is not None:
                timestamp_str = self.annotated_timestamp
                cv2.putText(combined, timestamp_str, (670, 40), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            if self.roi_timestamp is not None:
                timestamp_str = self.roi_timestamp
                cv2.putText(combined, timestamp_str, (20, 520), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                
            # Add instantanious respiration rate text if available
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 255, 0)
            thickness = 2
            if self.latest_rr_inst is not None:
                text = f"Resperation Rate: {self.latest_rr_inst:.1f} BPM"
                cv2.putText(combined, text, (790, 540), font, font_scale, color, thickness, cv2.LINE_AA)
            # Add average respiration rate text if available
            if self.latest_rr_avg is not None:
                text = f"Average Resperation Rate: {self.latest_rr_avg:.1f} BPM"
                cv2.putText(combined, text, (670, 570), font, font_scale, color, thickness, cv2.LINE_AA)

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
