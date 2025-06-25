#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
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

        # Simulated breath peak flag
        self.breath_peak = False  # You can replace with actual detection

        # Subscribers
        rospy.Subscriber("/boson/image_raw", Image, self.raw_callback)
        rospy.Subscriber("/boson/image_roi", Image, self.roi_callback)
        rospy.Subscriber("/boson/image_annotated", Image, self.annotated_callback)

        # Start periodic display update (10 Hz)
        rospy.Timer(rospy.Duration(0.1), self.display_timer_callback)
        
    def mono16_to_rgb(self, image_raw):
        # Normalize and convert to RGB
        min_val, max_val = np.min(image_raw), np.max(image_raw)
        # Avoid division by zero
        if max_val == min_val:
            rospy.logwarn("Image has no variation, using default color.")
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
        except Exception as e:
            rospy.logerr(f"[annotated_callback] {e}")

    def roi_callback(self, msg):
        try:
            image_roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            image_roi = self.mono16_to_rgb(image_roi)
            self.latest_roi = self.safe_resize(image_roi)
        except Exception as e:
            rospy.logerr(f"[roi_callback] {e}")

    def safe_resize(self, img):
        try:
            return cv2.resize(img, (self.disp_w, self.disp_h))
        except Exception as e:
            rospy.logerr(f"[safe_resize] Failed to resize image: {e}")
            return np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

    def display_timer_callback(self, event):
        try:
            # Flash effect on ROI
            flash_display = self.latest_roi.copy()
            if self.breath_peak:
                flash_display[:] = [0, 255, 0]

            # 2x2 grid
            top_row = np.hstack((self.latest_raw, self.latest_annotated))
            bottom_row = np.hstack((self.latest_roi, flash_display))
            combined = np.vstack((top_row, bottom_row))

            # Show in OpenCV window
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
