#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

import mediapipe as mp

class BlazePoseNode:
    def __init__(self):
        rospy.init_node("blazepose_node")
        self.bridge = CvBridge()

        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

        rospy.Subscriber("/boson/image_raw", Image, self.callback)
        self.image_annotated_pub = rospy.Publisher("/boson/image_annotated", Image, queue_size=1)
        self.image_roi_pub = rospy.Publisher("/boson/image_roi", Image, queue_size=1)

    def callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            # Normalize and convert to RGB
            min_val, max_val = np.min(image_raw), np.max(image_raw)
            if max_val == min_val:
                image_mono8 = np.zeros_like(image_raw, dtype=np.uint8)
            else:
                image_mono8 = ((image_raw - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            image_rgb = cv2.cvtColor(image_mono8, cv2.COLOR_GRAY2RGB)
            # Run BlazePose detection
            results = self.pose.process(image_rgb)
            if not results.pose_landmarks:
                return
            
            h, w, _ = image_rgb.shape
            lm = mp.solutions.pose.PoseLandmark
            landmarks = results.pose_landmarks.landmark

            # Get landmarks for nose, mouth corners, and eyes
            pts = [landmarks[lm.NOSE], landmarks[lm.MOUTH_LEFT], landmarks[lm.MOUTH_RIGHT],
                   landmarks[lm.LEFT_EYE], landmarks[lm.RIGHT_EYE]]
            # Convert normalized coordinates to pixel values
            xs = [int(p.x * w) for p in pts]
            ys = [int(p.y * h) for p in pts]

            # TROI: Bounding box around facial features
            x_min = max(min(xs), 0)
            x_max = min(max(xs), w)
            y_min = max(min(ys), 0)
            y_max = min(max(ys), h)
            troi_height = y_max - y_min
            troi_width = x_max - x_min
            # MROI: Region below nose (breathing area)
            centerX = np.mean(xs)
            centerY = np.mean(ys)
            mroi_y_min = int(centerY - 0.20*troi_height)
            mroi_y_max = int(centerY + 0.10*troi_height)
            mroi_x_min = int(centerX - 0.40*troi_width)
            mroi_x_max = int(centerX + 0.10*troi_width)
            # Ensure valid non-zero region
            if mroi_y_max <= mroi_y_min or mroi_x_max <= mroi_x_min:
                return

            # Draw boxes for annotation
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)  # TROI
            cv2.rectangle(image_rgb, (mroi_x_min, mroi_y_min), (mroi_x_max, mroi_y_max), (255, 0, 0), 1)  # MROI

            # Publish annotated RGB image
            annotated_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
            annotated_msg.header = msg.header

            self.image_annotated_pub.publish(annotated_msg)

            # Extract and publish mono16 MROI for breathing signal
            image_roi = image_raw[mroi_y_min:mroi_y_max, mroi_x_min:mroi_x_max]
            roi_msg = self.bridge.cv2_to_imgmsg(image_roi, encoding='mono16')
            roi_msg.header = msg.header
            self.image_roi_pub.publish(roi_msg)

        except Exception as e:
            rospy.logerr(f"[BlazePoseNode] Error: {e}")

if __name__ == "__main__":
    BlazePoseNode()
    rospy.spin()