#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp

class BlazePoseNode:
    def __init__(self):
        rospy.init_node("blazepose_node")
        self.bridge = CvBridge()

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

        self.image_sub = rospy.Subscriber("/boson/image_raw", Image, self.callback)
        self.image_annotated_pub = rospy.Publisher("/boson/image_annotated", Image, queue_size=1,latch=True)
        self.image_roi_pub = rospy.Publisher("/boson/image_roi", Image, queue_size=1,latch=True)

    def callback(self, msg):
        try:
            # Convert mono16 to OpenCV
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")

            # Normalize and convert to RGB
            min_val, max_val = np.min(image_raw), np.max(image_raw)
            # Avoid division by zero
            if max_val == min_val:
                rospy.logwarn("Image has no variation, using default color.")
                image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
            else:
                image_mono8 = ((image_raw - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                image_rgb = cv2.cvtColor(image_mono8, cv2.COLOR_GRAY2RGB)

            # Run BlazePose
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get nose, mouth_left, mouth_right landmarks
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                mouth_left = landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT]
                mouth_right = landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT]

                h, w, _ = image_rgb.shape

                # Convert normalized coords to pixel values
                pts_x = [int(nose.x * w), int(mouth_left.x * w), int(mouth_right.x * w)]
                pts_y = [int(nose.y * h), int(mouth_left.y * h), int(mouth_right.y * h)]

                # Add margin around the features (optional)
                margin = 10  # pixels

                # Compute bounding box with margin
                x_min = max(min(pts_x) - margin, 0)
                x_max = min(max(pts_x) + margin, w)
                y_min = max(min(pts_y) - margin, 0)
                y_max = min(max(pts_y) + margin, h)
                
                y_min = y_min - int(0.4 * (y_max - y_min))  # Adjust y_min to include more of the nose

                # Create a bounding box around the nose and mouth region
                cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Publish the annotated image (optional)
                annotated_image_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
                self.image_annotated_pub.publish(annotated_image_msg)
                
                # Extract ROI (Region of Interest)
                image_roi = image_raw[y_min:y_max, x_min:x_max]
                # publish the ROI
                image_roi_msg = self.bridge.cv2_to_imgmsg(image_roi, encoding='mono16')
                self.image_roi_pub.publish(image_roi_msg)
            else:
                rospy.logwarn("No pose landmarks detected.")
        except Exception as e:
            rospy.logerr(f"BlazePose node error: {e}")

if __name__ == "__main__":
    BlazePoseNode()
    rospy.spin()