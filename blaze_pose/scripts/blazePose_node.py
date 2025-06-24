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
        self.image_pub = rospy.Publisher("/boson/breath_centroid", Image, queue_size=1)
        self.debug_pub = rospy.Publisher("/boson/annotated", Image, queue_size=1)

    def callback(self, msg):
        try:
            # Convert mono16 to OpenCV
            mono16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")

            # Normalize and convert to RGB
            min_val, max_val = np.min(mono16), np.max(mono16)
            mono8 = ((mono16 - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            rgb_frame = cv2.cvtColor(mono8, cv2.COLOR_GRAY2RGB)

            # Run BlazePose
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get keypoints (nose, mouth_left, mouth_right)
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                mouth_left = landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT]
                mouth_right = landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT]

                h, w, _ = rgb_frame.shape
                pts_x = [int(nose.x * w), int(mouth_left.x * w), int(mouth_right.x * w)]
                pts_y = [int(nose.y * h), int(mouth_left.y * h), int(mouth_right.y * h)]

                x_min = max(min(pts_x) - 10, 0)
                x_max = min(max(pts_x) + 10, w)
                y_min = max(min(pts_y) - 10, 0)
                y_max = min(max(pts_y) + 10, h)

                # Draw box
                cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Extract ROI
                roi = rgb_frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    resized_roi = cv2.resize(roi, (128, 128))
                    out_msg = self.bridge.cv2_to_imgmsg(resized_roi, encoding="rgb8")
                    self.image_pub.publish(out_msg)

            # Show debug window
            cv2.imshow("BlazePose Annotated", rgb_frame)
            cv2.waitKey(1)

            debug_msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding='rgb8')
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"BlazePose node error: {e}")

if __name__ == "__main__":
    BlazePoseNode()
    rospy.spin()
