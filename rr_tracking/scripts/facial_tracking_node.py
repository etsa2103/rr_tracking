#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import mediapipe as mp

def valid_landmark(lm, threshold=0.5):
    return lm.visibility > threshold if hasattr(lm, 'visibility') else lm.visibility == 0 or lm.presence > threshold

def smooth_box(prev_box, new_box, alpha=0.3):
    if prev_box is None:
        return new_box
    return tuple([int(prev_box[i] * (1 - alpha) + new_box[i] * alpha) for i in range(4)])  

class BlazePoseFaceMeshSwitcher:
    def __init__(self):
        rospy.init_node("blazepose_facemesh_node")
        self.bridge = CvBridge()

        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
        )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.draw_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.draw_utils.DrawingSpec(thickness=1, circle_radius=1)

        rospy.Subscriber("/boson/image_raw", Image, self.callback)
        self.image_annotated_pub = rospy.Publisher("/boson/image_annotated", Image, queue_size=1)
        self.image_roi_pub = rospy.Publisher("/boson/image_roi", Image, queue_size=1)
        self.tracking_stable_pub = rospy.Publisher("/rr_tracking/tracking_stable", Bool, queue_size=1)

        self.last_mroi_box = None  # Smoothed box
        
    def estimate_stability(self, new_box, now):
        """Determine if the box is stable based on motion speed."""
        if self.last_mroi_box is None or self.last_mroi_time is None:
            return True

        prev_box = self.last_mroi_box
        prev_time = self.last_mroi_time
        dt = now - prev_time

        # Calculate box center
        cx1 = (prev_box[0] + prev_box[2]) / 2
        cy1 = (prev_box[1] + prev_box[3]) / 2
        cx2 = (new_box[0] + new_box[2]) / 2
        cy2 = (new_box[1] + new_box[3]) / 2

        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        speed = dist / dt if dt > 0 else 0

        # Tweak this threshold experimentally (~20–40 pixels/sec reasonable)
        return speed < 120
    
    def determine_pose_type(self, landmarks):
        lm = mp.solutions.pose.PoseLandmark
        try:
            nose = landmarks[lm.NOSE]
            left_eye = landmarks[lm.LEFT_EYE]
            right_eye = landmarks[lm.RIGHT_EYE]
        except IndexError:
            return 'unknown'

        if not (valid_landmark(nose) and valid_landmark(left_eye) and valid_landmark(right_eye)):
            return 'unknown'

        nose_pt = np.array([nose.x, nose.y])
        left_pt = np.array([left_eye.x, left_eye.y])
        right_pt = np.array([right_eye.x, right_eye.y])

        eye_vec = right_pt - left_pt
        if np.linalg.norm(eye_vec) < 1e-5:
            return 'unknown'

        eye_dir = eye_vec / np.linalg.norm(eye_vec)
        nose_vec = nose_pt - left_pt
        proj = np.dot(nose_vec, eye_dir)
        rel_pos = proj / np.linalg.norm(eye_vec)

        if 0.25 <= rel_pos <= 0.75:
            return 'frontal'
        elif rel_pos < 0.25:
            return 'right'
        elif rel_pos > 0.75:
            return 'left'
        return 'unknown'

    def get_box_from_blazepose(self, landmarks, pose, w, h):
        lm = mp.solutions.pose.PoseLandmark

        if pose == 'left':
            eye = landmarks[lm.LEFT_EYE]
            mouth = landmarks[lm.MOUTH_LEFT]
            ear = landmarks[lm.LEFT_EAR]
            direction = -1  # flip normal
        elif pose == 'right':
            eye = landmarks[lm.RIGHT_EYE]
            mouth = landmarks[lm.MOUTH_RIGHT]
            ear = landmarks[lm.RIGHT_EAR]
            direction = 1
        else:
            return None

        if not (valid_landmark(eye) and valid_landmark(mouth)):
            return None

        # Convert to image coordinates
        eye_pt = np.array([eye.x * w, eye.y * h])
        mouth_pt = np.array([mouth.x * w, mouth.y * h])
        mid_pt = (eye_pt + mouth_pt) / 2
        # find point 40 percent of the way from mouth to eye
        mid_pt = (mid_pt + mouth_pt * 0.4) / 1.4
        

        # Vector from mouth to eye
        vec = eye_pt - mouth_pt
        norm_vec = vec / (np.linalg.norm(vec) + 1e-6)

        # Rotate 90° to get perpendicular direction (CCW), flip for left view
        perp_vec = direction * np.array([-norm_vec[1], norm_vec[0]])

        # Use ear distance to set scale
        if valid_landmark(ear):
            ear_pt = np.array([ear.x * w, ear.y * h])
            ear_dist = np.linalg.norm(ear_pt - mouth_pt)
        else:
            ear_dist = np.linalg.norm(vec)

        # Estimate nose position
        offset_dist = 0.3 * ear_dist
        nose_pt = mid_pt + perp_vec * offset_dist

        # Define box around nose
        box_w = int(ear_dist * 0.5)
        box_h = int(ear_dist * 0.5)

        x_min = max(int(nose_pt[0] - box_w // 2), 0)
        y_min = max(int(nose_pt[1] - box_h // 2), 0)
        x_max = min(int(nose_pt[0] + box_w // 2), w)
        y_max = min(int(nose_pt[1] + box_h // 2), h)

        return (x_min, y_min, x_max, y_max)



    def get_box_from_facemesh(self, landmarks, w, h):
        idx_nose_tip = 1
        idx_left_cheek = 234
        idx_right_cheek = 454
        idx_chin = 152

        try:
            nose = landmarks[idx_nose_tip]
            left = landmarks[idx_left_cheek]
            right = landmarks[idx_right_cheek]
            chin = landmarks[idx_chin]
        except IndexError:
            return None

        cx = int(nose.x * w)
        cy = int(nose.y * h)
        fw = int(abs(right.x - left.x) * w * 0.25)
        fh = int(abs(chin.y - nose.y) * h * 0.4)

        return (
            max(cx - fw // 2, 0),
            max(cy - fh // 4, 0),
            min(cx + fw // 2, w),
            min(cy + fh * 3 // 4, h)
        )

    def callback(self, msg):
        try:
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            min_val, max_val = np.min(image_raw), np.max(image_raw)
            norm = ((image_raw - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

            h, w, _ = image_rgb.shape
            now = rospy.get_time()

            pose_result = self.pose_detector.process(image_rgb)
            if not pose_result.pose_landmarks:
                return

            pose_landmarks = pose_result.pose_landmarks.landmark
            pose_type = self.determine_pose_type(pose_landmarks)

            mroi_box = None
            annotated_image = image_rgb.copy()

            if pose_type in ['left', 'right']:
                # Use BlazePose for anchor
                self.draw_utils.draw_landmarks(
                    annotated_image,
                    pose_result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
                mroi_box = self.get_box_from_blazepose(pose_landmarks, pose_type, w, h)

            elif pose_type == 'frontal':
                face_result = self.face_mesh.process(image_rgb)
                if face_result.multi_face_landmarks:
                    landmarks = face_result.multi_face_landmarks[0]
                    self.draw_utils.draw_landmarks(
                        annotated_image,
                        landmarks,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        self.drawing_spec,
                        self.drawing_spec
                    )
                    mroi_box = self.get_box_from_facemesh(landmarks.landmark, w, h)

            # ====== Stability Check ======
            if mroi_box is not None:
                stable = self.estimate_stability(mroi_box, now)
                self.tracking_stable_pub.publish(Bool(stable))

                self.last_mroi_box = smooth_box(self.last_mroi_box, mroi_box)
                self.last_mroi_time = now
            else:
                self.tracking_stable_pub.publish(Bool(False))

            if self.last_mroi_box is not None:
                x_min, y_min, x_max, y_max = self.last_mroi_box
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                roi = image_raw[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_msg = self.bridge.cv2_to_imgmsg(roi, encoding='mono16')
                    roi_msg.header = msg.header
                    self.image_roi_pub.publish(roi_msg)

            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            annotated_msg.header = msg.header
            self.image_annotated_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"[PoseSwitcherNode] Error: {e}")

if __name__ == "__main__":
    BlazePoseFaceMeshSwitcher()
    rospy.spin()
