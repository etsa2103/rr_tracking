#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from rr_tracker.msg import TrackingState

# === Main Tracker Class ===
class FacialTrackingNode:
    def __init__(self):
        # Settings
        self.use_clahe = True
        self.use_face_mesh = False
        self.old_bagged_data = rospy.get_param("/old_bagged_data", False)
        
        # ROS and Model Setup
        rospy.init_node("facial_tracking_node")
        self.bridge = CvBridge()
        self.blaze_pose = mp.solutions.pose.Pose(
            static_image_mode=False, 
            model_complexity=1, 
            min_detection_confidence=0.5)
        if(self.use_face_mesh):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, 
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5)
        self.draw_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.draw_utils.DrawingSpec(thickness=1, circle_radius=1)

        # ROS Publishers/Subscribers
        if(self.old_bagged_data):
            rospy.Subscriber("/boson/image_raw", Image, self.image_raw_cb)
        else:
            rospy.Subscriber("/boson640/image_raw", Image, self.image_raw_cb)
        self.tracking_state_pub = rospy.Publisher("/facial_tracking/trackingState", TrackingState, queue_size=10)
        
        # MROI vars
        self.last_mroi_box = None
        self.last_mroi_time = None
        # Other vars
        self.temp_c = None
    
    # ================================================================================================================================      
    # ====================================================== Callback Functions ======================================================
    # ================================================================================================================================
    
    # ==== Image Callback =====
    def image_raw_cb(self, msg):
        try:
            # === Add images to state message ===
            tracking_state_msg = TrackingState()
            tracking_state_msg.image_annotated = self.bridge.cv2_to_imgmsg(np.zeros((640, 512, 3), dtype=np.uint8), encoding="rgb8")
            tracking_state_msg.image_roi = self.bridge.cv2_to_imgmsg(np.zeros((640, 512), dtype=np.uint16), encoding="mono16")
            
            # === Process the incoming image ===
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
            
            # Convert raw image to temperature in Celsius
            self.temp_c = image_raw * 0.009 - 238
            self.temp_c = np.clip(self.temp_c, -30, 140)
            #rospy.logwarn(f"[rr_tracker/facial_tracking] Temperature: {np.mean(self.temp_c):.2f} C")
            
            norm = ((image_raw - np.min(image_raw)) / (np.ptp(image_raw) + 1e-5) * 255).astype(np.uint8)
            if(self.use_clahe):
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
                clahe_applied = clahe.apply(norm)
                # Convert to RGB for BlazePose input
                image_rgb = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
            # Get image dimensions and current time
            h, w, _ = image_rgb.shape
            now = rospy.get_time()
            
            # === Pose Detection ===
            pose_result = self.blaze_pose.process(image_rgb)
            if not pose_result.pose_landmarks:
                rospy.logdebug(f"[rr_tracker/facial_tracking] No pose detected by BlazePose")
                return
            pose_landmarks = pose_result.pose_landmarks.landmark
            pose_type = self.determine_pose_type(pose_landmarks)
            # Update State message
            tracking_state_msg.pose_type = String(data = pose_type)

            # === Landmark Extraction Based on Pose Type ===
            annotated_image = image_rgb.copy()

            mroi_box = None
            if pose_type in ['left', 'right']:
                self.draw_utils.draw_landmarks(
                    annotated_image, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                mroi_box = self.get_box_from_blazepose(pose_landmarks, pose_type, w, h)
            elif pose_type == 'frontal':
                if(self.use_face_mesh):
                    face_result = self.face_mesh.process(image_rgb)
                    if face_result.multi_face_landmarks:
                        landmarks = face_result.multi_face_landmarks[0]
                        self.draw_utils.draw_landmarks(
                            annotated_image, landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            self.drawing_spec, self.drawing_spec)
                        mroi_box = self.get_box_from_facemesh(landmarks.landmark, w, h)
                else:
                    self.draw_utils.draw_landmarks(annotated_image, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    mroi_box = self.get_box_from_blazepose_frontal(pose_landmarks, w, h)

            # === Stability Check (Publish results) ===
            if mroi_box is not None:
                stable = self.estimate_stability(mroi_box, now)
                tracking_state_msg.tracking_stable = Bool(data=stable)
                self.last_mroi_box = smooth_box(self.last_mroi_box, mroi_box)
                self.last_mroi_time = now
            else:
                tracking_state_msg.tracking_stable = Bool(data=False)

            # === ROI Cropping (Publish results) ===
            if self.last_mroi_box:
                x_min, y_min, x_max, y_max = self.last_mroi_box
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                roi = image_raw[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_msg = self.bridge.cv2_to_imgmsg(roi, encoding='mono16')
                    roi_msg.header = msg.header
                    tracking_state_msg.image_roi = roi_msg
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='rgb8')
            annotated_msg.header = msg.header
            # Update State
            tracking_state_msg.image_annotated = annotated_msg
            self.tracking_state_pub.publish(tracking_state_msg)

        except Exception as e:
            rospy.logerr(f"[rr_tracker/facial_tracking] Error: {e}")
      
    # ================================================================================================================================      
    # ======================================================= Helper Functions =======================================================
    # ================================================================================================================================
     
    # ========= Stability Estimation =========
    def estimate_stability(self, new_box, now):
        if self.last_mroi_box is None or self.last_mroi_time is None:
            return True
        cx1, cy1 = (self.last_mroi_box[0] + self.last_mroi_box[2]) / 2, (self.last_mroi_box[1] + self.last_mroi_box[3]) / 2
        cx2, cy2 = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
        speed = np.linalg.norm([cx2 - cx1, cy2 - cy1]) / (now - self.last_mroi_time + 1e-5)
        return speed < 200

    # ====== Pose Type Classification ======
    def determine_pose_type(self, landmarks):
        lm = mp.solutions.pose.PoseLandmark
        try:
            nose, left_eye, right_eye = landmarks[lm.NOSE], landmarks[lm.LEFT_EYE], landmarks[lm.RIGHT_EYE]
        except IndexError:
            return 'unknown'
        if not all(valid_landmark(l) for l in [nose, left_eye, right_eye]):
            return 'unknown'

        nose_pt, left_pt, right_pt = np.array([nose.x, nose.y]), np.array([left_eye.x, left_eye.y]), np.array([right_eye.x, right_eye.y])
        eye_vec = right_pt - left_pt
        if np.linalg.norm(eye_vec) < 1e-5:
            return 'unknown'
        eye_dir = eye_vec / np.linalg.norm(eye_vec)
        proj = np.dot(nose_pt - left_pt, eye_dir)
        rel_pos = proj / np.linalg.norm(eye_vec)
        return 'frontal' if 0.25 <= rel_pos <= 0.75 else 'right' if rel_pos < 0.25 else 'left'

    # ====== Box Estimation for Side View (BlazePose) ======
    def get_box_from_blazepose(self, landmarks, pose, w, h):
        lm = mp.solutions.pose.PoseLandmark
        if pose == 'left':
            eye, mouth, ear, direction = landmarks[lm.LEFT_EYE], landmarks[lm.MOUTH_LEFT], landmarks[lm.LEFT_EAR], -1
        elif pose == 'right':
            eye, mouth, ear, direction = landmarks[lm.RIGHT_EYE], landmarks[lm.MOUTH_RIGHT], landmarks[lm.RIGHT_EAR], 1
        else:
            return None
        if not (valid_landmark(eye) and valid_landmark(mouth)):
            rospy.logdebug(f"[rr_tracker/facial_tracking] BlazePose does not have sufficient landmarks for nose tracking")
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
        perp_vec = direction * np.array([-norm_vec[1], norm_vec[0]]) * 1.1

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

    # === Box Estimation for Frontal View (Blazepose) ===
    def get_box_from_blazepose_frontal(self, landmarks, w, h):
        lm = mp.solutions.pose.PoseLandmark
        try:
            nose      = landmarks[lm.NOSE]
            left_face = landmarks[lm.LEFT_EAR]
            right_face= landmarks[lm.RIGHT_EAR]
            chin      = landmarks[lm.MOUTH_LEFT]  # BlazePose has no true chin; approx with mouth
        except IndexError:
            rospy.logdebug(f"[rr_tracker/facial_tracking] Missing key BlazePose landmarks for frontal box.")
            return None

        if not all(valid_landmark(l) for l in [nose, left_face, right_face, chin]):
            return None

        cx = int(nose.x * w)
        cy = int(nose.y * h)
        fw = int(abs(right_face.x - left_face.x) * w * 0.4)
        fh = int(abs(chin.y - nose.y) * h * 0.7)

        return (max(cx - fw // 2, 0), max(cy - fh // 4, 0), min(cx + fw // 2, w), min(cy + fh * 3 // 4, h)
        )
    # === Box Estimation for Frontal View (FaceMesh) ===
    def get_box_from_facemesh(self, landmarks, w, h):
        try:
            nose, left, right, chin = landmarks[1], landmarks[234], landmarks[454], landmarks[152]
        except IndexError:
            rospy.logdebug(f"[rr_tracker/facial_tracking] FaceMesh does not have sufficient landmarks for nose tracking")
            return None
        cx, cy = int(nose.x * w), int(nose.y * h)
        fw = int(abs(right.x - left.x) * w * 0.25)
        fh = int(abs(chin.y - nose.y) * h * 0.4)
        return (max(cx - fw // 2, 0), max(cy - fh // 4, 0), min(cx + fw // 2, w), min(cy + fh * 3 // 4, h))
    
# =========================================================================================================
# =========================================== Utility Functions ===========================================
# =========================================================================================================
def valid_landmark(lm, threshold=0.5):
    return lm.visibility > threshold if hasattr(lm, 'visibility') else lm.visibility == 0 or lm.presence > threshold

def smooth_box(prev_box, new_box, alpha=0.3):
    if prev_box is None:
        return new_box
    return tuple([int(prev_box[i] * (1 - alpha) + new_box[i] * alpha) for i in range(4)])

# ========= Main =========
if __name__ == "__main__":
    FacialTrackingNode()
    rospy.spin()
