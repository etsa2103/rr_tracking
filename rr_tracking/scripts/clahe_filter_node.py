#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CLAHEFilter:
    def __init__(self):
        rospy.init_node('clahe_filter_node')
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/boson/image_raw", Image, self.callback)
        self.image_pub1 = rospy.Publisher("/boson/image_clahe1", Image, queue_size=1)
        self.image_pub2 = rospy.Publisher("/boson/image_clahe2", Image, queue_size=1)

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2,2))

        rospy.loginfo("CLAHE filter node started")
        rospy.spin()

    def preprocess_ir_image(self, raw_img, colormap=False):
        """
        Preprocess a grayscale IR image (mono8 or converted thermal) to enhance contrast
        and make it suitable for BlazePose or visualization.

        Parameters:
        - raw_img: Grayscale input image (uint8 or mono16 converted to uint8)
        - colormap: If True, applies COLORMAP_INFERNO to the output

        Returns:
        - processed_img: Enhanced image (grayscale or BGR)
        """
        
        # STEP 1: Percentile-based contrast stretching (clip to 2nd–98th percentile)
        p_low, p_high = np.percentile(raw_img, (5, 98))
        img_clipped = np.clip(raw_img, p_low, p_high)
        
        # STEP 2: Normalize to full 0–255 range
        norm_img = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # STEP 3: Gamma correction (brighten midtones, compress highlights)
        gamma = 1.3
        gamma_corr = np.power(norm_img / 255.0, gamma)
        gamma_img = (gamma_corr * 255).astype('uint8')

        # STEP 4: CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gamma_img)

        # STEP 5: Optional smoothing (reduce sensor noise without blurring edges)
        smooth_img = cv2.bilateralFilter(clahe_img, d=5, sigmaColor=75, sigmaSpace=75)

        # STEP 6: Optional colormap for visualization
        if colormap:
            final_img = cv2.applyColorMap(smooth_img, cv2.COLORMAP_INFERNO)
        else:
            final_img = smooth_img

        return final_img


    def callback(self, msg):
        try:
            # Convert ROS image to grayscale OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            processed = self.preprocess_ir_image(cv_image, colormap=False)

            # Convert back to ROS message and publish
            ros_image1 = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
            self.image_pub1.publish(ros_image1)
            # ros_image2 = self.bridge.cv2_to_imgmsg(processed, encoding='mono8')
            # self.image_pub2.publish(ros_image2)
        except Exception as e:
            rospy.logerr("Failed to process image: %s", str(e))

if __name__ == '__main__':
    CLAHEFilter()
