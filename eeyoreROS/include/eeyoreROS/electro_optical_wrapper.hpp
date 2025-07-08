#ifndef ELECTRO_OPTICAL_WRAPPER
#define ELECTRO_OPTICAL_WRAPPER

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "ros/ros.h"

#include "eeyore/electro_optical.hpp"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

class ElectroOpticalWrapper
{
    public:
        ElectroOpticalWrapper(ros::NodeHandle *nh, int h, int w, std::string t);
        ~ElectroOpticalWrapper();

    private:
        image_transport::Publisher cam_pub_;
        ElectroOpticalCam camera_;

        int frame_rate_;
        std::string frame_id_;

        void main();
};
#endif
