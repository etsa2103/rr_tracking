/* Author: Jason Hughes
*  Date: April 2023
*  About: Header file for ros wrapper around boson
*/

#ifndef BOSON_WRAPPER_HPP
#define BOSON_WRAPPER_HPP

#include "ros/ros.h"
#include "eeyore/boson.hpp"

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

class BosonWrapper
{
    public:
        BosonWrapper(ros::NodeHandle *nh, uint32_t dev, uint32_t baud, int w, int h, std::string video_id, std::string sensor_name);
        ~BosonWrapper();

    private:
        image_transport::Publisher cam_pub_;
        Boson camera_;

        std::string frame_id_;
        std::string serial_number_;
        int frame_rate_;

        void main();
};
#endif
