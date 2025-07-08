/* Author: Jason Hughes
*  Date: July 2024
*  About: Starts the boson node
*/

#include "eeyoreROS/boson_wrapper.hpp"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "boson_node");
    ros::NodeHandle nh;

    int32_t dev, baud;
    int height, width;
    std::string video_id, sensor_name;

    nh.getParam("boson_node/height", height);
    nh.getParam("boson_node/width", width);
    nh.getParam("boson_node/video_id", video_id);
    nh.getParam("boson_node/sensor_name", sensor_name);

    dev = 47;
    baud = 921600;

    ROS_INFO("[BOSON-NODE] Got setup params, starting node");

    try 
    {
        BosonWrapper node(&nh, dev, baud, width, height, video_id, sensor_name);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
    }

    return 0;
}
