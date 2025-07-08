/* Author: Jason Hughes
*  Date: July 2023
*  About: file to start the eo node
*/

#include "eeyoreROS/electro_optical_wrapper.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "electro_optical_node");
    ros::NodeHandle nh;

    std::string trigger_type;

    nh.getParam("eo_node/trigger_type", trigger_type);

    try
    {
        int h = 1024;
        int w = 1280;
        ElectroOpticalWrapper node(&nh, h, w, trigger_type);
        ros::spin();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("%s: %s", nh.getNamespace().c_str(), e.what());
    }

    return 0;
}
