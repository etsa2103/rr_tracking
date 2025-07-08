/* Author: Jason Hughes
*  Date: May 2024
*  About: ROS node for the spinnaker wrapper in eeyore
*/

#include "eeyoreROS/electro_optical_wrapper.hpp"

ElectroOpticalWrapper::ElectroOpticalWrapper(ros::NodeHandle *nh, int h, int w, std::string t)
    : camera_(h,w,t)
{
    ROS_INFO("[EO CAMERA] Starting node");

    image_transport::ImageTransport it(*nh);
    cam_pub_ = it.advertise("camera/electro_optical", 10);

    camera_.quickStart();

    nh -> getParam("eo_node/frame_rate", frame_rate_);
    nh -> getParam("eo_node/frame_id", frame_id_);

    main();
}

ElectroOpticalWrapper::~ElectroOpticalWrapper()
{
    camera_.closeDevice();
}

void ElectroOpticalWrapper::main()
{
    ros::Rate rate(frame_rate_);
    int count = 0;
    ROS_INFO("[EO CAMERA] Waiting for camera...");
    while (count < 2*frame_rate_)
    {
        count ++;
        rate.sleep();
    }

    ROS_INFO("[EO CAMERA] Entering Aquisition Loop");
    while (ros::ok())
    {
        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
        cv::Mat img = camera_.getFrame();
        
        cv_ptr->encoding = "bgr8";
        cv_ptr->header.stamp = ros::Time::now();
        cv_ptr->header.frame_id = frame_id_;
        cv_ptr->image = img;
       
        sensor_msgs::ImagePtr msg = cv_ptr->toImageMsg();
        this -> cam_pub_.publish(msg);

        ros::spinOnce();
        rate.sleep();
    }
}
