/* Author: Jason Hughes
*  Date: April 2023
*  About: ROS wrapper for boson 
*/

#include "eeyoreROS/boson_wrapper.hpp"

BosonWrapper::BosonWrapper(ros::NodeHandle *nh,uint32_t dev, uint32_t baud, int w, int h, std::string video_id, std::string sensor_name) : camera_(dev,baud,w,h,video_id,sensor_name)
{
    image_transport::ImageTransport it(*nh);

    cam_pub_ = it.advertise("camera/boson", 10);

    std::string boson_cal_path;
    nh -> getParam("boson_node/cal_path", boson_cal_path);
    nh -> getParam("boson_node/frame_rate", frame_rate_);
    nh -> getParam("boson_node/frame_id", frame_id_);

    cv::Mat intrinsic = camera_.getParams(boson_cal_path, "K");
    cv::Mat distance = camera_.getParams(boson_cal_path, "D");
    camera_.setIntrinsicCoeffs(intrinsic);
    camera_.setDistanceCoeffs(distance);

    int result;
    // start the boson
    result = camera_.conductFcc();
    result = camera_.openSensor();

    serial_number_ = camera_.getSerialNumber();

    main();
}

BosonWrapper::~BosonWrapper()
{
    camera_.closeSensor();
}

void BosonWrapper::main()
{
    ros::Rate rate(frame_rate_);
    ROS_INFO("Acquiring Images...");

    while (ros::ok())
    {
        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
        cv::Mat img = camera_.getFrame();

        cv_ptr->encoding = "mono16";
        cv_ptr->header.stamp = ros::Time::now();
        cv_ptr->header.frame_id = frame_id_;
        cv_ptr->image = img;

        this -> cam_pub_.publish(cv_ptr->toImageMsg());

        ros::spinOnce();
        rate.sleep();
    }
}
