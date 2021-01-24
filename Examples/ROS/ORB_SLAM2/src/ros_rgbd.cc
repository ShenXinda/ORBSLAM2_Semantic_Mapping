/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h> 

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"

using namespace std;

bool createDirs(const std::string& dirName);
void saveImg(const cv::Mat& img, const string& base_path, const string& type, double timestamp);

std::vector<std::string> vrgb;
std::vector<std::string> vdepth;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD, const string& base_path);

    ORB_SLAM2::System* mpSLAM;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 4)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings <path_to_saveImages or null>" << endl;        
        ros::shutdown();
        return 1;
    }    

    string base_path = argv[3]; // null 表示不保存图像
    if (base_path != "null") {
        if (base_path[base_path.size()-1] != '/') {
            base_path.append("/");
        }
        if (!createDirs(base_path + "rgb/") || !createDirs(base_path + "depth/")) {
            std::cout << "Fail to create directories." << std::endl;
            return -1;
        }
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    sleep(5);  // 等待5s开始保存图像
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2,base_path));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // 按TUM数据集格式保存rgb.txt，depth.txt
    if (base_path!="null") {
        std::ofstream outfile; 
        outfile.open(base_path+"rgb.txt", std::ios::trunc); // ios::trunc表示在打开文件前将文件清空，由于是写入，文件不存在则创建
        if (!outfile.is_open()) {
            std::cout << "Open rgb.txt failed!" << std::endl;
            return -1;
        }
        outfile << "# color images\n" << "# timestamp filename\n";
        for (std::string line:vrgb) {
            outfile << line << "\n";
        }
        outfile.close();

        outfile.open(base_path+"depth.txt", std::ios::trunc);
        if (!outfile.is_open()) {
            std::cout << "Open depth.txt failed!" << std::endl;
            return -1;
        }
        outfile << "# depth images\n" << "# timestamp filename\n";
        for (std::string line:vdepth) {
            outfile << line << "\n";
        }
        outfile.close();
    }


    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD, const string& base_path)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (base_path != "null") {
        saveImg(cv_ptrRGB->image, base_path, "rgb", cv_ptrRGB->header.stamp.toSec());
        saveImg(cv_ptrD->image, base_path, "depth", cv_ptrD->header.stamp.toSec());
    }

    cv::Mat imRGB;
    cv::cvtColor(cv_ptrRGB->image, imRGB, cv::COLOR_BGR2RGB);
    mpSLAM->TrackRGBD(imRGB,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());
}

// 按TUM数据集格式保存RGB和深度图像
void saveImg(const cv::Mat& img, const string& base_path, const string& type, double timestamp)
{
    std::string path = base_path + type + "/" + std::to_string(timestamp) + ".png";
    std::cout << path << std::endl;
    if (type == "rgb") {
        cv::Mat image;
        cv::cvtColor(img, image, cv::COLOR_BGR2RGB);
        cv::imwrite(path, image);
    }else {
        cv::imwrite(path, img);
    }

    std::string line = std::to_string(timestamp) + " " + type + "/" + std::to_string(timestamp) + ".png";
    if (type == "rgb") {
        vrgb.push_back(line);
    }else {
        vdepth.push_back(line);
    }
}

// Linux下递归创建目录
bool createDirs(const std::string& dirName)
{
    uint32_t beginCmpPath = 0;
    uint32_t endCmpPath = 0;

    std::string fullPath = "";

    // LOGD("path = %s\n", dirName.c_str());
    
    if ('/' != dirName[0]) { //Relative path  
        fullPath = getcwd(nullptr, 0); //get current path
        beginCmpPath = fullPath.size();
        // LOGD("current Path: %s\n", fullPath.c_str());
        fullPath = fullPath + "/" + dirName;      
    }else { //Absolute path
        fullPath = dirName;
        beginCmpPath = 1;
    }

    if (fullPath[fullPath.size() - 1] != '/') {
        fullPath += "/";
    }  

    endCmpPath = fullPath.size();

    
    // create dirs;
    for (uint32_t i = beginCmpPath; i < endCmpPath ; i++ ) {
        if ('/' == fullPath[i]) {
        std::string curPath = fullPath.substr(0, i);
        if (access(curPath.c_str(), F_OK) != 0) {
            // if (mkdir(curPath.c_str(), S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH) == -1) {
            if (mkdir(curPath.c_str(), S_IRWXU|S_IRWXG|S_IRWXO) == -1) {
            // LOGD("mkdir(%s) failed(%s)\n", curPath.c_str(), strerror(errno));
            return false;
            }
        }
        }
    }
    
    return true;
}

