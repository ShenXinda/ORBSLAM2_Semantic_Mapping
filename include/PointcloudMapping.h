#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <map>
#include <unordered_map>
#include <boost/make_shared.hpp>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>  // Eigen核心部分
#include <Eigen/Geometry> // 提供了各种旋转和平移的表示
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/kdtree/kdtree_flann.h>



#include "KeyFrame.h"
#include "Converter.h"
#include "ProbPoint.h"


typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef pcl::PointXYZRGBA PointT; // A point structure representing Euclidean xyz coordinates, and the RGB color.
typedef pcl::PointCloud<PointT> PointCloud;

namespace ORB_SLAM2 {

class Converter;
class KeyFrame;
class ProbPoint;

class PointCloudMapping {
    public:
        PointCloudMapping(string savePCDPath, string pythonHome, double thprob=0.95, double thdepth=0.02);
        ~PointCloudMapping();
        void insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth); // 传入的深度图像的深度值单位已经是m
        void requestFinish();
        bool isFinished();
        void getGlobalCloudMap(PointCloud::Ptr &outputMap);

    private:
        void showPointCloud();

        void initColorMap();
        void runSegmantation();
        void getProbMap(PyObject* pModule, PyObject* pArg, cv::Mat& imD, cv::Mat& imRGB,const cv::Mat& pose);
        void assoiateAndUpdate(int channels, int height, int width, PyArrayObject *ListItem, cv::Mat& imD, const cv::Mat& pose);
        int mergeSomeClasses(int label);

        void generatePointCloud(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& pose, int nId); 
        

        double mCx, mCy, mFx, mFy;
        
        std::shared_ptr<std::thread>  viewerThread;
        std::shared_ptr<std::thread>  segThread;
  
        std::mutex mKeyFrameMtx;
        std::condition_variable mKeyFrameUpdatedCond;
        std::queue<KeyFrame*> mvKeyFrames;
        std::queue<cv::Mat> mvColorImgs, mvDepthImgs;

        std::mutex mSegMtx;
        std::condition_variable mSegCond;
        std::queue<cv::Mat> mvColorImgsSeg, mvDepthImgsSeg, mvPose;

        bool mbShutdown;
        bool mbFinish;
        bool mbFinishSeg;

        std::mutex mPointCloudMtx;
        PointCloud::Ptr mPointCloud, mSegPointCloud;

        std::unordered_map<int,std::pair<std::vector<int>, std::string>> colorMap;
        std::vector<std::string> things;

        std::map<int, ProbPoint*> mvPoint2ProbPoint;

        string mSavePCDPath;
        string mPythonHome;
        double mThdepth;
        double mThprob;
};

}
#endif