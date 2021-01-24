#include "PointcloudMapping.h"
using namespace cv;

namespace ORB_SLAM2 {

PointCloudMapping::PointCloudMapping(double resolution): mCx(0), mCy(0), mFx(0), mFy(0), mbShutdown(false), mbFinish(false)
{
    mthprob = 0.95;
    mthdepth = 0.02;

    initColorMap();

    mPointCloud = boost::make_shared<PointCloud>();  // 用boost::make_shared<>
    mSegPointCloud = boost::make_shared<PointCloud>();

    segThread = std::make_shared<std::thread>(&PointCloudMapping::runSegmantation, this);  // make_unique是c++14的
    viewerThread = std::make_shared<std::thread>(&PointCloudMapping::showPointCloud, this); 
}

PointCloudMapping::~PointCloudMapping()
{
    segThread->join();
    viewerThread->join();
}

void PointCloudMapping::requestFinish()
{
    {
        unique_lock<mutex> locker(mKeyFrameMtx);
        mbShutdown = true;
    }
    // 唤醒等待的条件变量
    mKeyFrameUpdatedCond.notify_one();
    mSegCond.notify_one();
}

bool PointCloudMapping::isFinished()
{
    return mbFinish && mbFinishSeg;
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth)
{
    unique_lock<mutex> locker(mKeyFrameMtx);
    mvKeyFrames.push(kf);
    mvColorImgs.push( color.clone() );  // clone()函数进行Mat类型的深拷贝，为什幺深拷贝？？
    mvDepthImgs.push( depth.clone() );

    mKeyFrameUpdatedCond.notify_one();
    cout << "receive a keyframe, id = " << kf->mnId << endl;
}

void PointCloudMapping::showPointCloud() 
{
    // pcl::visualization::CloudViewer viewer("Dense pointcloud viewer");
    while(true) {   
        KeyFrame* kf;
        cv::Mat colorImg, depthImg;

        {
            std::unique_lock<std::mutex> locker(mKeyFrameMtx);
            while(mvKeyFrames.empty() && !mbShutdown){  // !mbShutdown为了防止所有关键帧映射点云完成后进入无限等待
                mKeyFrameUpdatedCond.wait(locker); 
            }            
            
            if (!(mvDepthImgs.size() == mvColorImgs.size() && mvKeyFrames.size() == mvColorImgs.size())) {
                std::cout << "这是不应该出现的情况！" << std::endl;
                continue;
            }

            if (mbShutdown && mvColorImgs.empty() && mvDepthImgs.empty() && mvKeyFrames.empty()) {
                break;
            }

            kf = mvKeyFrames.front();
            colorImg = mvColorImgs.front();    
            depthImg = mvDepthImgs.front();    
            mvKeyFrames.pop();
            mvColorImgs.pop();
            mvDepthImgs.pop();
        }

        if (mCx==0 || mCy==0 || mFx==0 || mFy==0) {
            mCx = kf->cx;
            mCy = kf->cy;
            mFx = kf->fx;
            mFy = kf->fy;
        }

        
        {
            std::unique_lock<std::mutex> locker(mPointCloudMtx);
            generatePointCloud(colorImg, depthImg, kf->GetPose(), kf->mnId);
            // viewer.showCloud(mPointCloud);
        }
        
        std::cout << "show point cloud, size=" << mPointCloud->points.size() << std::endl;
    }

    // 存储点云
    // string save_path = "/home/xshen/pcd_files/resultPointCloudFile.pcd";
    // pcl::io::savePCDFile(save_path, *mPointCloud);
    // cout << "save pcd files to :  " << save_path << endl;
    mbFinish = true;
}

void PointCloudMapping::generatePointCloud(const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& pose, int nId)
{ 
    std::cout << "Converting image: " << nId;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();     
    PointCloud::Ptr current(new PointCloud);
    for(size_t v = 1; v < imRGB.rows ; v+=3){  // 取每3*3的像素块的中心点
        for(size_t u = 1; u < imRGB.cols ; u+=3){
            float d = imD.ptr<float>(v)[u];
            if(d <0.01 || d>10){ // 深度值为0 表示测量失败
                continue;
            }

            PointT p;
            p.z = d;
            p.x = ( u - mCx) * p.z / mFx;
            p.y = ( v - mCy) * p.z / mFy;

            p.b = imRGB.ptr<uchar>(v)[u*3];
            p.g = imRGB.ptr<uchar>(v)[u*3+1];
            p.r = imRGB.ptr<uchar>(v)[u*3+2];

            current->points.push_back(p);
        }        
    }

    Eigen::Isometry3d T = Converter::toSE3Quat( pose );
    PointCloud::Ptr tmp(new PointCloud);
    // tmp为转换到世界坐标系下的点云
    pcl::transformPointCloud(*current, *tmp, T.inverse().matrix()); 

    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize( 0.01, 0.01, 0.01);
    voxel.setInputCloud(tmp);
    voxel.filter(*current);
    current->is_dense = true; 
    *mPointCloud += *current;
    
    {
        std::unique_lock<std::mutex> locker(mSegMtx);
        mvPose.push(pose.clone());
        mvColorImgsSeg.push( imRGB.clone() ); 
        mvDepthImgsSeg.push( imD.clone() );

        mSegCond.notify_one();
    }

    // // depth filter and statistical removal，离群点剔除
    // statistical_filter.setInputCloud(tmp);  
    // statistical_filter.filter(*current);   
    // (*mPointCloud) += *current;

    // pcl::transformPointCloud(*mPointCloud, *tmp, T.inverse().matrix());
    // // 加入新的点云后，对整个点云进行体素滤波
    // voxel.setInputCloud(mPointCloud);
    // voxel.filter(*tmp);
    // mPointCloud->swap(*tmp);
    // mPointCloud->is_dense = true; 

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); 
    std::cout << ", Cost = " << t << std::endl;
}


void PointCloudMapping::getGlobalCloudMap(PointCloud::Ptr &outputMap)
{
    std::unique_lock<std::mutex> locker(mPointCloudMtx);
    outputMap = mPointCloud;
}

void PointCloudMapping::assoiateAndUpdate(int channels, int height, int width, PyArrayObject *ListItem, cv::Mat& imD, const cv::Mat& pose)
{
    set<int> labels;
    Eigen::Isometry3d T = Converter::toSE3Quat( pose ).inverse(); // 注意要取逆，这与离线建图时不同
    for (size_t i = 0; i < mSegPointCloud->points.size(); i++) {
        Eigen::Vector3d pw;
        pw[0] = mSegPointCloud->points[i].x;
        pw[1] = mSegPointCloud->points[i].y;
        pw[2] = mSegPointCloud->points[i].z;
        
        Eigen::Vector3d pc = T.inverse() * pw;

        // 重投影像素坐标[u,v]
        double u = mFx*(pc[0]/pc[2]) + mCx;
        double v = mFy*(pc[1]/pc[2]) + mCy;

        if (u < 0 || u >= width || v < 0 || v >=height) {
            continue;
        }


        int uleft = floor(u), uright = ceil(u), vup = floor(v), vdown = ceil(v);
        vector<pair<int,int>> candidatePoint = {pair<int,int>(uleft, vup), 
            pair<int,int>(uleft, vdown), pair<int,int>(uright, vup), pair<int,int>(uright, vdown)};  // 重投影点周围的四个点

        double minValue = 10.0;
        int w = -1, h = -1;
        for (size_t k = 0;  k < candidatePoint.size(); k++) {
            int row = candidatePoint[k].second, col = candidatePoint[k].first;
            if (row >= imD.rows || col >= imD.cols) {  // 防止越界（图像是一维存储的，只有行和列的乘积大于了总的元素，程序才会错误；但是行和列超出了边界，读取的数据是错的）
                continue;
            }
            float d = imD.ptr<float>(row)[col]; // 图像存储行是高，列是宽
            if(d <0.01 || d>10){ // 深度值测量失败
                continue;
            }
            if (fabs(pc[2]-d) < minValue) {
                minValue = fabs(pc[2]-d);
                h = row;
                w = col;
            }
        }
        // 此处有个深度误差阈值
        if (minValue>mthdepth || w==-1 || h==-1) {
            continue;
        }
        imD.ptr<float>(h)[w] = 0;  // 已经匹配过的像素深度设置为0

        ProbPoint* probPoint = mvPoint2ProbPoint[i]; 

        vector<float> vNewProbs(channels);
        for (int c = 0; c < channels; c++) {
            vNewProbs[c] = *(float *)(ListItem->data + c*ListItem->strides[0] + h*ListItem->strides[1] + w*ListItem->strides[2]);     
        }
        
        int label = probPoint->BayesianUpdate(vNewProbs);

        if (label == -1 || (probPoint->getProbVector())[label] < mthprob) {
            continue;
        }

        // 182类太多了，对一些类进行合并
        label = mergeSomeClasses(label);
        
        if (colorMap.find(label) != colorMap.end() && std::find(things.begin(), things.end(), colorMap[label].second) != things.end()) {
        // if (colorMap.find(label) != colorMap.end()) {
            labels.insert(label);
            mSegPointCloud->points[i].b = colorMap[label].first[2];
            mSegPointCloud->points[i].g = colorMap[label].first[1];
            mSegPointCloud->points[i].r = colorMap[label].first[0];
        }
    }
    for (int la: labels) {
        cout << la << " ";
    }
    cout << endl;
}

void PointCloudMapping::getProbMap(PyObject* pModule, PyObject* pArg, cv::Mat& imD, cv::Mat& imRGB, const cv::Mat& pose)
{   
    PointCloud::Ptr current(new PointCloud);

    Eigen::Isometry3d T = Converter::toSE3Quat( pose );

    int numberOfLabels = -1;

    PyObject* pFunc = PyObject_GetAttrString(pModule, "runSegModel"); // 直接获取模块中的函数
    PyObject *pReturn = PyObject_CallObject(pFunc, pArg);         // 函数调用，pArg为函数调用的参数

    if (PyList_Check(pReturn)) { // 检查是否为List对象
        int SizeOfList = PyList_Size(pReturn);  //List对象的大小
        
        for (int Index_i = 0; Index_i < SizeOfList; Index_i++) {
            PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(pReturn, Index_i);//读取List中的PyArrayObject对象，这里需要进行强制转换。
            int channels = ListItem->dimensions[0], height = ListItem->dimensions[1], width = ListItem->dimensions[2];
            numberOfLabels = channels;

            if (mSegPointCloud->points.size()!=0) { // 地图中存在点，先进行关联
                assoiateAndUpdate(channels, height, width, ListItem, imD, pose);
            }

            for (int v = 1; v < height; v+=3) {
                for (int u = 1; u < width; u+=3) {
                    float d = imD.ptr<float>(v)[u];  // float类型！！！
                    if(d <0.01 || d>10){ // 深度值测量失败（或者是已经匹配过的像素）
                        continue;
                    }

                    PointT p;
                    p.z = d;
                    p.x = ( u - mCx) * p.z / mFx;
                    p.y = ( v - mCy) * p.z / mFy;

                    // 根据深度距离给颜色，越远颜色越淡
                    p.b = d/10*255;
                    p.g = d/10*255;
                    p.r = d/10*255;
                    // p.b = imRGB.ptr<uchar>(v)[u*3];
                    // p.g = imRGB.ptr<uchar>(v)[u*3+1];
                    // p.r = imRGB.ptr<uchar>(v)[u*3+2];
                    current->points.push_back(p);
                }
            }
            Py_DECREF(ListItem);
        } 
    } else {
        cout << "The data return from the python is not a List." << endl;
    }

    PointCloud::Ptr tmp(new PointCloud);
    pcl::transformPointCloud(*current, *tmp, T.inverse().matrix()); 
    *current = *tmp;
    // pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    // statistical_filter.setMeanK(50);
    // statistical_filter.setStddevMulThresh(1.0); // The distance threshold will be equal to: mean + stddev_mult * stddev
    // statistical_filter.setInputCloud(tmp);
    // statistical_filter.filter(*current);
    // current->is_dense = false; 
    int k = mSegPointCloud->points.size();

    assert(numberOfLabels==182);
    for (size_t i = 0;  i < current->points.size(); i++) {  // 初始化点概率均匀分布
        mvPoint2ProbPoint[k++] = new ProbPoint(numberOfLabels);
    }

    *mSegPointCloud += *current;
}

void PointCloudMapping::runSegmantation() 
{
    
    // Py_SetPythonHome("/home/xshen/Anaconda3/envs/deeplab-pytorch");  // ??ImportError: No module named site
    Py_Initialize(); // 对python进行初始化，无返回值。使用py_IsInitialized检查系统是否初始化成功。

    // import_array();  // 遇到问题RuntimeError: _ARRAY_API is not PyCObject object？

    // anaconda 下 deeplab-pytorch 环境
    PyRun_SimpleString("import sys");   // sys 模块包含了与 Python 解释器和它的环境有关的函数。
    PyRun_SimpleString("sys.path = []");
    PyRun_SimpleString("sys.path.append('/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/python36.zip')");
    PyRun_SimpleString("sys.path.append('/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/python3.6')");
    PyRun_SimpleString("sys.path.append('/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/python3.6/lib-dynload')");
    PyRun_SimpleString("sys.path.append('/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages')");
    PyRun_SimpleString("sys.path.append('/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/python3.6/site-packages/tqdm-4.7.2-py3.6.egg')");

    // 以下替代了import_array();
    PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");
    PyObject *c_api = NULL;
    c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
    PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);

    // 设置运行的 python 文件路径
    PyRun_SimpleString("sys.path.append('/home/xshen/my_workspace/deeplab-pytorch')");

    
    PyObject* pModule = nullptr;
    PyObject* pArg = nullptr;
    PyObject* pFunc = nullptr;

    
    pModule = PyImport_ImportModule("inference"); 
    
    pFunc= PyObject_GetAttrString(pModule, "init");   //这里是要调用的函数名
    PyObject_CallObject(pFunc, pArg);

    pcl::visualization::CloudViewer viewer("Pointcloud viewer");

    while (true) {
        cv::Mat imRGB, imD, pose;
        {
            std::unique_lock<std::mutex> locker(mSegMtx);
            
            while(mvColorImgsSeg.empty() && !mbShutdown){  // !mbShutdown为了防止所有关键帧映射点云完成后进入无限等待
                mSegCond.wait(locker); 
            }  
            cout << mbShutdown << " " << mvColorImgsSeg.size() << " " << mvDepthImgsSeg.size() << " " << mvPose.size() << endl;
            if (mbShutdown && mvColorImgsSeg.empty() && mvDepthImgsSeg.empty() && mvPose.empty()) {
                break;
            }
            imRGB = mvColorImgsSeg.front();
            imD = mvDepthImgsSeg.front();
            pose = mvPose.front();
            mvPose.pop();
            mvColorImgsSeg.pop();
            mvDepthImgsSeg.pop();
        }


        int m, n;
        n = imRGB.cols*3;
        m = imRGB.rows;
        unsigned char *data = (unsigned  char*)malloc(sizeof(unsigned char) * m * n);
        int p = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[p] = imRGB.at<unsigned char>(i, j);
                p++;
            }
        }
        
        npy_intp Dims[2] = { m, n }; //给定维度信息
        PyObject* pArray = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, data);
        PyObject* pArg = PyTuple_New(1);
        PyTuple_SetItem(pArg, 0, pArray);


        getProbMap(pModule, pArg, imD, imRGB, pose);

        viewer.showCloud(mSegPointCloud); 
    }

    // Py_Finalize(); // 关闭Python。会造成段错误？？

    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize( 0.01, 0.01, 0.01); // 每1cm^3一个点
    PointCloud::Ptr tmp(new PointCloud);
    *tmp = *mSegPointCloud;
    voxel.setInputCloud(tmp);
    voxel.filter(*mSegPointCloud);

    // 存储点云
    string save_path = "/home/xshen/pcd_files/resultPointCloudFile.pcd";
    pcl::io::savePCDFile(save_path, *mSegPointCloud);
    cout << "save pcd files to :  " << save_path << endl;

    mbFinishSeg = true;
}

void PointCloudMapping::initColorMap()
{
    // fr1_desk1,2  0,46,61,71,72,73,75,76,83,117
    // things = {"penson" , "cup", "chair", "tv", "laptop", "mouse", "keyboard", "cell phone", "book", "floor"}; 

    // fr2_desk 0,46,61,63,71,73,75,83,87,109,117,174(wall)
    // things = {"person", "cup", "chair", "potted-plant", "tv", "mouse", "keyboard", "book","teddy-bear","desk","floor"};  // with stuff
    // things = {"person", "cup", "chair", "potted-plant", "tv", "mouse", "keyboard", "book","teddy-bear"}; 


    // lab_20 36,46,61,71,73,74,75,76,83,109,117,174(wall)
    // things = {"sports-ball", "cup", "chair", "tv", "mouse","remote", "keyboard","cell-phone","book","desk","floor","wall"};  // with stuff
    // things = {"sports-ball", "cup", "chair", "tv", "mouse","remote","keyboard","cell-phone","book"};

    // office_desk_20 36,43,46,51,52,54,61,63,71,72,73,74,75,76,83,109,117,174(wall)
    things = {"sports-ball", "bottle","cup", "banana","apple","orange","chair", "potted-plant","tv", "mouse","remote","keyboard","cell-phone","book","desk","floor","wall"};

    std::ifstream fcolormap; //文件读操作
    string path = "./colormap.txt";
    fcolormap.open(path.c_str());
    //C++ eof()函数可以帮助我们用来判断文件是否为空，或是判断其是否读到文件结尾。
    while(!fcolormap.eof()) {
        string s;
        getline(fcolormap,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            int id, r, g, b;
            string name;
            ss >> id >> name >> r >> g >> b;
            colorMap[id] = std::pair<std::vector<int>, std::string>({r,g,b}, name);
        }
    }

}

int PointCloudMapping::mergeSomeClasses(int label) 
{
    // floor相关的统一归为floor
    if ((label>=113 && label<=117) || label==139) {
        label = 117;
    }
    // desk、table相关的统一归为table
    if (label == 66 || label==68 || label==109 || label==164) {
        label = 109;
    }
    // wall相关的统一归类为wall
    if (label>=170 && label<=176) {
        label = 174;
    }

    return label;
}


}