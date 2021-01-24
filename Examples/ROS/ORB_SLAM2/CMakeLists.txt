cmake_minimum_required(VERSION 2.4.6)
# include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
# message("ROS_ROOTis found:" $ENV{ROS_ROOT})

# rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
   # SET(ROS_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})  # 设置可执行文件输出路径

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

set(cv_bridge_DIR "/opt/ros/melodic/share/cv_bridge/cmake")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
)

# set(OpenCV_DIR "/usr/share/OpenCV") # opencv3.2.0 (ROS自带的opencv路径)
# set(OpenCV_DIR "/home/xshen/other/share/OpenCV") # opencv3.4.7
find_package(OpenCV 3.0 QUIET)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
FIND_PACKAGE(PCL 1.8 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${Pangolin_INCLUDE_DIRS}

${PCL_INCLUDE_DIRS}
${catkin_INCLUDE_DIRS}
"/home/xshen/Anaconda3/envs/deeplab-pytorch/include/python3.6m"
)

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
/usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/usr/lib/x86_64-linux-gnu/libboost_system.so

${PCL_LIBRARIES}
${catkin_LIBRARIES} 
"/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/python3.6"
"/home/xshen/Anaconda3/envs/deeplab-pytorch/lib/libpython3.6m.so"
)

# Node for monocular camera
# rosbuild_add_executable(Mono
# src/ros_mono.cc
# )

add_executable(Mono 
src/ros_mono.cc)

target_link_libraries(Mono
${LIBS}
)

# Node for monocular camera (Augmented Reality Demo)
# rosbuild_add_executable(MonoAR
# src/AR/ros_mono_ar.cc
# src/AR/ViewerAR.h
# src/AR/ViewerAR.cc
# )
add_executable(MonoAR 
src/AR/ros_mono_ar.cc
src/AR/ViewerAR.h
src/AR/ViewerAR.cc
)

target_link_libraries(MonoAR
${LIBS}
)

# Node for stereo camera
# rosbuild_add_executable(Stereo
# src/ros_stereo.cc
# )

add_executable(Stereo
src/ros_stereo.cc
)

target_link_libraries(Stereo
${LIBS}
)

# Node for RGB-D camera
# rosbuild_add_executable(RGBD
# src/ros_rgbd.cc
# )
add_executable(RGBD
src/ros_rgbd.cc
)

target_link_libraries(RGBD
${LIBS}
)

