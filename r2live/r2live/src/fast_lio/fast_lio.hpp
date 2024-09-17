// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#pragma once
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <opencv/cv.h>
#include <common_lib.h>
#include <kd_tree/ikd_Tree.h>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/eigen.hpp>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <FOV_Checker/FOV_Checker.h>

#define INIT_TIME (0)
#define LASER_POINT_COV (0.00015)
#define NUM_MATCH_POINTS (5)

#define MAXN 360000
const int laserCloudWidth = 48;   //; 48*48*48 = 110592，原始LOAM是21x21x11
const int laserCloudHeight = 48;
const int laserCloudDepth = 48;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

//; 以下数据是在vio中定义的全局变量
// estimator inputs and output;
extern Camera_Lidar_queue g_camera_lidar_queue;
extern MeasureGroup Measures;
extern StatesGroup g_lio_state;   //; 维护的全局状态
extern std::shared_ptr<ImuProcess> g_imu_process;
extern double g_lidar_star_tim;  //; 在fast_lio用的IMU——Process中定义的变量


class Fast_lio
{
public:
    std::mutex m_mutex_lio_process;

    std::shared_ptr<ImuProcess> m_imu_process;
    std::string root_dir = ROOT_DIR;
    double m_maximum_pt_kdtree_dis = 1.0;
    double m_maximum_res_dis = 1.0;
    double m_planar_check_dis = 0.05;
    double m_lidar_time_delay = 0;
    double m_long_rang_pt_dis = 50.0;
    bool m_if_publish_feature_map = false;
    int iterCount = 0;
    int NUM_MAX_ITERATIONS = 0;
    int FOV_RANGE = 4; // range of FOV = FOV_RANGE * cube_len

    //; 注意下面这个是对应于上面的全局变量laserCloudWidth等变量的，也就是刚初始化的时候，世界坐标系原点应该
    //; 在立方体地图的中心，这里立方体地图长宽高取得都是偶数，影响也不大。但是感觉最好还是取成奇数吧，好理解一点。
    //; 比如这里24，那么长宽高应该是24*2-1 = 47
    int laserCloudCenWidth = 24;
    int laserCloudCenHeight = 24;
    int laserCloudCenDepth = 24;

    int laserCloudValidNum = 0;
    int laserCloudSelNum = 0;

    double T1[MAXN], T2[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN];
    int time_log_counter = 0;

    // IMU relative variables
    std::mutex mtx_buffer;
    std::condition_variable sig_buffer;
    bool lidar_pushed = false;
    bool flg_exit = false;
    bool flg_reset = false;

    // Buffers for measurements
    double cube_len = 0.0;
    double lidar_end_time = 0.0;
    double last_timestamp_lidar = -1;
    double last_timestamp_imu = -1;
    double HALF_FOV_COS = 0.0;
    double FOV_DEG = 0.0;
    double res_mean_last = 0.05;
    double total_distance = 0.0;
    Eigen::Vector3d position_last = Zero3d;
    double copy_time, readd_time, fov_check_time, readd_box_time, delete_box_time;
    double kdtree_incremental_time, kdtree_search_time;

    std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

    // surf feature in map
    //; 这个就是最后进行scan to map使用的局部地图！在lasermap_fov_segment函数中进行构建，
    //; 这个函数中还根据Livox小视角的特点对局部地图进行了是否在视角内的判断，所以比原始的LOAM多了一些步骤，
    //; 但是构建立方体地图的方式还是一样的，就是判断当前帧的位置，然后移动维护的整个立方体地图
    PointCloudXYZI::Ptr featsFromMap;    // (new PointCloudXYZI());
    PointCloudXYZI::Ptr cube_points_add; // (new PointCloudXYZI());
    // all points
    PointCloudXYZI::Ptr laserCloudFullRes2; // (new PointCloudXYZI());

    Eigen::Vector3f XAxisPoint_body;  // (LIDAR_SP_LEN, 0.0, 0.0);
    Eigen::Vector3f XAxisPoint_world; // (LIDAR_SP_LEN, 0.0, 0.0);

    std::vector<BoxPointType> cub_needrm;
    std::vector<BoxPointType> cub_needad;

    //; 指针数组，也就是存在laserCloudNum个点云指针，实际上也就是存在laserCloudNum块点云
    PointCloudXYZI::Ptr featsArray[laserCloudNum];  //; 这个就是维护的整个立方体局部地图的数组
    bool _last_inFOV[laserCloudNum];
    bool now_inFOV[laserCloudNum];
    bool cube_updated[laserCloudNum];
    int laserCloudValidInd[laserCloudNum];
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor; // (new pcl::PointCloud<pcl::PointXYZI>());

#ifdef USE_ikdtree
    KD_TREE ikdtree;
#else
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
#endif

#ifdef USE_FOV_Checker
    FOV_Checker fov_checker;
    double FOV_depth;
    double theta;
    Eigen::Vector3d FOV_axis;
    Eigen::Vector3d FOV_pos;
    vector<BoxPointType> boxes;
#endif

    ros::Publisher pubLaserCloudFullRes;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;
    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;
    bool dense_map_en, flg_EKF_inited = 0, flg_map_inited = 0, flg_EKF_converged = 0;
    int effect_feat_num = 0, frame_num = 0;
    double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg, deltaT, deltaR, aver_time_consu = 0, first_lidar_time = 0;
    double filter_size_surf_min_z;
    geometry_msgs::PoseStamped msg_body_pose;
    nav_msgs::Odometry odomAftMapped;
    PointType pointOri, pointSel, coeff;
    std::string map_file_path;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    /*** debug record ***/
    std::ofstream fout_pre, fout_out;
    ros::NodeHandle nh;

    void SigHandle(int sig)
    {
        flg_exit = true;
        ROS_WARN("catch sig %d", sig);
        sig_buffer.notify_all();
    }

    // 将激光点云投影到世界坐标系下（带强度信息）
    void pointBodyToWorld(PointType const *const pi, PointType *const po)
    {
        Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
        Eigen::Vector3d p_global(g_lio_state.rot_end * (p_body + Lidar_offset_to_IMU) + g_lio_state.pos_end);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    // 将激光点云投影到世界坐标系下
    template <typename T>
    void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po)
    {
        Eigen::Vector3d p_body(pi[0], pi[1], pi[2]);
        Eigen::Vector3d p_global(g_lio_state.rot_end * (p_body + Lidar_offset_to_IMU) + g_lio_state.pos_end);
        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }

    // 将激光点云投影到世界坐标系下（可根据强度信息生成颜色信息）
    void RGBpointBodyToWorld(PointType const *const pi, pcl::PointXYZI *const po)
    {
        Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
        Eigen::Vector3d p_global(g_lio_state.rot_end * (p_body + Lidar_offset_to_IMU) + g_lio_state.pos_end);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;

        float intensity = pi->intensity;
        intensity = intensity - std::floor(intensity);

        int reflection_map = intensity * 10000;

        // 根据强度信息生成颜色信息
        // if (reflection_map < 30)
        // {
        //     int green = (reflection_map * 255 / 30);
        //     po->r = 0;
        //     po->g = green & 0xff;
        //     po->b = 0xff;
        // }
        // else if (reflection_map < 90)
        // {
        //     int blue = (((90 - reflection_map) * 255) / 60);
        //     po->r = 0x0;
        //     po->g = 0xff;
        //     po->b = blue & 0xff;
        // }
        // else if (reflection_map < 150)
        // {
        //     int red = ((reflection_map-90) * 255 / 60);
        //     po->r = red & 0xff;
        //     po->g = 0xff;
        //     po->b = 0x0;
        // }
        // else
        // {
        //     int green = (((255-reflection_map) * 255) / (255-150));
        //     po->r = 0xff;
        //     po->g = green & 0xff;
        //     po->b = 0;
        // }
    }

    // 得到cube的索引
    int cube_ind(const int &i, const int &j, const int &k)
    {
        return (i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k);
    }

    // 中心点是否超出视场角范围
    bool CenterinFOV(Eigen::Vector3f cube_p)
    {
        Eigen::Vector3f dis_vec = g_lio_state.pos_end.cast<float>() - cube_p;
        float squaredSide1 = dis_vec.transpose() * dis_vec;

        if (squaredSide1 < 0.4 * cube_len * cube_len)
            return true;

        dis_vec = XAxisPoint_world.cast<float>() - cube_p;
        float squaredSide2 = dis_vec.transpose() * dis_vec;
        float ang_cos = fabs(squaredSide1 <= 3) ? 1.0 : (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));

        return ((ang_cos > HALF_FOV_COS) ? true : false);
    }

    // 角点是否超出视场角范围
    bool CornerinFOV(Eigen::Vector3f cube_p)
    {
        Eigen::Vector3f dis_vec = g_lio_state.pos_end.cast<float>() - cube_p;
        float squaredSide1 = dis_vec.transpose() * dis_vec;

        dis_vec = XAxisPoint_world.cast<float>() - cube_p;
        float squaredSide2 = dis_vec.transpose() * dis_vec;

        float ang_cos = fabs(squaredSide1 <= 3) ? 1.0 : (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));

        return ((ang_cos > HALF_FOV_COS) ? true : false);
        std::unique_lock<std::mutex> lock(m_mutex_lio_process);
    }

    // Segment the map in lidar FOV  分割在lidar fov之内的地图
    void lasermap_fov_segment()
    {
        laserCloudValidNum = 0;

        // Step 0 转这个点是什么意思，这个好像是确定的点？
        pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // 机体坐标系转世界坐标系

        // Step 1 这个是计算当前lidar的位置在立方体中的索引，其中+0.5就是为了四舍五入
        int centerCubeI = int((g_lio_state.pos_end(0) + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
        int centerCubeJ = int((g_lio_state.pos_end(1) + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
        int centerCubeK = int((g_lio_state.pos_end(2) + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

        //; 这里--是针对负数情况的四舍五入
        if (g_lio_state.pos_end(0) + 0.5 * cube_len < 0)
            centerCubeI--;

        if (g_lio_state.pos_end(1) + 0.5 * cube_len < 0)
            centerCubeJ--;

        if (g_lio_state.pos_end(2) + 0.5 * cube_len < 0)
            centerCubeK--;

        bool last_inFOV_flag = 0;
        int cube_index = 0;

        cub_needrm.clear();
        cub_needad.clear();

        T2[time_log_counter] = Measures.lidar_beg_time;
        double t_begin = omp_get_wtime();

        // Step 2.下面就是判断是否lidar当前位置已经接近立方体地图的边缘了，如果接近了那么就移动地图的位置
        //; x方向到达立方体左边缘了，那么把整个立方体往左移，右边的那些地图丢弃掉
        while (centerCubeI < FOV_RANGE + 1)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int i = laserCloudWidth - 1;

                    PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                    last_inFOV_flag = _last_inFOV[cube_index];

                    for (; i >= 1; i--)
                    {
                        featsArray[cube_ind(i, j, k)] = featsArray[cube_ind(i - 1, j, k)];
                        //; 注意这里不仅要交换地图存储的位置，还要交换立方体上次是否在FOV中的标志位
                        _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i - 1, j, k)];
                    }

                    featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                    _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeI++;
            laserCloudCenWidth++;
        }

        //; x方向达到右边源，立方体向右移，丢掉左边的地图
        while (centerCubeI >= laserCloudWidth - (FOV_RANGE + 1))
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int i = 0;

                    PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                    last_inFOV_flag = _last_inFOV[cube_index];

                    for (; i >= 1; i--)
                    {
                        featsArray[cube_ind(i, j, k)] = featsArray[cube_ind(i + 1, j, k)];
                        _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i + 1, j, k)];
                    }

                    featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                    _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeI--;
            laserCloudCenWidth--;
        }

        while (centerCubeJ < (FOV_RANGE + 1))
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int j = laserCloudHeight - 1;

                    PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                    last_inFOV_flag = _last_inFOV[cube_index];

                    for (; i >= 1; i--)
                    {
                        featsArray[cube_ind(i, j, k)] = featsArray[cube_ind(i, j - 1, k)];
                        _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j - 1, k)];
                    }

                    featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                    _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeJ++;
            laserCloudCenHeight++;
        }

        while (centerCubeJ >= laserCloudHeight - (FOV_RANGE + 1))
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int j = 0;
                    PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                    last_inFOV_flag = _last_inFOV[cube_index];

                    for (; i >= 1; i--)
                    {
                        featsArray[cube_ind(i, j, k)] = featsArray[cube_ind(i, j + 1, k)];
                        _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j + 1, k)];
                    }

                    featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                    _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeJ--;
            laserCloudCenHeight--;
        }

        while (centerCubeK < (FOV_RANGE + 1))
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    int k = laserCloudDepth - 1;
                    PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                    last_inFOV_flag = _last_inFOV[cube_index];

                    for (; i >= 1; i--)
                    {
                        featsArray[cube_ind(i, j, k)] = featsArray[cube_ind(i, j, k - 1)];
                        _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j, k - 1)];
                    }

                    featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                    _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeK++;
            laserCloudCenDepth++;
        }

        while (centerCubeK >= laserCloudDepth - (FOV_RANGE + 1))
        {
            for (int i = 0; i < laserCloudWidth; i++)
            {
                for (int j = 0; j < laserCloudHeight; j++)
                {
                    int k = 0;
                    PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray[cube_ind(i, j, k)];
                    last_inFOV_flag = _last_inFOV[cube_index];

                    for (; i >= 1; i--)
                    {
                        featsArray[cube_ind(i, j, k)] = featsArray[cube_ind(i, j, k + 1)];
                        _last_inFOV[cube_ind(i, j, k)] = _last_inFOV[cube_ind(i, j, k + 1)];
                    }

                    featsArray[cube_ind(i, j, k)] = laserCloudCubeSurfPointer;
                    _last_inFOV[cube_ind(i, j, k)] = last_inFOV_flag;
                    laserCloudCubeSurfPointer->clear();
                }
            }

            centerCubeK--;
            laserCloudCenDepth--;
        }

        cube_points_add->clear();  //; 本次需要新往ikdtree中添加的点，也就是上次不在FOV内这次进入FOV内的小立方体内的点
        featsFromMap->clear();  //; 注意这里每次构建局部地图的时候，都会把上一次的局部地图清空
        //; 重置本次立方体地图中，所有小的立方体是否在当前Lidar FOV内的标志位，认为都不在相机FOV内
        memset(now_inFOV, 0, sizeof(now_inFOV));  
        copy_time = omp_get_wtime() - t_begin;
        double fov_check_begin = omp_get_wtime();

#ifdef USE_FOV_Checker
        BoxPointType env_box;
        env_box.vertex_min[0] = max(centerCubeI - FOV_RANGE, 0) * cube_len - laserCloudWidth * cube_len / 2.0;
        env_box.vertex_max[0] = min(centerCubeI + FOV_RANGE, laserCloudWidth) * cube_len - laserCloudWidth * cube_len / 2.0;
        env_box.vertex_min[1] = max(centerCubeJ - FOV_RANGE, 0) * cube_len - laserCloudHeight * cube_len / 2.0;
        env_box.vertex_max[1] = min(centerCubeJ + FOV_RANGE, laserCloudHeight) * cube_len - laserCloudHeight * cube_len / 2.0;
        env_box.vertex_min[2] = max(centerCubeK - FOV_RANGE, 0) * cube_len - laserCloudDepth * cube_len / 2.0;
        env_box.vertex_max[2] = min(centerCubeK + FOV_RANGE, laserCloudDepth) * cube_len - laserCloudDepth * cube_len / 2.0;
        fov_checker.Set_Env(env_box);
        fov_checker.Set_BoxLength(cube_len);
        FOV_depth = FOV_RANGE * cube_len;
        theta = ceil(FOV_DEG / 2.0) / 180 * PI_M;
        Eigen::Vector3卡d tmp = g_lio_state.rot_end.transpose() * Eigen::Vector3d(1, 0, 0);
        FOV_axis(0) = tmp(0);
        FOV_axis(1) = -tmp(1);
        FOV_axis(2) = -tmp(2);
        FOV_pos = g_lio_state.pos_end;
        fov_checker.check_fov(FOV_pos, FOV_axis, theta, FOV_depth, boxes);

        int cube_i, cube_j, cube_k;
        for (int i = 0; i < boxes.size(); i++)
        {
            cube_i = floor((boxes[i].vertex_min[0] + eps_value + laserCloudWidth * cube_len / 2.0) / cube_len);
            cube_j = floor((boxes[i].vertex_min[1] + eps_value + laserCloudHeight * cube_len / 2.0) / cube_len);
            cube_k = floor((boxes[i].vertex_min[2] + eps_value + laserCloudDepth * cube_len / 2.0) / cube_len);
            cube_index = cube_ind(cube_i, cube_j, cube_k);

#ifdef USE_ikdtree
            *cube_points_add += *featsArray[cube_index];
            featsArray[cube_index]->clear();
            now_inFOV[cube_index] = true;
            if (!_last_inFOV[cube_index])
            {
                cub_needad.push_back(boxes[i]);
                laserCloudValidInd[laserCloudValidNum] = cube_index;
                laserCloudValidNum++;
                _last_inFOV[cube_index] = true;
            }
#else
            *featsFromMap += *featsArray[cube_index];
            laserCloudValidInd[laserCloudValidNum] = cube_index;
            laserCloudValidNum++;
#endif
        }

#ifdef USE_ikdtree
        BoxPointType rm_box;
        for (int i = 0; i < laserCloudNum; i++)
        {
            if (_last_inFOV[i] && !now_inFOV[i])
            {
                cube_i = i % laserCloudWidth;
                cube_j = (i % (laserCloudWidth * laserCloudHeight)) / laserCloudWidth;
                cube_k = i / (laserCloudWidth * laserCloudHeight);
                rm_box.vertex_min[0] = cube_i * cube_len - laserCloudWidth * cube_len / 2.0;
                rm_box.vertex_max[0] = rm_box.vertex_min[0] + cube_len;
                rm_box.vertex_min[1] = cube_j * cube_len - laserCloudHeight * cube_len / 2.0;
                rm_box.vertex_max[1] = rm_box.vertex_min[1] + cube_len;
                rm_box.vertex_min[2] = cube_k * cube_len - laserCloudDepth * cube_len / 2.0;
                rm_box.vertex_max[2] = rm_box.vertex_min[2] + cube_len;
                cub_needrm.push_back(rm_box);
                _last_inFOV[i] = false;
            }
        }
#endif

#else 
        // Step 3. 根据livox的FOV，筛选FOV内的点云作为局部地图点
        //; 下面就是根据Livox小视角的特点，筛选在视角内的哪些点云作为局部配准的地图点云
        //; 注意：这里选择布局地图点的时候跟LOAM是一样的，虽然维护的是很大的一个立方体地图，但是我每次构建局部地图的时候，
        //;    只在当前位置的周围的几个立方体里面进行选择，而不是选择整个大的立方体地图
        for (int i = centerCubeI - FOV_RANGE; i <= centerCubeI + FOV_RANGE; i++)
        {
            for (int j = centerCubeJ - FOV_RANGE; j <= centerCubeJ + FOV_RANGE; j++)
            {
                for (int k = centerCubeK - FOV_RANGE; k <= centerCubeK + FOV_RANGE; k++)
                {
                    //; 再次判断所以有效，其实经过上面移动地图立方体的调整，这里是一定满足的
                    if (i >= 0 && i < laserCloudWidth &&
                        j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth)
                    {
                        //; 这个就是利用小立方体的索引和每个小立方体的长度，计算这个小立方体的中心在世界坐标系中的位置
                        Eigen::Vector3f center_p(cube_len * (i - laserCloudCenWidth),
                                                 cube_len * (j - laserCloudCenHeight),
                                                 cube_len * (k - laserCloudCenDepth));

                        float check1, check2;
                        float squaredSide1, squaredSide2;
                        float ang_cos = 1;
                        //; 这个小立方体上次是否在FOV内
                        bool &last_inFOV = _last_inFOV[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                        bool inFOV = CenterinFOV(center_p);  //; 再检查这个小立方体本次在不在FOV内

                        //; 如果上面inFOV=true，说明这个小立方体在本次的FOV内，但是这还不够，因为我们是使用小立方体的中心判断的
                        //; 下面就是上下左右前后各移动一个小立方体，看移动之后的中心是否仍然在本次FOV内，如果还在的话，说明
                        //; 这个小立方体内所有的点（不仅是中心点）一定都在本次的FOV内
                        for (int ii = -1; (ii <= 1) && (!inFOV); ii += 2)
                        {
                            for (int jj = -1; (jj <= 1) && (!inFOV); jj += 2)
                            {
                                for (int kk = -1; (kk <= 1) && (!inFOV); kk += 2)
                                {
                                    Eigen::Vector3f corner_p(cube_len * ii, cube_len * jj, cube_len * kk);
                                    corner_p = center_p + 0.5 * corner_p;

                                    inFOV = CornerinFOV(corner_p);
                                }
                            }
                        }

                        //; 对所有的小立方体中，对应的当前小立方体是否在本次FOV内的标志进行赋值
                        now_inFOV[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = inFOV;

#ifdef USE_ikdtree
                        /*** readd cubes and points ***/
                        //; 如果这个小立方体在本次FOV内
                        if (inFOV)
                        {
                            //; 计算这个小立方体在整个大立方体地图中的索引，因为整个大立方体是1维数组存储的
                            int center_index = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                            *cube_points_add += *featsArray[center_index];  //; 拼接这个小立方体中的点云
                            
                            //! 疑问：这里为什么要把featsArray中的这块清空呢？
                            //! 解答：因为这里只要判断这个小立方体在本次FOV内，就会把它加到ikdtree中，而不管它上次是否在FOV内。
                            //!   一般情况下一个小立方体肯定是连续的n帧lidar处理过程中都在FOV内的，那么上次的小立方体里的点加入
                            //!   ikdtree中之后如果没有对里面的点情况，那么下次判断仍然在FOV内，还会把这些点再次加入ikdtree，
                            //!   这样就造成了点云的重复加入。
                            featsArray[center_index]->clear();

                            //; 如果上次这个小立方体不在FOV内，这次又在FOV内，那么就要把这个小立方体加到ikdtree中
                            if (!last_inFOV)
                            {
                                BoxPointType cub_points;
                                //; 注意这里i<3就是遍历角点的xyz三个坐标轴，然后取中心坐标前左下、后右上得到立方体的角点
                                for (int i = 0; i < 3; i++)
                                {
                                    cub_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
                                    cub_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
                                }

                                cub_needad.push_back(cub_points);  //; 需要添加的立方体角点坐标
                                laserCloudValidInd[laserCloudValidNum] = center_index;
                                laserCloudValidNum++;
                            }
                        }

#else                   
                        //; 如果判断当前这个立方体仍然在LiDAR的视角范围内，那么就把这个立方体内的点加入到待匹配的局部地图中
                        if (inFOV)
                        {
                            int center_index = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                            
                            //! 重要！这里就是从立方体中取出点云，构成当前帧的局部地图
                            *featsFromMap += *featsArray[center_index];

                            laserCloudValidInd[laserCloudValidNum] = center_index;
                            laserCloudValidNum++;
                        }

                        last_inFOV = inFOV;
#endif
                    }
                }
            }
        }


#ifdef USE_ikdtree
        /*** delete cubes ***/
        // Step 4 再次遍历大立方体地图中的所有小立方体
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                for (int k = 0; k < laserCloudDepth; k++)
                {
                    int ind = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                    //; 如果这个小立方体不在本次FOV内，但是在上次FOV内，那么就要把它从ikdtree中删除
                    if ((!now_inFOV[ind]) && _last_inFOV[ind])
                    {
                        BoxPointType cub_points;
                        Eigen::Vector3f center_p(cube_len * (i - laserCloudCenWidth),
                                                 cube_len * (j - laserCloudCenHeight),
                                                 cube_len * (k - laserCloudCenDepth));

                        for (int i = 0; i < 3; i++)
                        {
                            cub_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
                            cub_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
                        }

                        cub_needrm.push_back(cub_points);
                    }

                    //; 更新上次是否在立方体内的标志
                    _last_inFOV[ind] = now_inFOV[ind];
                }
            }
        }
#endif
#endif
        fov_check_time = omp_get_wtime() - fov_check_begin;

        double readd_begin = omp_get_wtime();


        // Step 5 上面遍历立方体，对要删除的立方体角点位置、要增加的立方体角点位置（以及要增加的点）进行了计算
        // Step     但是还没有真正操作ikdtree，所以这里就根据上面记录的信息对ikdtree进行操作
#ifdef USE_ikdtree
        //; 删除立方体角点：这一点很重要，感觉这个就是在程序后面查询历史点云的重要前提！
        if (cub_needrm.size() > 0)
            ikdtree.Delete_Point_Boxes(cub_needrm);

        // 增加立方体角点
        delete_box_time = omp_get_wtime() - readd_begin;
        if (cub_needad.size() > 0)
            ikdtree.Add_Point_Boxes(cub_needad);

        //; 重要操作：增加了立方体角点还没用，还没有把点云加到kdtree中啊？所以这里就把点云加到ikdtree中
        readd_box_time = omp_get_wtime() - readd_begin - delete_box_time;
        if (cube_points_add->points.size() > 0)
            ikdtree.Add_Points(cube_points_add->points, true);
#endif
        readd_time = omp_get_wtime() - readd_begin - delete_box_time - readd_box_time;
    }


    // 特征点数据解析回调函数
    //; 这里只订阅了激光雷达中的平面点，处理也就是把消息存到buff中
    void feat_points_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg_in)
    {
        sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2(*msg_in));
        //; m_lidar_time_delay一直是0
        msg->header.stamp = ros::Time(msg_in->header.stamp.toSec() - m_lidar_time_delay);
        //; lidar_in调用一次后，会置位m_if_have_lidar_data标志为1，代表有数据了
        //; 这里应该是
        if (g_camera_lidar_queue.lidar_in(msg_in->header.stamp.toSec() + 0.1) == 0)
        {
            return;
        }

        mtx_buffer.lock();
        if (msg->header.stamp.toSec() < last_timestamp_lidar) // 检查lidar时间戳，不能小于-1，也不能小于后一帧点云的时间
        {
            ROS_ERROR("lidar loop back, clear buffer");
            lidar_buffer.clear();
        }

        lidar_buffer.push_back(msg);

        last_timestamp_lidar = msg->header.stamp.toSec();

        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    // imu数据解析回调函数
    //; 这里收到imu消息只是存储到buff中，没有计算积分
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
    {
        sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
        double timestamp = msg->header.stamp.toSec();
        g_camera_lidar_queue.imu_in(timestamp);  //; 记录输入的IMU的时间
        mtx_buffer.lock();

        // 检查imu的时间戳，当前帧时间应该大于上一帧时间
        if (timestamp < last_timestamp_imu)
        {
            ROS_ERROR("imu loop back, clear buffer");
            imu_buffer.clear();
            flg_reset = true;
        }

        last_timestamp_imu = timestamp;
        if (g_camera_lidar_queue.m_if_acc_mul_G)
        {
            msg->linear_acceleration.x *= G_m_s2;
            msg->linear_acceleration.y *= G_m_s2;
            msg->linear_acceleration.z *= G_m_s2;
        }

        imu_buffer.push_back(msg);
        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    // doc: lidar数据与imu数据同步
    // doc: 实际操作很简单，就是取出最前面的lidar数据，然后把这段时间内的对应的imu数据也取出来。
    // doc: 都取出来之后，把lidar和imu都从他们的buff中删除
    bool sync_packages(MeasureGroup &meas)
    {
        if (lidar_buffer.empty() || imu_buffer.empty())
        {
            return false;
        }

        /*** push lidar frame ***/
        //; 这里面取出lidar_buff中最老的数据
        if (!lidar_pushed)
        {
            meas.lidar.reset(new PointCloudXYZI());
            pcl::fromROSMsg(*(lidar_buffer.front()), *(meas.lidar));
            meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec();
            // curvature，即曲率属性中存储了时间（当前点相对于当前帧基准时间的时间差），即msg->points[i].offset_time
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); 
            meas.lidar_end_time = lidar_end_time;
            lidar_pushed = true;
        }

        // imu时间应该能包住lidar，即最后一个点的时间应该比imu时间小，所以imu的频率必须比lidar的频率高
        if (last_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        meas.imu.clear();
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) // imu的时间必须比lidar的时间大或相等
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            //! +0.02什么意思？是lidar和IMU之间的时间延时吗？
            if (imu_time > lidar_end_time + 0.02) // IMU频率得大于50HZ
                break;

            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lidar_buffer.pop_front();
        lidar_pushed = false;

        return true;
    }

    std::thread m_thread_process;
    Fast_lio()
    {
        printf_line;
        //; 发布当前帧lidar的去畸变的点云
        pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
        //; 当前帧的点云中成功和地图点建立点到平面关系的哪些点，因此称为有效点
        pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
        pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
        //; 发布里程计数据，就是当前lidar的位置和姿态
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
        pubPath = nh.advertise<nav_msgs::Path>("/path", 10);

        //; 两个很重要的订阅：IMU和LiDAR数据，两个回调函数操作也很简单，就是把IMU和LiDAR数据存到buff中
        //; 注意这里lio订阅的和vio订阅的是同一个imu消息
        sub_imu = nh.subscribe("/livox/imu", 2000000, &Fast_lio::imu_cbk, this, ros::TransportHints().tcpNoDelay());
        sub_pcl = nh.subscribe("/laser_cloud_flat", 2000000, &Fast_lio::feat_points_cbk, this, ros::TransportHints().tcpNoDelay());

        get_ros_parameter(nh, "fast_lio/dense_map_enable", dense_map_en, true);
        get_ros_parameter(nh, "fast_lio/lidar_time_delay", m_lidar_time_delay, 0.0);
        get_ros_parameter(nh, "fast_lio/max_iteration", NUM_MAX_ITERATIONS, 4);
        ros::param::get("fast_lio/map_file_path", map_file_path);
        // get_ros_parameter(nh, "fast_lio/map_file_path", map_file_path, "./");
        get_ros_parameter(nh, "fast_lio/fov_degree", fov_deg, 70.00);  //; 配置参数中是360度？
        get_ros_parameter(nh, "fast_lio/filter_size_corner", filter_size_corner_min, 0.4);
        get_ros_parameter(nh, "fast_lio/filter_size_surf", filter_size_surf_min, 0.4);
        get_ros_parameter(nh, "fast_lio/filter_size_surf_z", filter_size_surf_min_z, 0.4);
        get_ros_parameter(nh, "fast_lio/filter_size_map", filter_size_map_min, 0.4);
        get_ros_parameter(nh, "fast_lio/cube_side_length", cube_len, 100.0);
        get_ros_parameter(nh, "fast_lio/maximum_pt_kdtree_dis", m_maximum_pt_kdtree_dis, 3.0);
        get_ros_parameter(nh, "fast_lio/maximum_res_dis", m_maximum_res_dis, 3.0);
        get_ros_parameter(nh, "fast_lio/planar_check_dis", m_planar_check_dis, 0.05);
        get_ros_parameter(nh, "fast_lio/long_rang_pt_dis", m_long_rang_pt_dis, 50.0);
        get_ros_parameter(nh, "fast_lio/publish_feature_map", m_if_publish_feature_map, false);

        printf_line;
        featsFromMap = boost::make_shared<PointCloudXYZI>();
        cube_points_add = boost::make_shared<PointCloudXYZI>();
        laserCloudFullRes2 = boost::make_shared<PointCloudXYZI>();
        laserCloudFullResColor = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        XAxisPoint_body = Eigen::Vector3f(LIDAR_SP_LEN, 0.0, 0.0);
        XAxisPoint_world = Eigen::Vector3f(LIDAR_SP_LEN, 0.0, 0.0);

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min_z);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        printf_line;
        // info: 重点：开启Fast-lio线程
        m_thread_process = std::thread(&Fast_lio::process, this);
        printf_line;
    }

    ~Fast_lio(){};

    //; 重点：FAST-LIO主线程
    int process()
    {
        nav_msgs::Path path;
        path.header.stamp = ros::Time::now();
        path.header.frame_id = "/world";

        /*** variables definition ***/
        //; 状态变量维度18
        Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> G, H_T_H, I_STATE;
        G.setZero();
        H_T_H.setZero();
        I_STATE.setIdentity();

        cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
        PointCloudXYZI::Ptr feats_down(new PointCloudXYZI());
        PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
        PointCloudXYZI::Ptr coeffSel(new PointCloudXYZI());

        /*** variables initialize ***/
        FOV_DEG = fov_deg + 10;  //; fov_deg在配置参数中是360度？
        HALF_FOV_COS = std::cos((fov_deg + 10.0) * 0.5 * PI_M / 180.0);

        //; 这里好像是划分一个方框，跟LOAM一样？
        for (int i = 0; i < laserCloudNum; i++)
        {
            featsArray[i].reset(new PointCloudXYZI());
        }

#ifdef USE_FOV_Checker
        BoxPointType env_box;
        env_box.vertex_min[0] = -laserCloudWidth / 2.0 * cube_len;
        env_box.vertex_max[0] = laserCloudWidth / 2.0 * cube_len;
        env_box.vertex_min[1] = -laserCloudHeight / 2.0 * cube_len;
        env_box.vertex_max[1] = laserCloudHeight / 2.0 * cube_len;
        env_box.vertex_min[2] = -laserCloudDepth / 2.0 * cube_len;
        env_box.vertex_max[2] = laserCloudDepth / 2.0 * cube_len;
        fov_checker.Set_Env(env_box);
#endif
        //; 定义IMU处理类的对象
        std::shared_ptr<ImuProcess> p_imu(new ImuProcess());
        m_imu_process = p_imu;

        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
        if (fout_pre && fout_out)
            std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
        else
            std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;

        ros::Rate rate(5000);
        bool status = ros::ok();
        //; lidar_buffer是lidar回调函数中存储的数据
        g_camera_lidar_queue.m_liar_frame_buf = &lidar_buffer;  //; 类成员变量赋值给全局变量
        
        //; 真正lio运行的函数
        while (ros::ok())
        {
            if (flg_exit)
                break;

            // doc: 先进行一次回调，处理发来的lidar数据
            ros::spinOnce();

            // ??? 无缘无故在这睡1ms干啥？
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // doc: 一直等待，直到可以处理lidar数据
            while (g_camera_lidar_queue.if_lidar_can_process() == false)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            //; 1.unique_lock有手动的unlock()方法，可以手动解锁，而lock_guard没有，只能等待对象的生命周期结束后
            //;   自动解锁，所以相对来说不是很灵活；
            //; 2.unique_lock可以配合环境变量condition_variable实现多线程之间的同步和通信，而lock_guard不行。

            std::unique_lock<std::mutex> lock(m_mutex_lio_process);

            //; 这里写一个没用的if(1)应该是线程锁的问题？相当于一个{}指定了线程锁工作的范围
            if (1)
            {
                //; 等待lidar数据和imu数据同步完成，同步后的数据会存储在Measures中
                if (sync_packages(Measures) == 0)
                {
                    continue;
                }

                //; 可见这个变量起着一票否决的作用，如果这个标志不是true，那么lidar永远不能启动。
                //; 在开启lio线程之前，手动把这个变量设置成flase了， 然后在vio线程启动后，它里面又把这个标志置位了
                if (g_camera_lidar_queue.m_if_lidar_can_start == 0)
                {
                    continue;
                }

                int lidar_can_update = 1;

                // ANCHOR - Determine if LiDAR can perform update
                //; lidar数据无效，也就是当前帧雷达时间比上一次状态估计的时间还早，那么lidar数据就是老数据了，没用
                //! 还是没有理解这里在干什么。g_lio_state.last_update_time变量在VIO中也被赋值了。
                //!   如果是纯lio的话，那么
                if (Measures.lidar_beg_time + 0.1 < g_lio_state.last_update_time)
                {
                    if (1)
                    {
                        ROS_WARN("Drop LiDAR frame [C|L|I]: [ %.4f |%.4f | %.4f], %.4f ", g_lio_state.last_update_time - g_camera_lidar_queue.m_first_imu_time,
                                 Measures.lidar_beg_time - g_camera_lidar_queue.m_first_imu_time,
                                 g_camera_lidar_queue.m_last_imu_time - g_camera_lidar_queue.m_first_imu_time,
                                 g_lio_state.last_update_time - Measures.lidar_beg_time);
                    }

                    lidar_can_update = 0;  //; 这帧lidar
                    //continue; 
                    // doc: 按理这里 continue 不应该注释掉的，如果lidar的时间戳早于 last_update_time，意味着这帧lidar数据是老数据，不应该处理。
                    // doc: 因为提取的 imu 数据只是在当前lidar帧时间戳之前的数据，然后从 last_update_time 时刻的 g_lio_state 进行积分，明显是不对的。
                }
                else
                {
                    if (0)
                    {
                        ROS_INFO("Acc LiDAR frame [C|L|I]: [ %.4f | %.4f | %.4f], %.4f  ", g_lio_state.last_update_time - g_camera_lidar_queue.m_first_imu_time,
                                 Measures.lidar_beg_time - g_camera_lidar_queue.m_first_imu_time,
                                 g_camera_lidar_queue.m_last_imu_time - g_camera_lidar_queue.m_first_imu_time,
                                 g_lio_state.last_update_time - Measures.lidar_beg_time);
                    }
                }

                lidar_can_update = 1;  //; ?又赋值了，那有啥用？
                //; 正常应该是0
                g_lidar_star_tim = first_lidar_time;   //; first_lidar_time是fast_lio的类成员变量

                if (flg_reset)
                {
                    ROS_WARN("reset when rosbag play back");
                    p_imu->Reset();
                    flg_reset = false;

                    continue;
                }

                double t0, t1, t2, t3, t4, t5, match_start, match_time, solve_start, solve_time, pca_time, svd_time;
                match_time = 0;
                kdtree_search_time = 0;
                solve_time = 0;
                pca_time = 0;
                svd_time = 0;
                t0 = omp_get_wtime();

                //;  1.IMU数据前向传播，注意是根据上一次的状态进行IMU积分，而不是预积分
                //;  2.把IMU数据后向反向传播，将激光雷达数据进行运动补偿到这一帧的最后时刻
                p_imu->Process(Measures, g_lio_state, feats_undistort);

                //; 注意这里是把IMU的acc gyro的协方差（vector3）更新到全局变量中，因为是通过状态方程传播的
                //; 初始化的时候给了常量？
                g_camera_lidar_queue.g_noise_cov_acc = p_imu->cov_acc;
                g_camera_lidar_queue.g_noise_cov_gyro = p_imu->cov_gyr;

                //; 定义一个局部变量，把IMU先验的状态存起来
                StatesGroup state_propagat(g_lio_state);  

                //; 判断有有效点云，再继续向下执行。
                //; 如果这里有一帧点云去畸变后是空的，那么会赋值lidar开始时间为当前这阵lidar数据的时间
                //; 正常情况下这个分支不会执行，所以first_lidar_time始终是0
                if (feats_undistort->empty() || (feats_undistort == NULL))
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    g_lio_state.last_update_time = first_lidar_time;
                    std::cout << "not ready for odometry" << std::endl;
                    continue;
                } 
                
                //! ? 这个条件怎么也不会成立啊？first_lidar_time正常情况是0，不正常情况也是上一帧lidar数据时间，
                //!   当前帧的时间肯定比它大啊？
                if ((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME)
                {
                    flg_EKF_inited = false;
                    std::cout << "||||||||||Initiallizing LiDar||||||||||" << std::endl;
                }
                //; ！！！！！！ 所以这里一定会执行
                else
                {
                    flg_EKF_inited = true;
                }

                /*** Compute the euler angle ***/
                Eigen::Vector3d euler_cur = RotMtoEuler(g_lio_state.rot_end);
                fout_pre << std::setw(10) << Measures.lidar_beg_time << " " << euler_cur.transpose() * 57.3 << " " << g_lio_state.pos_end.transpose() << " " << g_lio_state.vel_end.transpose()
                         << " " << g_lio_state.bias_g.transpose() << " " << g_lio_state.bias_a.transpose() << std::endl;
#ifdef DEBUG_PRINT
                std::cout << "current lidar time " << Measures.lidar_beg_time << " "
                          << "first lidar time " << first_lidar_time << std::endl;
                std::cout << "pre-integrated states: " << euler_cur.transpose() * 57.3 << " " << g_lio_state.pos_end.transpose() << " " << g_lio_state.vel_end.transpose() << " " << g_lio_state.bias_g.transpose() << " " << g_lio_state.bias_a.transpose() << std::endl;
#endif

                /*** Segment the map in lidar FOV ***/
                //; 根据lidar在W系下的位置，重新确定局部地图的包围盒角点，移除远端的点。
                //; 这个实际上和LOAM是一样的，就是排除栅格地图之外的点，方便后面进行点云匹配
                //! 注意：
                lasermap_fov_segment();   

                /*** downsample the features of new frame ***/
                //; 对当前帧的点云进行降采样处理，也是为了降低点云匹配时的计算量
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down);

#ifdef USE_ikdtree
                /*** initialize the map kdtree ***/
                //; 首先构建地图点的kdtree
                if ((feats_down->points.size() > 1) && (ikdtree.Root_Node == nullptr))
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    ikdtree.Build(feats_down->points);
                    flg_map_inited = true;

                    continue;
                }

                if (ikdtree.Root_Node == nullptr)
                {
                    flg_map_inited = false;
                    std::cout << "~~~~~~~Initiallize Map iKD-Tree Failed!" << std::endl;
                    continue;
                }

                //! 注意这里，地图中点的数目就是ikdtree的数目，应该是在前面lasermap_fov_segment()就选出了要使用的点
                int featsFromMapNum = ikdtree.size();   //; kdtree中点云的数目
#else
                if (featsFromMap->points.empty())
                {
                    downSizeFilterMap.setInputCloud(feats_down);
                }
                else
                {
                    downSizeFilterMap.setInputCloud(featsFromMap);
                }

                downSizeFilterMap.filter(*featsFromMap);
                int featsFromMapNum = featsFromMap->points.size();
#endif
                int feats_down_size = feats_down->points.size();  //; 当前帧点云降采样后的个数

                /*** ICP and iterated Kalman filter update ***/
                //; 最后计算出来的当前帧的点到平面的法向量，intensity中存储的是点到平面的距离
                PointCloudXYZI::Ptr coeffSel_tmpt(new PointCloudXYZI(*feats_down));
                PointCloudXYZI::Ptr feats_down_updated(new PointCloudXYZI(*feats_down));  //; 当前帧点云转到世界坐标系下之后的点
                std::vector<double> res_last(feats_down_size, 1000.0); // initial

                // Step 寻找平面，计算点到平面的距离作为残差，并计算H矩阵。
                //    Step 注意r2live里面只使用了平面点，和使用的是livox lidar有关吗？
                //; 大条件：kdtree中点云数据>=5, 因为要找最近的5个点
                if (featsFromMapNum >= 5)
                {
                    t1 = omp_get_wtime();

#ifdef USE_ikdtree 
                    //; 默认不发布特征地图，这里的特征地图应该就是用于匹配当前帧的点的栅格地图
                    if (m_if_publish_feature_map)
                    {
                        PointVector().swap(ikdtree.PCL_Storage);
                        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                        featsFromMap->clear();
                        featsFromMap->points = ikdtree.PCL_Storage;
                    }
#else
                    kdtreeSurfFromMap->setInputCloud(featsFromMap);
                    kdtree_incremental_time = omp_get_wtime() - t1;
#endif
                    //; 当前帧的这个点是否成功构成了点到平面的距离约束
                    std::vector<bool> point_selected_surf(feats_down_size, true);  
                    std::vector<std::vector<int>> pointSearchInd_surf(feats_down_size);
                    //; 当前帧的这个点在kdtree中找到的距离最近的五个点
                    std::vector<PointVector> Nearest_Points(feats_down_size);

                    int rematch_num = 0;
                    bool rematch_en = 0;
                    flg_EKF_converged = 0;
                    deltaR = 0.0;
                    deltaT = 0.0;
                    t2 = omp_get_wtime();
                    double maximum_pt_range = 0.0;
                    
                    //! 注意：就在这个大的for循环中进行了点云残差计算和迭代卡尔曼滤波
                    //; NUM_MAX_ITERATIONS 是 4， 配置文件中写的。也就是说这里也仅仅进行了4次卡尔曼滤波
                    for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++)
                    {
                        match_start = omp_get_wtime();
                        laserCloudOri->clear();
                        coeffSel->clear();

                        /** closest surface search and residual computation **/
                        //; 寻找最近的平面点，并进行残差计算
                        omp_set_num_threads(4);  //; 这里是使用多线程
                        // #pragma omp parallel for
                        //; 遍历的当前帧所有的降采样后的点云，寻找点到平面的残差匹配
                        for (int i = 0; i < feats_down_size; i++)
                        {
                            //! 注意这里是引用，只是为了防止复制降低计算量
                            PointType &pointOri_tmpt = feats_down->points[i];
                            //; 这个点云到它的坐标原点的距离
                            double ori_pt_dis = sqrt(pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z);
                            maximum_pt_range = std::max(ori_pt_dis, maximum_pt_range);  //; 每个点云都计算最大距离
                            
                            //! 注意这里的引用不仅为了防止赋值，还有直接更改原来的值的作用
                            //! 后面转移到世界坐标系下之后，原来的内存中存储的lidar坐标系下的点云也就转成了世界坐标系下的点云
                            PointType &pointSel_tmpt = feats_down_updated->points[i];  //; 当前帧点云转到世界系下
                            double search_start = omp_get_wtime();

                            /* transform to world frame */
                            //; 把当前帧的点转到世界系下，根据全局状态变量得到的最后的IMU相对实际坐标系的位姿进行转换
                            pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt);
                            std::vector<float> pointSearchSqDis_surf;
#ifdef USE_ikdtree
                            auto &points_near = Nearest_Points[i];
#else
                            auto &points_near = pointSearchInd_surf[i];
#endif
                            //; 如果是第一次迭代，才寻找点云的最近点，也就是建立观测方程。
                            //; 或者上一次迭代收敛了，那么rematch_en被置位，重新寻找一下最近点
                            if (iterCount == 0 || rematch_en)
                            {
                                point_selected_surf[i] = true;
                                /** Find the closest surfaces in the map **/
#ifdef USE_ikdtree
                                //; 在kdtree中寻找距离最近的5个点
                                //; 参数：当前点， 寻找几个距离最近的点， 找到的距离最近的点， 每个点的距离
                                ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
#else
                                kdtreeSurfFromMap->nearestKSearch(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
#endif
                                //; 寻找的五个点中的最大距离
                                float max_distance = pointSearchSqDis_surf[NUM_MATCH_POINTS - 1];
                                // max_distance to add residuals
                                // ANCHOR - Long range pt stragetry
                                //; 最大距离 > 0.5, 则无效
                                if (max_distance > m_maximum_pt_kdtree_dis)
                                {
                                    //; 当前帧点云的这个点找到的kdtree中匹配的点距离太远，说明当前帧这个点可能是外点
                                    point_selected_surf[i] = false;  
                                }
                            }

                            kdtree_search_time = omp_get_wtime() - search_start;
                            if (point_selected_surf[i] == false)
                                continue;

                            double pca_start = omp_get_wtime();

                            // PCA (using minimum square method)
                            //; PCA (最小二乘法)
                            cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
                            cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
                            cv::Mat matX0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(0));

                            for (int j = 0; j < NUM_MATCH_POINTS; j++)
                            {
#ifdef USE_ikdtree
                                matA0.at<float>(j, 0) = points_near[j].x;
                                matA0.at<float>(j, 1) = points_near[j].y;
                                matA0.at<float>(j, 2) = points_near[j].z;
#else
                                matA0.at<float>(j, 0) = featsFromMap->points[points_near[j]].x;
                                matA0.at<float>(j, 1) = featsFromMap->points[points_near[j]].y;
                                matA0.at<float>(j, 2) = featsFromMap->points[points_near[j]].z;
#endif
                            }

                            // matA0*matX0=matB0
                            // AX+BY+CZ+D = 0 <=> AX+BY+CZ=-D <=> (A/D)X+(B/D)Y+(C/D)Z = -1
                            // (X,Y,Z)<=>mat_a0
                            // A/D, B/D, C/D <=> mat_x0
                            //; 这里直接求解Ax = b这个超定方程，得到了平面的方程系数，跟LOAM有点像
                            cv::solve(matA0, matB0, matX0, cv::DECOMP_QR); //TODO
                            //; 求出来x的就是这个平面的法向量
                            float pa = matX0.at<float>(0, 0);
                            float pb = matX0.at<float>(1, 0);
                            float pc = matX0.at<float>(2, 0);
                            float pd = 1;

                            // ps is the norm of the plane norm_vec vector
                            // pd is the distance from point to plane
                            //; 归一化，将法向量模长统一为1
                            float ps = sqrt(pa * pa + pb * pb + pc * pc);
                            pa /= ps;
                            pb /= ps;
                            pc /= ps;
                            pd /= ps;

                            bool planeValid = true;
                            //; 还要依次检查这些点是否符合这个平面
                            for (int j = 0; j < NUM_MATCH_POINTS; j++)
                            {
#ifdef USE_ikdtree
                                // ANCHOR -  Planar check
                                //; 每个点代入平面方程，计算点到平面的距离，如果距离大于0.05m认为这个平面曲率偏大，就是无效的平面
                                if (fabs(pa * points_near[j].x +
                                         pb * points_near[j].y +
                                         pc * points_near[j].z + pd) > m_planar_check_dis) // Raw 0.05
#else
                                if (fabs(pa * featsFromMap->points[points_near[j]].x +
                                         pb * featsFromMap->points[points_near[j]].y +
                                         pc * featsFromMap->points[points_near[j]].z + pd) > 0.1)
#endif
                                {
                                    // ANCHOR - Far distance pt processing
                                    //; 对距离太远的点也进行舍弃
                                    if (ori_pt_dis < maximum_pt_range * 0.90 || (ori_pt_dis < m_long_rang_pt_dis))
                                    {
                                        planeValid = false;
                                        point_selected_surf[i] = false;
                                        break;
                                    }
                                }
                            }

                            //; 如果平面有效，那么就构造点到平面的距离的残差
                            if (planeValid)
                            {
                                //! 重要：点到平面的距离公式推导：https://www.cnblogs.com/graphics/archive/2010/07/10/1774809.html
                                // loss fuction
                                //; 计算当前点到平面的距离，注意这里并不是严格的距离，因为没有abs绝对值
                                float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                                //; 一个简单的鲁棒核函数：分母不是很明白，为了更多的面点用起来？
                                float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y + pointSel_tmpt.z * pointSel_tmpt.z));
                                // ANCHOR -  Point to plane distance
                                //; 这里一直是0.3
                                double acc_distance = (ori_pt_dis < m_long_rang_pt_dis) ? m_maximum_res_dis : 1.0;
                               
                                //; 这里点到平面的距离要<0.3，不能太远，如果太远那么就认为这是个外点，或者这是一个新的点匹配不上
                                if (pd2 < acc_distance)
                                {
                                    point_selected_surf[i] = true;
                                    //; 平面的法向量，就是梯度下降的方向？
                                    //! 梯度下降的方向和这个法向量有什么关系？
                                    coeffSel_tmpt->points[i].x = pa;
                                    coeffSel_tmpt->points[i].y = pb;
                                    coeffSel_tmpt->points[i].z = pc;
                                    //; 注意强度存储的是论文中公式17, 对应u' * (p-q)
                                    coeffSel_tmpt->points[i].intensity = pd2; 

                                    res_last[i] = std::abs(pd2);  //; 把残差（点到平面距离）存储到数组中
                                }
                                else
                                {
                                    point_selected_surf[i] = false;
                                }
                            }

                            pca_time += omp_get_wtime() - pca_start;
                        }
                        
                        //! ------ 至此，已经把当前lidar帧中所有点云寻找了点到平面的匹配关系

                        double total_residual = 0.0;
                        laserCloudSelNum = 0;
                        
                        //; 遍历当前帧中成功构造了点到平面距离残差的那些点
                        for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
                        {
                            //; 这个点建立了残差关系，并且残差<2。
                            if (point_selected_surf[i] && (res_last[i] <= 2.0))
                            {
                                //; 建立残差的当前帧的lidar点云
                                laserCloudOri->push_back(feats_down->points[i]);
                                //; 建立平面的法向量，强度存储残差
                                coeffSel->push_back(coeffSel_tmpt->points[i]);
                                total_residual += res_last[i];  //; 残差和
                                laserCloudSelNum++;   //; 构成残差的个数
                            }
                        }

                        res_mean_last = total_residual / laserCloudSelNum;  //; 残差均值（距离）

                        match_time += omp_get_wtime() - match_start;
                        solve_start = omp_get_wtime();

                        /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                        //; 计算雅克比矩阵H
                        //! 问题：这里H的列数是6，fast-lio中是12（论文中应该是18），都对应不上啊？
                        //! 解答：不是这样的，雅克比矩阵的列数是关于和这个残差有关的状态的，这里只有位姿，所以是6维度
                        Eigen::MatrixXd Hsub(laserCloudSelNum, 6);  //; N行6列的残差，因为LiDAR观测只和PQ有关
                        Eigen::VectorXd meas_vec(laserCloudSelNum);  //; 残差向量
                        Hsub.setZero();

                        // omp_set_num_threads(4);
                        // #pragma omp parallel for
                        //; 求观测值与误差的雅克比矩阵
                        for (int i = 0; i < laserCloudSelNum; i++)
                        {
                            const PointType &laser_p = laserCloudOri->points[i];  //; 建立残差关系的当前帧点云
                            Eigen::Vector3d point_this(laser_p.x, laser_p.y, laser_p.z);
                            //; lidar到imu的偏移，看来是没有旋转了。到这里就要弄明白，point_this是这个点云在当前帧的IMU系下的表示 
                            point_this += Lidar_offset_to_IMU; 
                            Eigen::Matrix3d point_crossmat;
                            //; 当前状态imu系下 点坐标反对称矩阵， 补充材料的Session D中S(4)的Pa
                            point_crossmat << SKEW_SYM_MATRX(point_this); 

                            /*** get the normal vector of closest surface/corner ***/
                            //; world系下 表示的平面法向量
                            const PointType &norm_p = coeffSel->points[i];
                            Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);  

                            /*** calculate the Measuremnt Jacobian matrix H ***/
                            //! 计算雅克比，但是这里始终对不上？
                            //; 计算点到平面的距离e对旋转的雅克比，正常应该是1x3的行向量，然后这里用列向量存储
                            //;  但是这里用公式推导的话，得到的行向量应该是 -u^T * R * P^, 转成列向量是-(P^)^T * R^T * u, 
                            //;               也和下面对不上啊
                            //! 破案了！靠，这里就是简写的写法，正确答案就是论文中的-(P^)^T * R^T * u，但是P^是反对称矩阵，所以
                            //! 先转置，前面再加符号，恰好抵消了！然后结果就可以化简成 (P^) * R^T * u
                            Eigen::Vector3d A(point_crossmat * g_lio_state.rot_end.transpose() * norm_vec);
                            Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;

                            /*** Measuremnt: distance to the closest surface/corner ***/
                            //! 注意：这里赋值是给残差负号，所以应该是z_m - h(x_op) = 0 - h(x_op)，
                            //! 即测量值z_m应该是点恰好在平面上， 从而点到平面的距离是0.
                            //; 这部分应该就是为了方便后面计算，后面写成zm - h(x_op) - H*(x^-x_op),
                            //; 但是zm应该是0（这个没有很想明白，但是也差不多，因为现在是误差状态，zm测量误差只能是0
                            //;  这样就简化成-h(x_op) - H*(x^-x_op)了，而H*(x^-x_op)就是后面的Hsub * vec.block<6, 1>(0, 0)
                            //; 所以这里meas_vec直接就是-h(x_op)了
                            meas_vec(i) = -norm_p.intensity;  //; 这个维度的残差进行赋值
                        }

                        Eigen::Vector3d rot_add, t_add, v_add, bg_add, ba_add, g_add;
                        Eigen::Matrix<double, DIM_OF_STATES, 1> solution;  //; 18x1，求解的状态更新量
                        Eigen::MatrixXd K(DIM_OF_STATES, laserCloudSelNum);  //; 卡尔曼增益，18xN

                        // Step 2 : 迭代卡尔曼滤波求解，更新当前状态
                        /*** Iterative Kalman Filter Update ***/
                        //; 这个分支正常情况下不会被执行
                        //! 不是！这个是IEKF第一次的时候需要进行初始化，后面的迭代就不用初始化了
                        if (!flg_EKF_inited)
                        {
                            cout << ANSI_COLOR_RED_BOLD << "Run EKF init" << ANSI_COLOR_RESET << endl;
                            /*** only run in initialization period ***/
                            //; 9行的9是怎么来的呢？
                            Eigen::MatrixXd H_init(Eigen::Matrix<double, 9, DIM_OF_STATES>::Zero());
                            Eigen::MatrixXd z_init(Eigen::Matrix<double, 9, 1>::Zero());
                            H_init.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();    //; Q
                            H_init.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();    //; P
                            H_init.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity();   //; g

                            //; 下面赋值重复了，有错误。
                            //; 这里的z_init就是g, 卡尔曼观测方程中 x_end = x_begin + K*(y-g)，g是观测方程在先验点结果，y是观测方程观测结果
                            z_init.block<3, 1>(0, 0) = -Log(g_lio_state.rot_end);
                            z_init.block<3, 1>(0, 0) = -g_lio_state.pos_end;

                            auto H_init_T = H_init.transpose();
                            //; 卡尔曼增益，fast-lio论文中的公式18。这里R是一个很小的单位阵
                            auto &&K_init = g_lio_state.cov * H_init_T * (H_init * g_lio_state.cov * H_init_T + 0.0001 * Eigen::Matrix<double, 9, 9>::Identity()).inverse();
                            solution = K_init * z_init;  //; K*(y-g)

                            solution.block<9, 1>(0, 0).setZero();
                            g_lio_state += solution;   //; x_end = x_begin + K*(y-g)
                            //; 更新后验协方差
                            g_lio_state.cov = (Eigen::MatrixXd::Identity(DIM_OF_STATES, DIM_OF_STATES) - K_init * H_init) * g_lio_state.cov;
                        }
                        
                        //; 正常情况，执行这里
                        //; 注意观测方程只和QP有关
                        else
                        {
                            //; 注意这里的用法，如果向输出特定颜色之后就把颜色恢复正常，那么后面要加RESET的命令
                            //; 另外注意紧接着使用了endl，这个接着刷新了IO缓冲区，不知道是不是必须的？
                            // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph" << ANSI_COLOR_RESET << endl;

                            //; K = (H^T * R^-1 * H + P^-1)^-1 * H^T * R^-1  (18xn * nxn *nx18) * 18xn * n*n = 18xn
                            //; Hsub是Nx6, Hsub_T就是6xN
                            auto &&Hsub_T = Hsub.transpose();
                            H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;  //; 18x18的左上6x6, 其他全部是0
                            //; r2live论文公式36，这里把公式又简化了，因为R是观测的协方差矩阵，也就是LiDAR观测的误差，是一个对角阵。
                            //; 求逆就相当于把每个点的观测误差都取倒数。然后下面的操作是把K中最右侧的R^-1先放到左边求逆的括号里抵消掉，
                            //; 这样可以降低计算量。所以这里的K_1就相当于在计算K中的括号部分，即18x18的矩阵
                            Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> &&K_1 =
                                //; LASER_POINT_COV 点云观测协方差，0.0015，也就是R^-1
                                (H_T_H + (g_lio_state.cov / LASER_POINT_COV).inverse()).inverse();
                            K = K_1.block<DIM_OF_STATES, 6>(0, 0) * Hsub_T;  //; 18x18 * 18xn → 18x6 * 6xn = 18xn
                            
                            // solution = K * meas_vec;
                            // g_lio_state += solution;

                            //! 公式对不上，看这个博客里面推导迭代卡尔曼滤波：
                            //!    https://blog.csdn.net/m0_38144614/article/details/116294266 
                            //! 另一篇文章写ESKF的，是GPS+IMU，写的不错，可以看一下：https://blog.csdn.net/u011341856/article/details/114262451
                            //; state_propagat是IMU计算出来的先验，g_lio_state是这次滤波的泰勒展开点
                            auto vec = state_propagat - g_lio_state;  //; IMU积分状态 - 上一次估计的后验状态
                            //; 18xn * (nx1 - nx6 * 6x1) = 18x1
                            solution = K * (meas_vec - Hsub * vec.block<6, 1>(0, 0));  //; vec.block<6, 1>(0, 0)得到Q和P
                            //; 这里和论文公式不一样，这里用的还是我们推导的那种形式的IEKF。思想不一样而已，论文中的公式是从最小二乘的角度
                            g_lio_state = state_propagat + solution;  //! 注意：这个地方可以看一下solution，我感觉是只更新了状态的

                            // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph, vec = " << vec.head<9>().transpose() << ANSI_COLOR_RESET << endl;


                            rot_add = solution.block<3, 1>(0, 0);
                            t_add = solution.block<3, 1>(3, 0);

                            flg_EKF_converged = false;
                            
                            //; 如果这次的更新量 - 上次的更新量 足够小，则EKF收敛
                            //! 变化<0.01度才收敛？？？要求这么高吗？
                            if (((rot_add.norm() * 57.3 - deltaR) < 0.01) && ((t_add.norm() * 100 - deltaT) < 0.015))
                            {
                                flg_EKF_converged = true;
                            }

                            deltaR = rot_add.norm() * 57.3;
                            deltaT = t_add.norm() * 100;
                        }

                        euler_cur = RotMtoEuler(g_lio_state.rot_end);
#ifdef DEBUG_PRINT
                        std::cout << "update: R" << euler_cur.transpose() * 57.3 << " p " << g_lio_state.pos_end.transpose() << " v " << g_lio_state.vel_end.transpose() << " bg" << g_lio_state.bias_g.transpose() << " ba" << g_lio_state.bias_a.transpose() << std::endl;
                        std::cout << "dR & dT: " << deltaR << " " << deltaT << " res norm:" << res_mean_last << std::endl;
#endif

                        /*** Rematch Judgement ***/
                        rematch_en = false;  //; 下一次迭代是否需要重新寻找kdtree中的最近点
                        //; rematch_num : 重新寻找最近点的次数
                        //; 1. 后面一个条件为真的情况：迭代三次了都没收敛，那么需要在kdtree中重新寻找最近点（那就剩一次迭代机会了啊？？）
                        //!     后面这个条件的判断真的存疑，不明白
                        if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
                        {
                            rematch_en = true;
                            rematch_num++;
                        }

                        /*** Convergence Judgements and Covariance Update ***/
                        //; 如果已经收敛或者到达最大迭代次数。这里用-1是有道理的，因为上面已经执行迭代了，但是iterCout还没++
                        if (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1)) // Fast lio ori version.
                        {
                            //; 正常都能进入这个分支
                            if (flg_EKF_inited)
                            {
                                /*** Covariance Update ***/
                                //; 更新协方差矩阵
                                G.block<DIM_OF_STATES, 6>(0, 0) = K * Hsub;  //; P_后验 = (I - KH) * P_先验
                                //; g_lio_state.cov在迭代过程中没更新，因此它也就是IMU先验的协方差，所以这里就是P_先验
                                g_lio_state.cov = (I_STATE - G) * g_lio_state.cov;  

                                //; 下面这俩变量没用
                                total_distance += (g_lio_state.pos_end - position_last).norm();
                                position_last = g_lio_state.pos_end;
                            }

                            solve_time += omp_get_wtime() - solve_start;
                            break;   //; 跳出整个迭代循环
                        }

                        solve_time += omp_get_wtime() - solve_start;
                    }

                    //! ----------------  以上步骤就已经完成了当前帧的位姿估计工作 -----------------------

                    t3 = omp_get_wtime();

                    /*** add new frame points to map ikdtree ***/
#ifdef USE_ikdtree
                    //! 重要：这里查询被删除的点。我感觉在ikdtree中删除点的操作很简单，应该就是用一个标志位就可以标识。
                    //!     而不用真的去删除这个点的内存，因为如果一个点先被删除，后来同样这个点又被添加进来了，
                    //!     如果之前的点被删除了，就要重新把它加到树中。
                    PointVector points_history;
                    ikdtree.acquire_removed_points(points_history);

                    memset(cube_updated, 0, sizeof(cube_updated));

                    //; 这里就是把在优化之前标记为FOV外的那些点，重新计算他们在大的立方体地图中属于哪个小立方体，
                    //; 然后重新加入到这些小立方体中。可见上面查询历史地图点的解释是错误的，也就是说，ikdtree确实会把
                    //; 要删除的点  真正  从ikdtree中删除
                    for (int i = 0; i < points_history.size(); i++)
                    {
                        PointType &pointSel = points_history[i];
                        //; 下面对地图的这个处理像是LOAM中的体素正方体的处理？
                        int cubeI = int((pointSel.x + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
                        int cubeJ = int((pointSel.y + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
                        int cubeK = int((pointSel.z + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

                        if (pointSel.x + 0.5 * cube_len < 0)
                            cubeI--;

                        if (pointSel.y + 0.5 * cube_len < 0)
                            cubeJ--;

                        if (pointSel.z + 0.5 * cube_len < 0)
                            cubeK--;

                        if (cubeI >= 0 && cubeI < laserCloudWidth &&
                            cubeJ >= 0 && cubeJ < laserCloudHeight &&
                            cubeK >= 0 && cubeK < laserCloudDepth)
                        {
                            int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                            featsArray[cubeInd]->push_back(pointSel);
                        }
                    }

                    // omp_set_num_threads(4);
                    // #pragma omp parallel for
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        /* transform to world frame */
                        //; 重要！ 这里就把当前帧的点云利用优化后的最准确的位姿，再投影到世界坐标系下！
                        pointBodyToWorld(&(feats_down->points[i]), &(feats_down_updated->points[i]));
                    }

                    t4 = omp_get_wtime();
                    //; 然后在这里把当前帧的点云加入到ikdtree中，注意这里是加入到ikdtree中，而不是立方体地图中，
                    //; 因为本质上我们最后用的还是ikdtree, 而非立方体地图！
                    ikdtree.Add_Points(feats_down_updated->points, true);
                    kdtree_incremental_time = omp_get_wtime() - t4 + readd_time + readd_box_time + delete_box_time;
#else
                    // Step 更新立方体地图
                    //; 这里就是把当前帧扫描到的点云利用当前帧最准确的位姿再次转移到世界坐标系下，然后加入到立方体中
                    bool cube_updated[laserCloudNum] = {0};
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        //; 这里我感觉还是有点小问题的，因为最后一个滤波之后得到了最终的位姿，但是此时feats_down_updated
                        //; 中存储的还是上一次滤波之后位姿转到世界坐标系下的点云，所以这里应该在把点云从原始扫描数据中
                        //; 利用最终优化后的位姿转移到世界坐标系下，得到最终的点云
                        PointType &pointSel = feats_down_updated->points[i];

                        //; 这里就是判断当前这个点所在的位置，应该属于总的立方体中的哪个小立方体
                        int cubeI = int((pointSel.x + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
                        int cubeJ = int((pointSel.y + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
                        int cubeK = int((pointSel.z + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

                        if (pointSel.x + 0.5 * cube_len < 0)
                            cubeI--;

                        if (pointSel.y + 0.5 * cube_len < 0)
                            cubeJ--;

                        if (pointSel.z + 0.5 * cube_len < 0)
                            cubeK--;
                        
                        //; 需要在总的立方体范围内，如果太远的点那么就直接抛弃不维护它在局部地图中
                        if (cubeI >= 0 && cubeI < laserCloudWidth &&
                            cubeJ >= 0 && cubeJ < laserCloudHeight &&
                            cubeK >= 0 && cubeK < laserCloudDepth)
                        {
                            int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                            //! 重要：把这个点加入到立方体中
                            featsArray[cubeInd]->push_back(pointSel);
                            cube_updated[cubeInd] = true;
                        }
                    }   

                    //; 为了防止点太多，往立方体地图中添加了当前帧的点之后，还要对齐进行降采样处理
                    for (int i = 0; i < laserCloudValidNum; i++)
                    {
                        int ind = laserCloudValidInd[i];

                        if (cube_updated[ind])
                        {
                            downSizeFilterMap.setInputCloud(featsArray[ind]);
                            downSizeFilterMap.filter(*featsArray[ind]);
                        }
                    }
#endif
                    t5 = omp_get_wtime();
                }

                // Step 开始发布rviz可视化消息
                /******* Publish current frame points in world coordinates:  *******/
                laserCloudFullRes2->clear();
                //; 是否显示稠密地图，配置文件中设置了true。这样会显示原始去畸变的点云，否则显示降采样的点云
                *laserCloudFullRes2 = dense_map_en ? (*feats_undistort) : (*feats_down);

                int laserCloudFullResNum = laserCloudFullRes2->points.size();   

                pcl::PointXYZI temp_point;
                laserCloudFullResColor->clear();  //; 转换到world系下的点云
                {
                    for (int i = 0; i < laserCloudFullResNum; i++)
                    {
                        //; 点云投到world系，并且根据强度设置颜色。但是实际颜色部分被作者注释掉了
                        RGBpointBodyToWorld(&laserCloudFullRes2->points[i], &temp_point);
                        laserCloudFullResColor->push_back(temp_point);
                    }

                    sensor_msgs::PointCloud2 laserCloudFullRes3;  //; 转成ROS消息
                    pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
                    laserCloudFullRes3.header.stamp.fromSec(Measures.lidar_end_time);
                    laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
                    pubLaserCloudFullRes.publish(laserCloudFullRes3); 
                    if (g_camera_lidar_queue.m_if_write_res_to_bag)
                    {
                        g_camera_lidar_queue.m_bag_for_record.write(pubLaserCloudFullRes.getTopic(), laserCloudFullRes3.header.stamp, laserCloudFullRes3);
                    }
                }

                /******* Publish Effective points *******/
                //; 发布有效点云：当前帧提取出来的面点，和世界坐标中的地图点成功建立了点到平面的距离
                {
                    laserCloudFullResColor->clear();
                    pcl::PointXYZI temp_point;
                    for (int i = 0; i < laserCloudSelNum; i++)
                    {
                        RGBpointBodyToWorld(&laserCloudOri->points[i], &temp_point);
                        laserCloudFullResColor->push_back(temp_point);
                    }

                    sensor_msgs::PointCloud2 laserCloudFullRes3;
                    pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
                    laserCloudFullRes3.header.stamp.fromSec(Measures.lidar_end_time); // .fromSec(last_timestamp_lidar);
                    laserCloudFullRes3.header.frame_id = "world";
                    pubLaserCloudEffect.publish(laserCloudFullRes3);
                }

                /******* Publish Maps:  *******/
                //; 这里发布的应该是地图中进行点云匹配用的栅格地图，也就是一个正方体里面的点
                sensor_msgs::PointCloud2 laserCloudMap;
                pcl::toROSMsg(*featsFromMap, laserCloudMap);
                laserCloudMap.header.stamp.fromSec(Measures.lidar_end_time); // ros::Time().fromSec(last_timestamp_lidar);
                laserCloudMap.header.frame_id = "world";
                pubLaserCloudMap.publish(laserCloudMap);

                /******* Publish Odometry ******/
                //; 发布里程计数据：里程计包括位姿和速度，其中位姿必须定义header.frame_id中，
                //; 速度必须定义在chidl_frame_id中, 可以发现下面没有给速度
                geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
                odomAftMapped.header.frame_id = "world";  
                odomAftMapped.child_frame_id = "/aft_mapped";  
                odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
                odomAftMapped.pose.pose.orientation.x = geoQuat.x;
                odomAftMapped.pose.pose.orientation.y = geoQuat.y;
                odomAftMapped.pose.pose.orientation.z = geoQuat.z;
                odomAftMapped.pose.pose.orientation.w = geoQuat.w;
                odomAftMapped.pose.pose.position.x = g_lio_state.pos_end(0);
                odomAftMapped.pose.pose.position.y = g_lio_state.pos_end(1);
                odomAftMapped.pose.pose.position.z = g_lio_state.pos_end(2);
                pubOdomAftMapped.publish(odomAftMapped);

                if (g_camera_lidar_queue.m_if_write_res_to_bag)
                {
                    g_camera_lidar_queue.m_bag_for_record.write(pubOdomAftMapped.getTopic(), ros::Time().fromSec(Measures.lidar_end_time), odomAftMapped);
                }
                
                /************   发布当前位姿和世界坐标的tf变换  ************/
                //! 为什么要发布tf变换？
                //! 解答：发布tf变换并不是必须的，而是看情况而定。
                // 1.首先如果自己的程序里面就需要用到查询tf变换（这个
                //  在lvi-sam里面有用到，就是vio的视觉前端查询最新的IMU频率的vio的tf变换得到当前图像帧的位姿，
                //  然后利用这个位姿把world系的局部点云投影到相机系中，进行视觉特征点和点云特征点的匹配），那么
                //  肯定要发布，但是这种情况一般都是用于两个线程之间通信的时候用到(比如上面说的lvi-sam的例子).
                // 2.其他很多时候的tf变换都是跟rviz显示有关的。比如说vio估计出来的特征点的点云是在第一次看到点
                //  的相机系下的，或者说我有一帧新扫描到的点云(lidar系下，原始扫描到的点云或者去畸变的点云)，
                //  那么在rviz中显示的时候，我都是把固定坐标系设置成world系（或者map系等），而我发布的点云
                //  是相对于camera系或者lidar系的，如果没有camera系/lidar系相对于world系的tf变换，那么rviz
                //  就不知道该怎么在世界坐标系下显示这些点云。所以很多情况下，发布tf是给rivz显示服务的
                static tf::TransformBroadcaster br;
                tf::Transform transform;
                tf::Quaternion q;
                transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                                odomAftMapped.pose.pose.position.y,
                                                odomAftMapped.pose.pose.position.z));
                q.setW(odomAftMapped.pose.pose.orientation.w);
                q.setX(odomAftMapped.pose.pose.orientation.x);
                q.setY(odomAftMapped.pose.pose.orientation.y);
                q.setZ(odomAftMapped.pose.pose.orientation.z);
                transform.setRotation(q);
                //; tf广播当前里程计的位姿和世界坐标系之间的关系
                br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(Measures.lidar_end_time), "world", "/aft_mapped"));
                
                //; msg_body_pose : PoseStamped，有时间戳的位姿
                msg_body_pose.header.stamp = ros::Time::now();
                msg_body_pose.header.frame_id = "/camera_odom_frame";
                msg_body_pose.pose.position.x = g_lio_state.pos_end(0);
                msg_body_pose.pose.position.y = g_lio_state.pos_end(1);
                msg_body_pose.pose.position.z = g_lio_state.pos_end(2);
                msg_body_pose.pose.orientation.x = geoQuat.x;
                msg_body_pose.pose.orientation.y = geoQuat.y;
                msg_body_pose.pose.orientation.z = geoQuat.z;
                msg_body_pose.pose.orientation.w = geoQuat.w;


                if (g_camera_lidar_queue.m_if_write_res_to_bag)
                {
                    // Trick from https://answers.ros.org/question/65556/write-a-tfmessage-to-bag-file/
                    tf::tfMessage message;
                    geometry_msgs::TransformStamped msg;
                    msg.header.frame_id = "/world";
                    msg.child_frame_id = "/aft_mapped";
                    msg.transform.rotation.w = geoQuat.w;
                    msg.transform.rotation.x = geoQuat.x;
                    msg.transform.rotation.y = geoQuat.y;
                    msg.transform.rotation.z = geoQuat.z;
                    msg.transform.translation.x = g_lio_state.pos_end(0);
                    msg.transform.translation.y = g_lio_state.pos_end(1);
                    msg.transform.translation.z = g_lio_state.pos_end(2);

                    message.transforms.push_back(msg);
                    g_camera_lidar_queue.m_bag_for_record.write("/tf", ros::Time().fromSec(Measures.lidar_end_time), message);
                }
//; 这是什么宏定义?   deploy 部署
#ifdef DEPLOY
                mavros_pose_publisher.publish(msg_body_pose);
#endif

                /******* Publish Path ********/
                msg_body_pose.header.frame_id = "world";
                //; 发布运动轨迹
                path.poses.push_back(msg_body_pose);
                pubPath.publish(path);


                if (g_camera_lidar_queue.m_if_write_res_to_bag)
                {
                    g_camera_lidar_queue.m_bag_for_record.write(pubPath.getTopic(), msg_body_pose.header.stamp, path);
                }

                /*** save debug variables ***/
                frame_num++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = aver_time_consu;
                s_plot2[time_log_counter] = kdtree_incremental_time;
                s_plot3[time_log_counter] = kdtree_search_time;
                s_plot4[time_log_counter] = fov_check_time;
                s_plot5[time_log_counter] = t5 - t0;
                s_plot6[time_log_counter] = readd_box_time;
                time_log_counter++;

                fout_out << std::setw(8) << laserCloudSelNum << " " << Measures.lidar_beg_time << " " << t2 - t0 << " " << match_time << " " << t5 - t3 << " " << t5 - t0 << std::endl;
            }

            status = ros::ok();
            rate.sleep();
        }

        return 0;
    }
};
