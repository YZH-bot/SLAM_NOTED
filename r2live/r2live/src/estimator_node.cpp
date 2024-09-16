#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include "./fast_lio/fast_lio.hpp"
#define CAM_MEASUREMENT_COV 1e-3


//; 全局变量，Camera_Lidar_queue是一个结构体，存储camera_lidar数据
Camera_Lidar_queue g_camera_lidar_queue;
MeasureGroup Measures;    //; 一帧lidar和IMU 数据，这个是不是不应该定义？
StatesGroup g_lio_state;  //; lio的状态，里面包括 Q P V bg ba G  一共3*6=18维

Estimator estimator;

std::condition_variable con; // 条件变量
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

// 互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

// IMU项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

eigen_q diff_vins_lio_q = eigen_q::Identity();
vec_3 diff_vins_lio_t = vec_3::Zero();

bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = -1;

// 从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();

    // init_imu=1表示第一个IMU数据
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;

        return;
    }

    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.m_gravity;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.m_gravity;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
// 对imu_buf中剩余的imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

/*
    对imu和图像数据进行对齐并组合
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    std::unique_lock<std::mutex> lk(m_buf);
    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // 对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            sum_of_wait++;
            return measurements;
        }

        // 对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning [%.5f | %.5f] ", imu_buf.front()->header.stamp.toSec(), feature_buf.front()->header.stamp.toSec());
            feature_buf.pop();

            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        // 图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            // emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        // 这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, img_msg);
    }

    return measurements;
}

// imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &_imu_msg)
{
    sensor_msgs::ImuPtr imu_msg = boost::make_shared<sensor_msgs::Imu>();
    *imu_msg = *_imu_msg;

    //; 对于livox内置IMU，以G为单位进行了归一化，这里就是去归一化得到真实值
    //; m_if_acc_mul_G ： acc 是否 multipy G 
    if (g_camera_lidar_queue.m_if_acc_mul_G) // For LiVOX Avia built-in IMU
    {
        imu_msg->linear_acceleration.x *= 9.805;
        imu_msg->linear_acceleration.y *= 9.805;
        imu_msg->linear_acceleration.z *= 9.805;
    }
    //?-------- 增加结束 ----------

    // 判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }

    g_camera_lidar_queue.imu_in(imu_msg->header.stamp.toSec());

    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one(); // 唤醒作用于process线程中的获取观测值数据的函数

    last_imu_t = imu_msg->header.stamp.toSec();
    {
        // 构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        //; 仅仅使用IMU数据中值积分得到最新的PVQ，目的是为了实现高频的IMU里程计
        predict(imu_msg); // 递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        // 发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();

        while (!imu_buf.empty())
            imu_buf.pop();

        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }

    return;
}

// relocalization回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void unlock_lio(Estimator &estimator)
{
    if (estimator.m_fast_lio_instance)
    {
        estimator.m_fast_lio_instance->m_mutex_lio_process.unlock();
    }
}

void lock_lio(Estimator &estimator)
{
    if (estimator.m_fast_lio_instance)
    {
        estimator.m_fast_lio_instance->m_mutex_lio_process.lock();
    }
}

// ANCHOR - sync lio to cam
void sync_lio_to_vio(Estimator &estimator)
{
    check_state(g_lio_state);   //; 检查速度是否过大，但这里也没有接受返回值，所以没用
    int frame_idx = estimator.frame_count;
    frame_idx = WINDOW_SIZE;   //; 窗口大小改成了7

    //!  看g_lio_state.last_update_time 到底是什么时候被赋值的？
    if (abs(g_camera_lidar_queue.m_last_visual_time - g_lio_state.last_update_time) < 1.0)
    {
        if (g_lio_state.bias_a.norm() < 0.5 && g_lio_state.bias_g.norm() < 1.0)
        {
            estimator.Bas[frame_idx] = g_lio_state.bias_a;
            estimator.Bgs[frame_idx] = g_lio_state.bias_g;
            //! ???
            estimator.Vs[frame_idx] = diff_vins_lio_q.toRotationMatrix().inverse() * g_lio_state.vel_end;
            estimator.m_gravity = g_lio_state.gravity;       
            G_gravity = estimator.m_gravity;
            
            update();
        }
    }
}

void visual_imu_measure(const Eigen::Vector3d &pts_i, const Eigen::Vector3d &pts_j,
                        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
                        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj,
                        const Eigen::Vector3d &tic, const Eigen::Quaterniond &qic,
                        const double inv_dep_i,
                        Eigen::Vector2d &residual,
                        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> &j_mat)

{
    Eigen::Matrix2d sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;  //; 特征点在imu_i下的3D坐标
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;  //; 转到世界坐标系下
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);  //; 转到imu_j下的3D坐标
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);  //; 转到camera_j下的3D坐标

    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();

    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();   //; 观测残差，像素坐标
    residual = sqrt_info * residual;   //; 再乘以协方差矩阵的逆，相当于去掉单位归一化

    //; partial(e) / partial(p)
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
    reduce = sqrt_info * reduce;

    //; 这里只算对于imu_j的雅克比
    Eigen::Matrix<double, 3, 6> jaco_j;
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> j_mat_temp;   //; eigen默认是按照列优先在内存中存储数组
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);
    j_mat_temp = reduce * jaco_j;
    //; 下面就是把计算的关于旋转和平移的雅克比前后顺序交换了一下
    j_mat.block(0, 3, 2, 3) = j_mat_temp.block(0, 0, 2, 3);
    j_mat.block(0, 0, 2, 3) = j_mat_temp.block(0, 3, 2, 3);
}

//; 构造相机的测量
void construct_camera_measure(int frame_idx, Estimator &estimator,
                              std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &reppro_err_vec,
                              std::vector<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>> &J_mat_vec)
{
    J_mat_vec.clear();
    reppro_err_vec.clear();
    scope_color(ANSI_COLOR_GREEN_BOLD);
    int f_m_cnt = 0;
    int feature_index = -1;
    int min_frame = 3e8;
    int max_frame = -3e8;

    //; 遍历所有的视觉特征点，计算残差和雅克比
    for (auto &it_per_id : estimator.f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 1))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        //; 第一次看到这个特征点的那一帧下的归一化坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        Eigen::Vector2d residual_vec, residual_vec_old;
        std::vector<double *> parameters_vec;
        Eigen::Matrix<double, 2, 7, Eigen::RowMajor> j_mat_tq;
        Eigen::Matrix<double, 2, 6, Eigen::RowMajor> j_mat, j_mat_old;
        j_mat.setZero();
        j_mat_tq.setZero();

        //; 遍历看到这个特征点的所有帧，建立观测关系
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            //; WINDOW_SIZE = 7， 最后结果右边是2.3
            if (fabs(imu_i - imu_j) < std::max(double(WINDOW_SIZE / 3), 2.0))
            {
                continue;
            }

            min_frame = std::min(imu_j, min_frame);
            max_frame = std::max(imu_j, max_frame);
            min_frame = std::min(imu_i, min_frame);
            max_frame = std::max(imu_i, max_frame);
   
            //; 可以看到，这里只对滑窗中的最新帧建立约束关系
            if (imu_j == (frame_idx))
            {
                Vector3d pts_j = it_per_frame.point;
                double *jacobian_mat_vec[4];
                parameters_vec.push_back(estimator.m_para_Pose[imu_i]);  //; 位姿，7
                parameters_vec.push_back(estimator.m_para_Pose[imu_j]);  //; 位姿，7
                parameters_vec.push_back(estimator.m_para_Ex_Pose[0]);   //; 外参位姿，7
                parameters_vec.push_back(estimator.m_para_Feature[feature_index]);  //; 特征点逆深度

                //; 上面是优化参数的double数组，下面转成eigen数据类型
                Eigen::Vector3d Pi(parameters_vec[0][0], parameters_vec[0][1], parameters_vec[0][2]);
                Eigen::Quaterniond Qi(parameters_vec[0][6], parameters_vec[0][3], parameters_vec[0][4], parameters_vec[0][5]);

                Eigen::Vector3d Pj(parameters_vec[1][0], parameters_vec[1][1], parameters_vec[1][2]);
                Eigen::Quaterniond Qj(parameters_vec[1][6], parameters_vec[1][3], parameters_vec[1][4], parameters_vec[1][5]);

                Eigen::Vector3d tic(parameters_vec[2][0], parameters_vec[2][1], parameters_vec[2][2]);
                Eigen::Quaterniond qic(parameters_vec[2][6], parameters_vec[2][3], parameters_vec[2][4], parameters_vec[2][5]);

                double inverse_depth = parameters_vec[3][0];

                if (0)
                {
                    jacobian_mat_vec[0] = nullptr;
                    jacobian_mat_vec[1] = j_mat_tq.data();
                    jacobian_mat_vec[2] = nullptr;
                    jacobian_mat_vec[3] = nullptr;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    f->Evaluate(parameters_vec.data(), residual_vec_old.data(), (double **)jacobian_mat_vec);
                    j_mat_old.block(0, 3, 2, 3) = j_mat_tq.block(0, 0, 2, 3);
                    j_mat_old.block(0, 0, 2, 3) = j_mat_tq.block(0, 3, 2, 3);
                }

                if (1)
                {
                    //; 入参：i帧特征点归一化坐标， j帧特征点归一化坐标
                    visual_imu_measure(pts_i, pts_j, Pi, Qi, Pj, Qj, tic, qic, inverse_depth, residual_vec, j_mat);
                }

                if (0 && reppro_err_vec.size() == 0)
                {
                    cout << "============================" << endl;
                    cout << "Old ESIKF first res_vec: " << residual_vec_old.transpose() << endl;
                    cout << "ESIKF first res_vec: " << residual_vec.transpose() << endl;
                    cout << "ESIKF first H_mat [2,7]:\r\n"
                         << j_mat_tq << endl;
                    cout << "Old ESIKF first H_mat [2,6]:\r\n"
                         << j_mat_old << endl;
                    cout << "ESIKF first H_mat [2,6]:\r\n"
                         << j_mat << endl;
                }

                //; 误差太大，直接取消
                if (std::isnan(residual_vec.sum()) || std::isnan(j_mat.sum()))
                {
                    continue;
                }

                reppro_err_vec.push_back(residual_vec);
                J_mat_vec.push_back(j_mat);
            }
        }
        f_m_cnt++;
    }
}

int need_refresh_extrinsic = 0;

/*
    VIO的主线程
    等待并获取measurements：(IMUs, img_msg)s，计算dt
    estimator.processIMU()进行IMU预积分
    estimator.setReloFrame()设置重定位帧
    estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化
*/
void process()
{
     //?-------- 增加开始 ----------
    Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());
    
    //; 这里赋值了，所以m_if_lidar_can_start一直是true
    g_camera_lidar_queue.m_if_lidar_can_start = g_camera_lidar_queue.m_if_lidar_start_first;
    std_msgs::Header header;
    //?-------- 增加结束 ----------

    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        measurements = getMeasurements();
        if (measurements.size() == 0)
        {
            continue;
        }

        m_estimator.lock();
        //; 每次循环都会给设置一个负无穷大的数
        g_camera_lidar_queue.m_last_visual_time = -3e8;

        TicToc t_s;
        //; 遍历测量到的所有camera和imu数据
        for (auto &measurement : measurements)
        {
            // 对应这段的img data
            auto img_msg = measurement.second;

            //?-------- 增加开始 ----------
            int if_camera_can_update = 1;
            //; cam_update_tim ：当前帧的图像时间
            double cam_update_tim = img_msg->header.stamp.toSec() + estimator.td;
            
            // ANCHOR - determine if update of not.
            //; 此时已经开始了LIO线程，所以这个一定成立
            if (estimator.m_fast_lio_instance != nullptr)
            {
                g_camera_lidar_queue.m_camera_imu_td = estimator.td;

                //; 更新最新的图片时间
                g_camera_lidar_queue.m_last_visual_time = img_msg->header.stamp.toSec();
                
                //; 判断是否能够处理这一帧的camera数据，跟lidar的判断是一样的
                while (g_camera_lidar_queue.if_camera_can_process() == false)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                //! 注意：这里锁住了lio，也就是vio运行的时候lio是被锁住的！这样保证vio运行的时候， lio无法运行
                lock_lio(estimator);   
                t_s.tic();
                //; 这个变量没有使用
                double camera_LiDAR_tim_diff = img_msg->header.stamp.toSec() + g_camera_lidar_queue.m_camera_imu_td - g_lio_state.last_update_time;
                *p_imu = *(estimator.m_fast_lio_instance->m_imu_process);  //; IMU处理的类
            }

            //; 下面这个同步lio到vio什么意思？没太看懂
            //;   前面一个条件在while循环外赋值了是true了，所以一直满足。  后面一个条件配置文件写的是10，所以一直满足。
            if ((g_camera_lidar_queue.m_if_lidar_can_start == true) && (g_camera_lidar_queue.m_lidar_drag_cam_tim >= 0))
            {
                m_state.lock();
                //! 同步lio状态到vio状态! 注意里面没有同步P和Q
                sync_lio_to_vio(estimator);
                m_state.unlock();
            }
            //?-------- 增加结束 ----------

            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            int skip_imu = 0;
            // Step 1 首先处理所有的IMU数据，对IMU数据进行预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;

                // 发送IMU数据进行预积分
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;

                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // imu预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
                else
                {
                    //; 这里对最后一帧的IMU数据进行了插值处理，那为啥不对第一帧的IMU数据进行插值处理？
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
            }

            // Step 2 再检测是否有回环产生
            //! r2live中应该没有用到回环，所以这部分代码是从VINS中复制过来忘了删
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;

            // 取出最后一个重定位帧
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != NULL)  //; 有效的回环信息
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }

                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            //? ------------ 增加开始 -----------------
            std::deque<sensor_msgs::Imu::ConstPtr> imu_queue;
            int total_IMU_cnt = 0;
            int acc_IMU_cnt = 0;
            //; 下面就是在统计这些imu数据中在上次lio状态更新时刻之后的那些imu数据
            for (auto &imu_msg : measurement.first)
            {
                total_IMU_cnt++;
                //; 如果这帧IMU数据在上次状态更新之后
                //! g_lio_state.last_update_time : 在lio中对应lidar点云的最后一个点的时间， 在vio中对应图像的时间
                if (imu_msg->header.stamp.toSec() > g_lio_state.last_update_time)
                {
                    acc_IMU_cnt++;
                    imu_queue.push_back(imu_msg);
                }
            }
            StatesGroup state_aft_integration = g_lio_state;  //; 把状态赋值给局部变量：积分后的状态变量


            int esikf_update_valid = false;
            if (imu_queue.size())
            {
                //; 正常情况下这个条件不会满足
                if (g_lio_state.last_update_time == 0)
                {
                    g_lio_state.last_update_time = imu_queue.front()->header.stamp.toSec();
                }
                
                //; start_dt < 0, end_dt < 0
                double start_dt = g_lio_state.last_update_time - imu_queue.front()->header.stamp.toSec();
                double end_dt = cam_update_tim - imu_queue.back()->header.stamp.toSec();  //; 图像时间应该是<最后一个IMU时间的，所以这里也是负数
                esikf_update_valid = true;

                //; 注意：只要历史上收到过一帧lidar数据，m_if_have_lidar_data就会被置位为1。
                if (g_camera_lidar_queue.m_if_have_lidar_data && (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR))
                {
                    //; 如果有激光雷达的数据，那么利用激光雷达算出来的上一次的全局位姿进行积分，得到此时的一个IMU计算的全局位姿
                    *p_imu = *(estimator.m_fast_lio_instance->m_imu_process);
                    //! 这里应该有点问题啊，lidar+imu是前面多取一个IMU；而image+imu是后面多取一个IMU。现在传入image的IMU给lidar的算，
                    //!     恰好错位了一个IMU的位置
                    //; 注意这里，得到的state_aft_intergration是最新的预测位姿，它的last_update_time也更新了，
                    //;  但是传入的g_lio_state是常量！不会改变！
                    state_aft_integration = p_imu->imu_preintegration(g_lio_state, imu_queue, 0, cam_update_tim - imu_queue.back()->header.stamp.toSec());
                    
                    //; 滑窗中状态变量的先验值，注意是全局的位姿
                    //! 重要：这个地方就是利用IMU积分的预测位姿，作为vio的滑窗里LiDAR的先验位姿，这个是在vio的因子图优化中使用的！
                    estimator.m_lio_state_prediction_vec[WINDOW_SIZE] = state_aft_integration;
                    
                    //; vins滑窗中的位姿和lio的全局位姿之间的差值？
                    //; 此时的estimator.Rs[WINDOW_SIZE]是vio中从最新帧的位姿开始，使用IMU积分得到的预测位姿。
                    diff_vins_lio_q = eigen_q(estimator.Rs[WINDOW_SIZE].transpose() * state_aft_integration.rot_end);
                    diff_vins_lio_t = state_aft_integration.pos_end - estimator.Ps[WINDOW_SIZE];
                    
                    //! 此时vio和lio位姿产生了很大的偏移， 就重新优化一下vio系统。
                    //; 这里说的优化就是把滑窗中的最新帧对齐到lio状态上，然后把滑窗中其他帧都对齐到最新帧上
                    if (diff_vins_lio_t.norm() > 1.0)
                    {
                        estimator.refine_vio_system(diff_vins_lio_q, diff_vins_lio_t);
                        diff_vins_lio_q.setIdentity();
                        diff_vins_lio_t.setZero();
                    }
                    
                    //! 直接用IMU计算的先验位姿赋值给全局的位姿，这里跟lio是一样的，因为lio也是用IMU计算的结果作为先验位姿，
                    //; 这里虽然赋值给了最后的全局优化位姿，但是只是暂时的，后面还会进行iekf优化
                    //; 只要IMU频率 > 50HZ的话，一般这里都会满足
                    if ((start_dt > -0.02) &&
                        (fabs(end_dt) < 0.02))
                    {
                        //; 现在全局的位姿被赋值为了IMU积分得到的先验位姿，last_update_time也被更新为了当前帧图像的时间
                        g_lio_state = state_aft_integration;
                        g_lio_state.last_update_time = cam_update_tim;
                    }
                    //; 跑harbor包的过程中，没有发现下面的打印输出，也就是上面的条件一般都是满足的。
                    else
                    {
                        scope_color(ANSI_COLOR_RED_BOLD);
                        cout << "Start time = " << std::setprecision(8) << imu_queue.front()->header.stamp.toSec() - estimator.m_fast_lio_instance->first_lidar_time << endl;
                        cout << "Final time = " << std::setprecision(8) << cam_update_tim - estimator.m_fast_lio_instance->first_lidar_time << endl;
                        cout << "Start dt = " << start_dt << std::setprecision(2) << endl;
                        cout << "Final dt = " << end_dt << std::setprecision(2) << endl;
                        cout << "LiDAR->Image preintegration: " << start_dt << " <--> " << end_dt << endl;
                    }
                }
            }
            //? ------------ 增加结束 -----------------


            // 建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id
            std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            //; 处理图像数据的主函数，但是里面只进行了对图像添加特征点等操作，而把后端优化的操作全都去掉了。
            estimator.processImage(image, img_msg->header);

            //! 下面这些才是重点，基于迭代卡尔曼滤波进行优化
            //; state_prediction还是IMU积分得到的先验位姿，时间戳也是当前帧图像的时间
            StatesGroup state_prediction = state_aft_integration;   
            double mean_reprojection_error = 0.0;
            int minmum_number_of_camera_res = 10;


            //; state_before_esikf 也是IMU预积分的先验位姿
            StatesGroup state_before_esikf = g_lio_state;  //; lio预测状态


            //; 最后一个条件应该是 上一次更新状态的时间 - 视觉初始化时间 > 设置的阈值10s，那么才相信相机的数据，才使用相机的观测约束
            //; 也就是说，只有当vio初始化完成之后，并且lio更新系统状态的时间超过vio初始化的时间10s，才进行vio的滤波
            if (esikf_update_valid && (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) && (g_lio_state.last_update_time - g_camera_lidar_queue.m_visual_init_time > g_camera_lidar_queue.m_lidar_drag_cam_tim))
            {
                estimator.vector2double();  
                //; 先三角化当前帧的特征点
                estimator.f_manager.triangulate(estimator.Ps, estimator.tic, estimator.ric);

                double deltaR = 0, deltaT = 0;
                int flg_EKF_converged = 0;
                Eigen::Matrix<double, DIM_OF_STATES, 1> solution;
                Eigen::Vector3d rot_add, t_add, v_add, bg_add, ba_add, g_add;

                std::vector<Eigen::Vector3d> pts_i, pts_j;
                std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> reppro_err_vec;
                std::vector<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>> J_mat_vec;
                Eigen::Matrix<double, -1, -1> Hsub;
                Eigen::Matrix<double, -1, 1> meas_vec;
                int win_idx = WINDOW_SIZE;

                Eigen::Matrix<double, -1, -1> K;
                Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> K_1;
                //; 迭代两次
                for (int iter_time = 0; iter_time < 2; iter_time++)
                {
                    
                    Eigen::Quaterniond q_pose_last = Eigen::Quaterniond(state_aft_integration.rot_end * diff_vins_lio_q.inverse());
                    Eigen::Vector3d t_pose_last = state_aft_integration.pos_end - diff_vins_lio_t;
                    estimator.m_para_Pose[win_idx][0] = t_pose_last(0);
                    estimator.m_para_Pose[win_idx][1] = t_pose_last(1);
                    estimator.m_para_Pose[win_idx][2] = t_pose_last(2);
                    estimator.m_para_Pose[win_idx][3] = q_pose_last.x();
                    estimator.m_para_Pose[win_idx][4] = q_pose_last.y();
                    estimator.m_para_Pose[win_idx][5] = q_pose_last.z();
                    estimator.m_para_Pose[win_idx][6] = q_pose_last.w();

                    //; 构造残差和雅克比，注意里面只计算了当前帧形成的残差和对当前帧的位姿的雅克比
                    construct_camera_measure(win_idx, estimator, reppro_err_vec, J_mat_vec);

                    //; 构成重投影误差的数量<10 ，这样认为不准确，这里就跳出
                    if (reppro_err_vec.size() < minmum_number_of_camera_res)
                    {
                        cout << "Size of reppro_err_vec: " << reppro_err_vec.size() << endl;
                        break;
                    }

                    // TODO: Add camera residual here
                    Hsub.resize(reppro_err_vec.size() * 2, 6);  //; 雅克比维度，每一个残差都是2x6, 其中前3是旋转，后3是平移
                    meas_vec.resize(reppro_err_vec.size() * 2, 1);
                    K.resize(DIM_OF_STATES, reppro_err_vec.size());
                    int features_correspondences = reppro_err_vec.size();

                    for (int residual_idx = 0; residual_idx < reppro_err_vec.size(); residual_idx++)
                    {
                        //; 注意这里加了负号，因为正常的z_m应该是0，这里用预测观测值计算的是h, 所以带入iekf中z_m - h = -h
                        meas_vec.block(residual_idx * 2, 0, 2, 1) = -1 * reppro_err_vec[residual_idx];
                        Hsub.block(residual_idx * 2, 0, 2, 6) = J_mat_vec[residual_idx];
                    }

                    K_1.setZero();
                    auto Hsub_T = Hsub.transpose();

                    H_T_H.setZero();
                    H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;
                    //; CAM_MEASUREMENT_COV = 0.001，这里的化简与lio中的化简同理，就是把相机的观测方差放到卡尔曼增益K的左侧括号求逆里面去
                    //; 但是这里是乘，lidar里面是除，为啥？这样相差的也太大了吧？
                    K_1 = (H_T_H + (state_aft_integration.cov * CAM_MEASUREMENT_COV).inverse()).inverse();
                    K = K_1.block<DIM_OF_STATES, 6>(0, 0) * Hsub_T;
                    //; state_prediction是IMU预测得到的状态，state_aft_integration是迭代优化得到的状态
                    auto vec = state_prediction - state_aft_integration;
                    solution = K * (meas_vec - Hsub * vec.block<6, 1>(0, 0));
                    
                    mean_reprojection_error = abs(meas_vec.mean());   //; 平均的重投影误差

                    if (std::isnan(solution.sum()))
                    {
                        break;
                    }
                    
                    //! 重要：更新迭代后的状态，并计算状态的变化量
                    state_aft_integration = state_prediction + solution;
                    solution = state_aft_integration - state_prediction;  //; 有毛病？上边加完了，下边再减，这不什么也没干吗？

                    rot_add = (solution).block<3, 1>(0, 0);
                    t_add = solution.block<3, 1>(3, 0);
                    flg_EKF_converged = false;
                    //; 判断迭代是否收敛：这一次在IMU预测状态上的调整量 距离 上一次在IMU预测状态上的调整量 非常接近
                    if (((rot_add.norm() * 57.3 - deltaR) < 0.01) && ((t_add.norm() - deltaT) < 0.015))
                    {
                        flg_EKF_converged = true;
                    }

                    deltaR = rot_add.norm() * 57.3;
                    deltaT = t_add.norm();
                }
                //!  ------------   iekf迭代完成

                
                //; 如果构造残差个数足够，说明进行了iekf，那么就判断迭代结果是否满足要求
                if (reppro_err_vec.size() >= minmum_number_of_camera_res)
                {
                    G.setZero();
                    G.block<DIM_OF_STATES, 6>(0, 0) = K * Hsub;
                    
                    //; 在预测状态上调整的角度 <3度，距离<0.5米，平均重投影误差<1，才认为这次滤波得到的结果满足要求
                    if ((rot_add.norm() * 57.3 < 3) &&
                        (t_add.norm() < 0.5) &&
                        (mean_reprojection_error < 1.0))
                    {
                        //? cc-modified! 
                        // printf("g_lio_state = state_aft_integration" );

                        g_lio_state = state_aft_integration;  //; 赋值给全局的变量
                        eigen_q q_I = eigen_q(1.0, 0, 0, 0);

                        //! 这里又算优化状态和初始状态之间的差距？结果不就是 rot_add 和 t_add 吗？搞毛线？
                        double angular_diff = eigen_q(g_lio_state.rot_end.transpose() * state_before_esikf.rot_end).angularDistance(q_I) * 57.3;
                        double t_diff = (g_lio_state.pos_end - state_before_esikf.pos_end).norm();
                        
                        if ((t_diff > 0.2) || (angular_diff > 2.0))
                        {
                            g_lio_state = state_before_esikf;
                        }
                    }
                }
            }

            // Update state with pose graph optimization
            //! 服了，滤波滤了半天，这里又改成IMU的预测状态了！
            //! 确实有这个问题，看github上的issue ：https://github.com/hku-mars/r2live/issues/30
            //; vio初始化之后的10s时间内，g_lio_state都是由imu积分得到的预测状态
            g_lio_state = state_before_esikf;  

            t_s.tic();
            //; 这里又调用VINS的后端进行优化，但是作者自己写的LM算法进行后端优化，没有使用ceres的库
            //; 这个函数里面就是作者把原来vins的processImage函数中的绝大多数操作又拿来了！然后后端自己写了LM算法，没用ceres
            estimator.solve_image_pose(img_msg->header);

            //! 没看懂在干什么
            //; 首先只要收到过lidar的数据，那么这个条件一定满足
            if (g_camera_lidar_queue.m_if_have_lidar_data)
            {
                //; 在非初始化阶段，前面条件都满足。 而后面的条件就是判断这次的IMU数据时间是否 > 上次状态更新时间，一般也满足。
                if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && esikf_update_valid)
                {
                    //; vio初始化的最后会把这个时间设置为3e88，然后这里判断如果是3e88的话，那么就设置成当前帧图像的时间
                    if (g_camera_lidar_queue.m_visual_init_time == 3e88)
                    {
                        scope_color(ANSI_COLOR_RED_BOLD);
                        printf("G_camera_lidar_queue.m_visual_init_time = %2.f \r\n", cam_update_tim);
                        g_camera_lidar_queue.m_visual_init_time = cam_update_tim;
                    }

                    //; 上次状态更新的时间 - VIO初始化的时间 > 10秒
                    if ((g_lio_state.last_update_time - g_camera_lidar_queue.m_visual_init_time > g_camera_lidar_queue.m_lidar_drag_cam_tim))
                    {
                        //! ??? 这个变量前面不是已经定义了吗？
                        StatesGroup state_before_esikf = g_lio_state;

                        //; 判断是否赋值ba
                        if (estimator.Bas[WINDOW_SIZE].norm() < 0.5)
                        {
                            g_lio_state.bias_a = estimator.Bas[WINDOW_SIZE];
                        }
                        
                        //; bg是相信的，直接赋值
                        g_lio_state.bias_g = estimator.Bgs[WINDOW_SIZE];
                        //; 还是相信LIO的旋转，这里把估计的速度乘以旋转差值
                        g_lio_state.vel_end = diff_vins_lio_q.toRotationMatrix() * estimator.Vs[WINDOW_SIZE];
                        //! iekf过程中没有给 state_aft_integration.cov 赋值！
                        g_lio_state.cov = state_aft_integration.cov; 

                        //; diff_vins_lio_q = eigen_q(estimator.Rs[WINDOW_SIZE].transpose() * state_aft_integration.rot_end);
                        //; diff_vins_lio_t = state_aft_integration.pos_end - estimator.Ps[WINDOW_SIZE];
                        Eigen::Matrix3d temp_R = estimator.Rs[WINDOW_SIZE] * diff_vins_lio_q.toRotationMatrix();
                        Eigen::Vector3d temp_T = estimator.Ps[WINDOW_SIZE] + diff_vins_lio_t;
                    
                        eigen_q q_I = eigen_q(1.0, 0, 0, 0);
                        double angular_diff = eigen_q(temp_R.transpose() * state_before_esikf.rot_end).angularDistance(q_I) * 57.3;
                        //; t_diff结果就等于 estimator.Ps[WINDOW_SIZE] - estimator.Ps[WINDOW_SIZE];
                        //;   也就是后端优化之前和之后的位置变化
                        double t_diff = (temp_T - state_before_esikf.pos_end).norm();
                        
                        if ((t_diff < 0.2) && (angular_diff < 2.0))
                        {
                            //! 这个协方差的赋值实在没有看懂，这里用的应该还是vio使用iekf得到的协方差矩阵？那优化之后的协方差矩阵呢？
                            g_lio_state.cov = state_aft_integration.cov;
                            g_lio_state.last_update_time = cam_update_tim;
                            g_lio_state.rot_end = temp_R;
                            g_lio_state.pos_end = temp_T;
     
                            //? cc-modified 
                            //; 经过下面的测试，基本上每次都会更新这里vio姿态到lio姿态上去
                            //; 也就是说整个r2live系统还是以lio系统的位姿为准？
                            //;  那么如果在lio系统没有失效的时候，vio系统还有什么用呢？
                            //! 解答：其实这里就是在用vio去修正lio！
                            // printf("%d  ,  lio_state <<< vio_state \n", cc_num++);
                        }

                        unlock_lio(estimator);
                    }
                }
            }

            //; 这里又来判断，如果平移>0.1, 角度>0.1，就对齐vio到lio上
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                m_state.lock();
                if ((diff_vins_lio_t.norm() > 0.1) &&
                    (diff_vins_lio_q.angularDistance(eigen_q::Identity()) * 57.3 > 0.1))
                {
                    estimator.refine_vio_system(diff_vins_lio_q, diff_vins_lio_t);
                    diff_vins_lio_q.setIdentity();
                    diff_vins_lio_t.setZero();
                }
                m_state.unlock();
            }

            unlock_lio(estimator);
            m_state.lock();


            //; 结束，发布一些消息
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            header = img_msg->header;
            header.frame_id = "world";

            //; 一般不会执行这个
            //; 如果没有lidar消息，就只能发送纯视觉vio的消息
            if (g_camera_lidar_queue.m_if_have_lidar_data == false)
            {
                pubOdometry(estimator, header); // "odometry" 里程计信息
            }

            //; 如果有LiDAR消息，就发布LiDAR里程计信息
            else
            {
                //! 注意，这里发送的是滤波的结果！
                pub_LiDAR_Odometry(estimator, state_aft_integration, header); // "lidar_odometry" 雷达里程计信息
            }

            // 给RVIZ发送topic
            pubCameraPose(estimator, header); // "camera_pose" 相机位姿
            pubKeyPoses(estimator, header);   // "key_poses" 关键点三维坐标
            pubPointCloud(estimator, header); // "history_cloud" 点云信息
            pubTF(estimator, header);         // "extrinsic" 相机到IMU的外参
            pubKeyframe(estimator);           // "keyframe_point"、"keyframe_pose" 关键帧位姿和点云
            m_state.unlock();
            if (relo_msg != NULL)
            {
                pubRelocalization(estimator); // "relo_relative_pose" 重定位位姿
            }
        }

        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        //; 如果当前后端求解器已经进入了非线性优化阶段（即初始化已经完成），那么还要更新一下本文件中的全局位姿变量
        //;    主要是用于基于滑出中最新帧的位姿完全依靠IMU的当前最新位姿推算，就是为了得到一个比较高频的位姿信息
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            update(); // 更新IMU参数[P,Q,V,ba,bg,a,g]
        }

        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    // ROS初始化，设置句柄n
    //; 注意这里的vins_estimator在运行的时候被launch文件改成了r2live，
    //; 所以实际上这个节点才是真正运行的r2live后端节点
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // 读取参数，设置估计器参数
    readParameters(nh);
    estimator.setParameter();

    //?-------- 增加开始 ----------
    get_ros_parameter(nh, "/lidar_drag_cam_tim", g_camera_lidar_queue.m_lidar_drag_cam_tim, 1.0);  //; 配置文件中是10
    get_ros_parameter(nh, "/acc_mul_G", g_camera_lidar_queue.m_if_acc_mul_G, 0);  //; 加速度是否以G为单位进行了归一化
    //; 是否雷达先启动的标志，在配置文件中写的是1
    get_ros_parameter(nh, "/if_lidar_start_first", g_camera_lidar_queue.m_if_lidar_start_first, 1.0);
    get_ros_parameter<int>(nh, "/if_write_to_bag", g_camera_lidar_queue.m_if_write_res_to_bag, false);
    get_ros_parameter<int>(nh, "/if_dump_log", g_camera_lidar_queue.m_if_dump_log, 0);
    get_ros_parameter<std::string>(nh, "/record_bag_name", g_camera_lidar_queue.m_bag_file_name, "./");
    if (g_camera_lidar_queue.m_if_write_res_to_bag)
    {
        // 初始化写入rosbag包
        g_camera_lidar_queue.init_rosbag_for_recording();
    }
    // ANCHOR - Start lio process
    //; 锚 —— 开启LIO线程
    g_camera_lidar_queue.m_if_lidar_can_start = false;  //; 首先置位雷达不能启动标志
    if (estimator.m_fast_lio_instance == nullptr)
    {
        estimator.m_fast_lio_instance = new Fast_lio();   //; 在类构造函数中就开启了LIO线程
    }
    //?-------- 增加结束 ----------

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // 用于RVIZ显示的Topic
    registerPub(nh);

    // 订阅IMU、feature、restart、match_points的topic,执行各自回调函数
    ros::Subscriber sub_imu = nh.subscribe(IMU_TOPIC, 20000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = nh.subscribe("/feature_tracker/feature", 20000, feature_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_restart = nh.subscribe("/feature_tracker/restart", 20000, restart_callback, ros::TransportHints().tcpNoDelay());
    
    //; 看这个消息是干嘛的？好像根本没有这个话题，应该是从vins里复制来忘了删
    ros::Subscriber sub_relo_points = nh.subscribe("/pose_graph/match_points", 20000, relocalization_callback, ros::TransportHints().tcpNoDelay());

    // 创建VIO主线程
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
