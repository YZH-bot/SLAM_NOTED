#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

// info: imu 初始化, 只会执行一次: 初始化了重力, 陀螺仪bias, 加速度计和角速度的协方差.
// doc: IMU静止，加速度偏移(bias_a)和重力耦合，肯定是初始化不了的。必须给予足够的激励
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();  //重置参数
    N = 1;    //将迭代次数置1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;  //从common_lib.h中拿到imu初始时刻的加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;     //从common_lib.h中拿到imu初始时刻的角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;                  //加速度测量作为初始化均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;                  //角速度测量作为初始化均值
    first_lidar_time = meas.lidar_beg_time;                       //将当期imu帧对应的lidar时间作为初始时间
  }

  // doc: 计算方差
  for (const auto &imu : meas.imu)  // doc: 拿到所有的imu帧
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    // doc: 根据当前帧和均值差作为均值的更新
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    //.cwiseProduct()对应系数相乘
    //每次迭代之后均值都会发生变化，最后的方差公式中减的应该是最后的均值
    // ?: https://zhuanlan.zhihu.com/p/445729443 方差迭代计算公式
    //按照博客推导出来的下面方差递推公式有两种
    //第一种是
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);
    //第二种是
    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - 上一次的mean_acc)  / N;
    // cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - 上一次的mean_gyr)  / N;
    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  // doc: 在esekfom.hpp获得x_的状态
  state_ikfom init_state = kf_state.get_x();
  // ? https://zhuanlan.zhihu.com/p/437872881
  // doc: 从common_lib.h中拿到重力，并与加速度测量均值的单位重力求出SO2的旋转矩阵类型的重力加速度
  // info: 这里需要静止初始化, 估算出初始重力的方向
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;                  // doc: 角速度测量作为陀螺仪bias
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;  // doc: 将lidar和imu外参位移量传入
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;  // doc: 将lidar和imu外参旋转量传入
  kf_state.change_x(init_state);              // doc: 更新状态x_

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();  // doc: 获取P_协方差矩阵
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;         // doc: 将协方差矩阵的位置和旋转的协方差置为0.00001
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;     // doc: 将协方差矩阵的速度和位姿的协方差置为0.00001
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;    // doc: 将协方差矩阵的重力和姿态的协方差置为0.0001
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;     // doc: 将协方差矩阵的陀螺仪偏差和姿态的协方差置为0.001
  init_P(21,21) = init_P(22,22) = 0.00001;                   // doc: 将协方差矩阵的lidar和imu外参位移量的协方差置为0.00001
  kf_state.change_P(init_P);                                 // doc: 更新P_
  last_imu_ = meas.imu.back();                               // doc: 将最后一帧的imu数据传入last_imu_中，在UndistortPcl使用到了

}

// info: 正向传播 反向传播 去畸变，这里涉及到了Lidar的去畸变问题
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);                                        // doc: 将上一帧最后尾部的imu添加到当前帧头部的imu
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  // doc: 根据点云中每个点的时间戳对点云进行重排序
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  // doc: 获取上一次KF估计的后验状态作为本次IMU预测的初始状态
  state_ikfom imu_state = kf_state.get_x(); 
  IMUpose.clear();
  // doc: 将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  // doc: 前向传播对应的参数
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  // doc: imu旋转矩阵
  M3D R_imu;

  double dt = 0;

  // doc: eksf 用于运动方程进行状态递推的参数
  input_ikfom in;
  // doc: 遍历本次估计的所有IMU测量并且进行积分，离散中值法 前向传播
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    // doc: 判断时间先后顺序 不符合直接continue
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    // doc: 中值积分
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    // doc: 通过重力数值对加速度进行一下微调, 可以认为这里是乘以 imu 的scale参数
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    // doc: 如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    // doc: 原始测量的中值作为更新
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    // doc: 初始化的时候估计的协方差
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    // doc: IMU前向传播，每次传播的时间间隔为dt
    kf_state.predict(dt, Q, in);  // ? 看看 bias 怎么更新的

    /* save the poses at each IMU measurements */
    // doc: 保存IMU预测过程的状态
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;                  // doc: 减去 bias 的影响
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);   // doc: 减去 bias 的影响, 并转到world坐标系下
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];   // doc: 转到world坐标系之后, 去除重力向量, 得到物体加速度
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;  // doc: 后一个IMU时刻距离此次雷达开始的时间间隔
    // doc: 保存IMU预测过程的状态
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // doc: 把最后一帧IMU测量和lidar之间的时间间隔给补上
  // doc: 判断雷达结束时间是否晚于IMU，最后一个IMU时刻可能早于雷达末尾 也可能晚于雷达末尾
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();         // doc: 每次更新imu测量之后, 都要更新状态x，以便于后面使用
  last_imu_ = meas.imu.back();          // doc: 保存最后一个IMU测量，以便于下一次使用
  last_lidar_end_time_ = pcl_end_time;  // doc: 保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

  /*** undistort each lidar point (backward propagation) ***/
  /*** //doc: 在处理完所有的IMU预测后，剩下的就是对激光的去畸变了 ***/
  // doc: 基于IMU预测对lidar点云去畸变
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot); // doc: 拿到前一帧的IMU旋转矩阵
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    // doc: 点云时间需要迟于前一个IMU时刻 因为是在两个IMU时刻之间去畸变，此时默认雷达的时间戳在head IMU时刻之前
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));     // doc: 点所在时刻的旋转, world下
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);                                     // doc: 点所在时刻的位置(雷达坐标系下)
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);   // doc: 从点所在的世界位置 - 雷达末尾世界位置
      // doc: .conjugate()取旋转矩阵的共轭,rot.conjugate（）是四元数共轭，即旋转求逆
      // doc: imu_state.offset_R_L_I是从雷达到惯性的旋转矩阵 简单记为I^R_L
      // doc: imu_state.offset_T_L_I是惯性系下雷达坐标系原点的位置简单记为I^t_L
      // doc: (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) 把 point 转换到 i 时刻的world坐标系下,
      // doc: 然后又投影到最后一个时刻的lidar坐标系下, 这里好像少了加上 平移 的一步, 即: imu_state.rot.conjugate() * imu_state.pos
      // info: 更正: imu_state.rot.conjugate() * imu_state.pos这一步已经包含在 T_ei 中
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  // doc: 拿到的当前帧的imu测量为空，则直接返回
  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  // doc: 需要对 imu 进行初始化, 初始化完成之后不再进入 if
  if (imu_need_init_)
  {
    /// The very first lidar frame
    // doc: 只会执行一次
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    // doc: init_iter_num 大于阈值之后代表 imu 初始化完成
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  // doc: IMU初始化完之后才会进行点云的畸变校正
  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
