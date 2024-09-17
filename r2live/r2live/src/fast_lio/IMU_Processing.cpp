#include "IMU_Processing.hpp"

#define COV_OMEGA_NOISE_DIAG 1e-2     // n_g
#define COV_ACC_NOISE_DIAG 1e-5       // n_a
#define COV_GYRO_NOISE_DIAG 1e-5      //! 疑问：玄学参数，这个很奇怪
#define COV_BIAS_ACC_NOISE_DIAG 0.2   // n_ba
#define COV_BIAS_GYRO_NOISE_DIAG 0.2  // n_bg

#define COV_START_ACC_DIAG 1e-10
#define COV_START_GYRO_DIAG 1e-10

double g_lidar_star_tim = 0;
ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), last_imu_(nullptr), start_timestamp_(-1)
{
  Eigen::Quaterniond q(0, 1, 0, 0);
  Eigen::Vector3d t(0, 0, 0);
  init_iter_num = 1;
  cov_acc = Eigen::Vector3d(COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG);
  cov_gyr = Eigen::Vector3d(COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG);
  mean_acc = Eigen::Vector3d(0, 0, -9.805);
  mean_gyr = Eigen::Vector3d(0, 0, 0);
  angvel_last = Zero3d;
  cov_proc_noise = Eigen::Matrix<double, DIM_OF_PROC_N, 1>::Zero();
}

ImuProcess::~ImuProcess()
{
}

void ImuProcess::Reset()
{
  ROS_WARN("Reset ImuProcess");
  angvel_last = Zero3d;
  cov_proc_noise = Eigen::Matrix<double, DIM_OF_PROC_N, 1>::Zero();

  cov_acc = Eigen::Vector3d(COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG);
  cov_gyr = Eigen::Vector3d(COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG);
  mean_acc = Eigen::Vector3d(0, 0, -9.805);
  mean_gyr = Eigen::Vector3d(0, 0, 0);

  imu_need_init_ = true;
  b_first_frame_ = true;
  init_iter_num = 1;

  last_imu_ = nullptr;

  start_timestamp_ = -1;
  v_imu_.clear();
  IMU_pose.clear();

  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::IMU_Initial(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  Eigen::Vector3d cur_acc, cur_gyr;

  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    //; 在上面的Reset函数中赋值了：
    //;  mean_acc = Eigen::Vector3d(0, 0, -9.805);  mean_gyr = Eigen::Vector3d(0, 0, 0);
    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    // cwiseProduct：输出相同位置的两个矩阵中各个系数的乘积所组成的矩阵
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N++;
  }

  // TODO: fix the cov
  cov_acc = Eigen::Vector3d(COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG);
  cov_gyr = Eigen::Vector3d(COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG);
  state_inout.gravity = Eigen::Vector3d(0, 0, 9.805);
  state_inout.rot_end = Eye3d;
  state_inout.bias_g = mean_gyr; 
  //! 问题：为什么没有给bias_a赋值？
}

void ImuProcess::lic_state_propagate(const MeasureGroup &meas, StatesGroup &state_inout)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;

  /*** sort point clouds by offset time ***/
  PointCloudXYZI pcl_out = *(meas.lidar);
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  double end_pose_dt = pcl_end_time - imu_end_time;   //; 这应该是一个正数？

  //; 这里是积分，不是预积分
  state_inout = imu_preintegration(state_inout, v_imu, 1, end_pose_dt);
  last_imu_ = meas.imu.back();
}

// Avoid abnormal state input
bool check_state(StatesGroup &state_inout)
{
  bool is_fail = false;
  for (int idx = 0; idx < 3; idx++)
  {
    if (fabs(state_inout.vel_end(idx)) > 10)
    {
      is_fail = true;
      scope_color(ANSI_COLOR_RED_BG);
      for (int i = 0; i < 10; i++)
      {
        cout << __FILE__ << ", " << __LINE__ << ", check_state fail !!!! " << state_inout.vel_end.transpose() << endl;
      }

      state_inout.vel_end(idx) = 0.0;
    }
  }

  return is_fail;
}

// Avoid abnormal state input
void check_in_out_state(const StatesGroup &state_in, StatesGroup &state_inout)
{
  if ((state_in.pos_end - state_inout.pos_end).norm() > 1.0)
  {
    scope_color(ANSI_COLOR_RED_BG);
    for (int i = 0; i < 10; i++)
    {
      cout << __FILE__ << ", " << __LINE__ << ", check_in_out_state fail !!!! " << state_in.pos_end.transpose() << " | "
           << state_inout.pos_end.transpose() << endl;
    }

    state_inout.pos_end = state_in.pos_end;
  }
}

std::mutex g_imu_premutex;

StatesGroup ImuProcess::imu_preintegration_fast_lio(const StatesGroup &state_in,
                                                    std::deque<sensor_msgs::Imu::ConstPtr> &v_imu,
                                                    int if_multiply_g, double end_pose_dt)
{
  std::unique_lock<std::mutex> lock(g_imu_premutex);
  StatesGroup state_inout = state_in;
  if (check_state(state_inout))
  {
    state_inout.display(state_inout, "state_inout");
    state_in.display(state_in, "state_in");
  }

  Eigen::Vector3d acc_imu(0, 0, 0), angvel_avr(0, 0, 0), acc_avr(0, 0, 0), vel_imu(0, 0, 0), pos_imu(0, 0, 0);
  vel_imu = state_inout.vel_end;
  pos_imu = state_inout.pos_end;
  Eigen::Matrix3d R_imu(state_inout.rot_end);
  Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
  Eigen::MatrixXd cov_w(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Zero());
  double dt = 0;
  int if_first_imu = 1;

  Eigen::Vector3d vel_imu_0 = state_inout.vel_end;
  Eigen::Vector3d pos_imu_0 = state_inout.pos_end;
  Eigen::Quaterniond q_imu_0 = Eigen::Quaterniond(state_inout.rot_end).normalized();
  Eigen::Vector3d ba_0 = state_inout.bias_a;
  Eigen::Vector3d bg_0 = state_inout.bias_g;
  Eigen::Vector3d acc_1, gyro_1;
  Eigen::Vector3d acc_r, gyro_r;

  IntegrationBase *imu_integration = nullptr;
  for (auto it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    angvel_avr -= state_inout.bias_g;

    acc_avr = acc_avr - state_inout.bias_a;

    if (tail->header.stamp.toSec() < state_inout.last_update_time)
    {
      continue;
    }

    if (if_first_imu)
    {
      if_first_imu = 0;
      dt = tail->header.stamp.toSec() - state_inout.last_update_time;
      acc_imu = R_imu * acc_avr - state_inout.gravity;

      imu_integration = new IntegrationBase(acc_imu, angvel_avr, ba_0, bg_0);
      imu_integration->set_initial_val(q_imu_0, pos_imu_0, vel_imu_0);
      imu_integration->set_cov(state_inout.cov);
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    if (dt > 0.05)
    {
      scope_color(ANSI_COLOR_RED_BOLD);
      for (int i = 0; i < 1; i++)
      {
        cout << __FILE__ << ", " << __LINE__ << ", dt = " << dt << endl;
      }

      dt = 0.05;
    }

    /* covariance propagation */
    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);
    Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();

    F_x.block<3, 3>(0, 0) = Exp_f.transpose();
    F_x.block<3, 3>(0, 9) = -Jr_omega_dt * dt;
    F_x.block<3, 3>(3, 3) = Eye3d; // Already the identity.
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
    F_x.block<3, 3>(6, 12) = -R_imu * dt;
    F_x.block<3, 3>(6, 15) = Eye3d * dt;

    Eigen::Matrix3d cov_acc_diag, cov_gyr_diag, cov_omega_diag;
    cov_omega_diag = Eigen::Vector3d(COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG).asDiagonal();
    cov_acc_diag = Eigen::Vector3d(COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG).asDiagonal();
    cov_gyr_diag = Eigen::Vector3d(COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG).asDiagonal();

    cov_w.block<3, 3>(0, 0) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;
    cov_w.block<3, 3>(3, 3) = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt;
    cov_w.block<3, 3>(6, 6) = cov_acc_diag * dt * dt;
    cov_w.block<3, 3>(9, 9).diagonal() = Eigen::Vector3d(COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG) * dt * dt; // bias gyro covariance
    cov_w.block<3, 3>(12, 12).diagonal() = Eigen::Vector3d(COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG) * dt * dt;  // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    R_imu = R_imu * Exp_f;
    acc_imu = R_imu * acc_avr - state_inout.gravity;
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    vel_imu = vel_imu + acc_imu * dt;
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;

    imu_integration->push_back(dt, acc_avr, angvel_avr);
  }

  /* calculated the pos and attitude prediction at the frame-end */
  dt = end_pose_dt;
  imu_integration->push_back(dt, acc_avr, angvel_avr);
  state_inout.last_update_time = v_imu.back()->header.stamp.toSec() + dt;
  if (dt > 0.1)
  {
    scope_color(ANSI_COLOR_RED_BOLD);
    for (int i = 0; i < 1; i++)
    {
      cout << __FILE__ << ", " << __LINE__ << "dt = " << dt << endl;
    }

    dt = 0.1;
  }

  state_inout.vel_end = vel_imu + acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(angvel_avr, dt);
  state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

  cout << "Vins_pre = " << imu_integration->delta_v.transpose() << " | Lio_pre = " << state_inout.vel_end.transpose()
       << ", diff = " << (imu_integration->delta_v - state_inout.vel_end).norm() << endl;

  state_inout.vel_end = imu_integration->delta_v;
  state_inout.rot_end = imu_integration->delta_q.toRotationMatrix();
  state_inout.pos_end = imu_integration->delta_p;

  if (0)
  {
    if (check_state(state_inout))
    {
      std::cout << __FILE__ << " " << __LINE__ << std::endl;
      state_inout.display(state_inout, "state_inout");
      state_in.display(state_in, "state_in");
    }

    check_in_out_state(state_in, state_inout);
  }

  return state_inout;
}

//; IMU 状态传播方程
StatesGroup ImuProcess::imu_preintegration(const StatesGroup &state_in,
                                           std::deque<sensor_msgs::Imu::ConstPtr> &v_imu,
                                           int if_multiply_g, double end_pose_dt)
{
  std::unique_lock<std::mutex> lock(g_imu_premutex);
  StatesGroup state_inout = state_in;
  //; 这里检查就是检查速度是否过大
  if (check_state(state_inout))
  {
    state_inout.display(state_inout, "state_inout");
    state_in.display(state_in, "state_in");
  }

  Eigen::Vector3d acc_imu(0, 0, 0), angvel_avr(0, 0, 0), acc_avr(0, 0, 0), vel_imu(0, 0, 0), pos_imu(0, 0, 0);
  vel_imu = state_inout.vel_end;
  pos_imu = state_inout.pos_end;
  Eigen::Matrix3d R_imu(state_inout.rot_end);
  Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
  Eigen::MatrixXd cov_w(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Zero());
  double dt = 0;
  int if_first_imu = 1;

  
  //; 遍历所有的IMU数据， 注意end指向deque的最后一个元素的下一个位置，因此这里有效元素只到end-1
  for (std::deque<sensor_msgs::Imu::ConstPtr>::iterator it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++)
  {
    sensor_msgs::Imu::ConstPtr head = *(it_imu);
    sensor_msgs::Imu::ConstPtr tail = *(it_imu + 1);

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    //; 去零偏后的中值acc和gyro
    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr - state_inout.bias_a;

    // doc: 非常非常重要：状态就是在这里同步的！上次vio的状态更新到了lio上，所以这次lio状态传播的时候，在last_update_time 之前的IMU数据就不能用了
    if (tail->header.stamp.toSec() < state_inout.last_update_time)
    {
      continue;
    }

    //; 这一帧中的第一个IMU数据是上一帧的最后一个IMU数据，不准确，
    if (if_first_imu)
    {
      if_first_imu = 0;
      dt = tail->header.stamp.toSec() - state_inout.last_update_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    //; 时间太长，频率 < 20HZ, IMU数据肯定有问题
    if (dt > 0.05)
    {
      scope_color(ANSI_COLOR_RED_BOLD);
      for (int i = 0; i < 1; i++)
      {
        cout << __FILE__ << ", " << __LINE__ << ", dt = " << dt << endl;
      }

      dt = 0.05;
    }

    /* covariance propagation */
    //! 下面首先计算协方差。注意对比一下vins，这是计算误差的协方差吗？
    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);
    //; 这个地方十四讲讲过，以为它非常接近单位阵，所以可以直接用单位阵来近似
    Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();  
     
    //; 下面的计算跟论文都能对上。除了最后一个状态变量是g，论文中没有g
    //; 注意：下面这个跟fast-lio是能够对上的，状态变量的定义和fast-lio中的定义一样
    //! 问题：下面的F的计算和fast-lio一样，但是论文中说的是状态误差，但是这里IMU积分是对状态积分，他们雅克比能一样吗？
    //! 解答：是一样的，看自己写的笔记
    //; 注意：这里F_x是18x18的，但是并没有所有地方都赋值，应该是因为初值设置成了I
    F_x.block<3, 3>(0, 0) = Exp_f.transpose();  //; 为什么要转置？因为公式中是exp(-wt)，这里算的是wt，所以转置就相当取逆
    F_x.block<3, 3>(0, 9) = -Jr_omega_dt * dt;
    // F_x.block<3, 3>(3, 3) = Eye3d; // Already the identity.  //; CC注释掉了，对比论文看有几个不是I，然后赋值几个，这样好看点。
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
    F_x.block<3, 3>(6, 12) = -R_imu * dt;
    F_x.block<3, 3>(6, 15) = Eye3d * dt;  //; fast-lio论文中有公式，但是没有推到

    //; 过程噪声，也就是本次的测量噪声
    //!  这里啥情况啊？代码和公式对不上！！！
    //! 解答：这里直接计算了W*cov_w*W^T了，噪声顺序是n_g, n_a, n_bg, n_ba。里面有的数据符号不对，
    //!  但是因为三个矩阵都是对角阵，相乘W矩阵里面的符号就被消掉了
    //!     这里跟r2live论文对不上，突然想起来这个是和fast-lio论文对应的。
    Eigen::Matrix3d cov_acc_diag, cov_gyr_diag, cov_omega_diag;
    // cov_omega_diag就是n_g
    cov_omega_diag = Eigen::Vector3d(COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG).asDiagonal();
    cov_acc_diag = Eigen::Vector3d(COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG).asDiagonal();
    cov_gyr_diag = Eigen::Vector3d(COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG).asDiagonal();

    //! 18 x 18，最后3x3是重力加速度，直接就是0. 下面都是直接算了F_w * n * F_w.T
    // 1.正确：旋转R部分的过程噪声，但是好像没有转置啊？（靠，这里直接用单位阵近似了，转置不转置结果一样）
    cov_w.block<3, 3>(0, 0) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;
    //! 2.错误：平移P部分的位置，按照公式推应该是0. 右边对应公式是cov_gyr_diag * dt * dt。但是结果很小，不知道有没有影响
    cov_w.block<3, 3>(3, 3) = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt;
    // 3.正确：(6,6)的位置应该是速度部分，结果应该是 R_imu * cov_acc_diag * R_imu.transpose() * dt * dt;
    //   然后这里其实是做了简化，因为cov_acc_diag就是一个COV_ACC_NOISE_DIAG*I_33，因此两边的旋转矩阵就抵消了
    //   结果就是cov_acc_diag * dt * dt了，所以说速度这部分没有问题
    cov_w.block<3, 3>(6, 6) = cov_acc_diag * dt * dt;
    // 4.正确：最后两个位置对应bg ba的更新，没有问题
    cov_w.block<3, 3>(9, 9).diagonal() = Eigen::Vector3d(COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG) * dt * dt; // bias gyro covariance
    cov_w.block<3, 3>(12, 12).diagonal() = Eigen::Vector3d(COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG) * dt * dt;  // bias acc covariance
  
    //; 状态变量协方差的维度，18x18
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    //! 这里并不是预积分，仍然是IMU积分，仍然是根据上一次的优化位姿，然后用IMU积分计算此次的P V Q，acc gyro都没算
    R_imu = R_imu * Exp_f;
    acc_imu = R_imu * acc_avr - state_inout.gravity;
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    vel_imu = vel_imu + acc_imu * dt;
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
  }

  dt = end_pose_dt;  //; 最后lidar点和最后IMU数据之间的时间差，正数

  //; 上次更新时间更新为最后的lidar时间
  state_inout.last_update_time = v_imu.back()->header.stamp.toSec() + dt;

  if (dt > 0.1)
  {
    scope_color(ANSI_COLOR_RED_BOLD);
    for (int i = 0; i < 1; i++)
    {
      cout << __FILE__ << ", " << __LINE__ << "dt = " << dt << endl;
    }

    dt = 0.1;
  }

  //; 最后一帧IMU没有计算，因为它时间戳和最后一个lidar不对齐，因此这里要单独计算一小段
  //; 注意这里状态方程只计算了Q P V三个，所以状态方程是9维的。而状态变量分别是Q P V bg ba g, 一共18维                                                                                                                                                                                        
  state_inout.vel_end = vel_imu + acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(angvel_avr, dt);
  state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

  if (0)
  {
    if (check_state(state_inout))
    {
      std::cout << __FILE__ << " " << __LINE__ << std::endl;
      state_inout.display(state_inout, "state_inout");
      state_in.display(state_in, "state_in");
    }

    check_in_out_state(state_in, state_inout);
  }

  return state_inout;
}

//; 点云补偿：把这一帧lidar的点云全部对齐到最后一个点云的坐标系下，即进行了去畸变。但是注意去畸变后的点
//;          都是在最后一个点云的坐标系下的
void ImuProcess::lic_point_cloud_undistort(const MeasureGroup &meas, const StatesGroup &_state_inout, PointCloudXYZI &pcl_out)
{
  /*
    IMU预积分
    把堆积的IMU数据进行预积分处理，推导出每个IMU数据时刻相对于全局坐标系的T，以及对应的方差。
    每次开始之前，都会把上一次预积分的最后一个IMU加入到本次IMU队列的头部，以及对应的姿态结果作为此次预积分的开始姿态。
    这种推理会导致IMU推理的轨迹飞的很快，但是不影响我们总的结果，因为我们取的是短时间内的相对运动姿态。
    这里IMU的预积分使用中值积分。
  */

  /* 数据准备 */

  /*** add the imu of the last frame-tail to the of current frame-head ***/
  StatesGroup state_inout = _state_inout;
  auto v_imu = meas.imu;
  //; 上一次的IMU时间放到本次IMU序列开头
  v_imu.push_front(last_imu_); // 让IMU的时间能包住Lidar的时间
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  //; 点云按先后时间排序
  //; end_time为啥不用meas.lidar_end_time？下面的计算和lidar_end_time应该是一样的
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);

  /*** Initialize IMU pose ***/
  //; 可以看到，每次执行后面的积分之前，都会把IMU_pose清空，因此这个IMU_pose虽然是类成员变量，但是作用和局部变量类似
  IMU_pose.clear();
  //; 先插入第一帧IMu位姿，时间偏移0，然后用上一次状态变量的PVQ和acc gyro
  IMU_pose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));

  /* 预积分过程 */
  //! 里面做的好像不是预积分，而是IMU积分？
  //; 回答：看完了后面的函数，这个确实不是预积分，而完全是IMU积分。也就是根据上一次lidar的位姿，然后利用这一帧lidar的
  //;      所有IMU数据积分，得到各个IMU时刻的body系在world系下的坐标。然后把i时刻的点云，先转到world系下，在转到最后一帧lidar系下
  /*** forward propagation at each imu point ***/
  Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  Eigen::Matrix3d R_imu(state_inout.rot_end);
  Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
  Eigen::MatrixXd cov_w(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Zero());
  double dt = 0;
  //; 遍历所有的imu数据
  for (auto it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++)
  {
    // 中值积分
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    //; 去掉bias的acc 和 gyro中值
    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr - state_inout.bias_a;

    dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    
    /* covariance propagation */

    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

    /* propagation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr - state_inout.gravity;  // 这里旋转还没有更新

    /* propagation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    //; 这一帧IMU数据计算出来的位置Pose6d相对lidar开始帧的时间偏移，
    IMU_pose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  // PCL的最后一个点的时间和IMU的最后一个时间往往不是精准对齐的，这里根据最后一个IMU的姿态，
  // 计算了最后一个点云的姿态，这个是我们其他所有其他时刻的点云要对准的坐标系

  /*** calculated the pos and attitude prediction at the frame-end ***/
  dt = pcl_end_time - imu_end_time;
  state_inout.vel_end = vel_imu + acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(angvel_avr, dt);
  state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

  /*
    点云补偿
    基本过程：把点云的POSE倒序处理，每两个为一对，在队列前面叫head，下一个叫tail，对于点云数据，也是倒序处理，找大于head时间戳的点来处理，点云处理的位置用一个变量来维护。
    详细过程：首先将PCL的指针it_pcl指向最后一个点云的位置，取出imu pose中最后面的两个pose，时间小的叫head，时间大的叫tail，
    以it_pcl为起点，倒序处理每个点，对于某个点，假设对应的时刻为i，判断i是否大于head的时间，如果大于，则以这个head对应的姿态为起点、pcl点与head的时间差为dt，计算head到i时刻的姿态。
    然后对比最后一个点云（结束时刻）的姿态，计算出i时刻的这个点，在结束时刻的lidar坐标系下的坐标。
    P_at_lidar_i=Pi; // i时刻lidar坐标系下的点

    P_at_imu_i=TblP_at_lidar_i; // i时刻lidar点在imu坐标系的坐标
    P_at_world_i=TwiP_at_imu_i; // 转换到world坐标系
    P_at_imu_e=Twe.inverse()*P_at_world_i; // 转换到结束帧时刻的imu坐标
    P_at_lidar_e=Tbl.inverse()*P_at_imu_e; // 有imu再转到lidar坐标
  */
  //; T_wl = T_wi * Til
  Eigen::Vector3d pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lidar_offset_to_IMU;
  // pos_liD_e的计算假设是imu与lidar的坐标系之间没有旋转变换，只有平移量，这个是livox集成了bmi088，它们的坐标系都是前左上，只有平移
  // 如果是用自己的装配，需要根据条件调整，本质上就是Pw=Tib*Tbl*Pl
  // Pl是雷达坐标系的原点，在雷达坐标系就是(0,0,0)
  // Tib表示imu body坐标系到全局坐标系的转换，这里就是计算出来的rot_end,pos_end
  // Tbl表示lidar坐标系到imu body坐标系的转换，这里旋转为单位阵，平移就是Lidar_offset_to_IMU，即lidar原点在imu坐标系下的表示

#ifdef DEBUG_PRINT
  std::cout << "[ IMU Process ]: vel " << state_inout.vel_end.transpose() << " pos " << state_inout.pos_end.transpose() << " ba" << state_inout.bias_a.transpose() << " bg " << state_inout.bias_g.transpose() << std::endl;
  std::cout << "propagated cov: " << state_inout.cov.diagonal().transpose() << std::endl;
#endif

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMU_pose.end() - 1; it_kp != IMU_pose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    acc_imu << VEC_FROM_ARRAY(head->acc);
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    angvel_avr << VEC_FROM_ARRAY(head->gyr);

    //;  点云相对第一个点的时间偏执 > IMU相对第一个点的时间偏执
    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
    {
      //; 点云和imu时间偏执差
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       //; 补偿i时刻的P_l到lidar的最后一帧位姿上
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      Eigen::Matrix3d R_i(R_imu * Exp(angvel_avr, dt));  //; i时刻的lidar点，此时对应的IMU全局姿态
      Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_offset_to_IMU - pos_liD_e);
      // T_ei表示i时刻的lidar坐标原点到end时刻的坐标平移量，用的全局坐标系的衡量
      // T_ei=Ti-Te
      // 为什么是Ti-Te，在global坐标系下原点为o，Ti是向量oi,Te是向量oe,
      // Ti-Te就是向量oi-oe，得到的是向量ei,
      // 即以e指向i的向量，以e为原点的向量
      // end时刻已经计算出来了就是pos_liD_e
      // i时刻的坐标计算方式与pos_liD_e一样
      // pos_liD_i=Pos_i+rot_i*Lidar_offset_to_IMU
      // 而Pos_i=Pos_head+velocity_head*dt+0.5*acc_head*dt*dt

      // 本质上是将P_i转换到全局坐标系，再转换到end的坐标系下
      Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      //; 这个公式可能不太好理解，先把Pi用公式转到world系下，然后画向量图看一下就明白了
      Eigen::Vector3d P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin())
        break;
    }
  }
}

void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1, t2, t3;
  t1 = omp_get_wtime();

  if (meas.imu.empty())
  {
    std::cout << "no imu data" << std::endl;
    return;
  };

  ROS_ASSERT(meas.lidar != nullptr);

  //; 先对IMU数据进行初始化
  if (imu_need_init_)
  {
    // The very first lidar frame
    //; 赋值初始位姿为0，测量gyro零偏赋值，g赋值为9.8
    //! 问题：这里面的初始化好像没有调整第1帧和重力对齐？只是简单的把第一帧的位姿设置为单位阵和原点。到时候可以
    //!     测试一下，录一个包初始的时候让lidar偏一个角度，最后建出来的图应该是斜着的
    IMU_Initial(meas, stat, init_iter_num);

    imu_need_init_ = true;
    last_imu_ = meas.imu.back();

    //; MAX_INI_COUNT宏定义是0， init_iter_num初始值是1，经过上面IMU_Initial之后会变得很大，所以这里肯定满足
    if (init_iter_num > MAX_INI_COUNT)
    {
      imu_need_init_ = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",
               stat.gravity[0], stat.gravity[1], stat.gravity[2], stat.bias_g[0], stat.bias_g[1], stat.bias_g[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
    }

    //; 第一次初始化之后直接返回，不进行下面的积分、点云去畸变等操作
    return;
  }

  // Undistort points： the first point is assummed as the base frame
  // Compensate lidar points with IMU rotation (with only rotation now)

  if (0)
  {
    // UndistortPcl(meas, stat, *cur_pcl_un_);
  }
  else
  {
    //; 利用IMU数据对这一帧的lidar点云进行运动补偿，对齐到最后一个lidar点的坐标系中
    lic_point_cloud_undistort(meas, stat, *cur_pcl_un_);

    //; 利用IMU数据进行状态传播，注意就是根据上一次的状态用IMU积分得到本次状态，并不是预积分
    //; 步骤和上面的点云补偿里面的步骤完全一样，再算一遍有啥意思？
    lic_state_propagate(meas, stat);
  }

  t2 = omp_get_wtime();

  last_imu_ = meas.imu.back();

  t3 = omp_get_wtime();
}
