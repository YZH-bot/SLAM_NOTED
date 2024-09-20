#include "estimator.h"

extern Camera_Lidar_queue g_camera_lidar_queue;
extern MeasureGroup Measures;
extern StatesGroup g_lio_state;

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

// 视觉测量残差的协方差矩阵
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = READ_TIC[i];
        ric[i] = READ_RIC[i];
    }

    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

// 清空或初始化滑动窗口中所有的状态量
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];

        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : m_all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    m_all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;

    tmp_pre_integration = nullptr;
    m_vio_margin_ptr = nullptr;

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

int apply_tr_times = 0;

//; 估计的vio状态变量和lio状态变量相差太大，以lio为准修复vio。注意这里的修复不包括零偏
int Estimator::refine_vio_system(eigen_q q_extrinsic, vec_3 t_extrinsic)
{
    Eigen::Matrix<double, 3, 3> R_mat_e = q_extrinsic.toRotationMatrix();
    for (int win_idx = 0; win_idx <= WINDOW_SIZE; win_idx++)
    {
        Rs[win_idx] = Rs[win_idx] * R_mat_e;
        Ps[win_idx] = Ps[win_idx] + t_extrinsic;
        Vs[win_idx] = R_mat_e * Vs[win_idx];
    }

    for (int win_idx = 0; win_idx <= WINDOW_SIZE; win_idx++)
    {
        pre_integrations[win_idx]->delta_p = R_mat_e * pre_integrations[win_idx]->delta_p;
    }

    back_R0 = back_R0 * R_mat_e;
    last_R0 = last_R0 * R_mat_e;
    last_R = last_R * R_mat_e;

    back_P0 = back_P0 + t_extrinsic;
    last_P = last_P + t_extrinsic;
    last_P0 = last_P0 + t_extrinsic;

    return 1;
}

/*
    处理IMU数据
    IMU预积分，中值积分得到当前PQV作为优化初值
    dt 时间间隔
    linear_acceleration 线加速度
    angular_velocity 角速度
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - m_gravity;

        // 采用的是中值积分的传播方式
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - m_gravity;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//; 这个是原先VINS中最重要的处理图片的函数，里面实现后端优化、滑窗等操作，这里单独摘出来了
void Estimator::solve_image_pose(const std_msgs::Header &header)
{
    // doc: Step 1： 外参初始化，也就是在线标定IMU和camera之间的外参
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // doc: Step 2 : 下面根据solver_flag来判断是进行初始化还是进行非线性优化
    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // Step 3： VIO初始化，视觉-惯导联合初始化
            if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                result = initialStructure();   //; 视觉-惯导联合初始化
                initial_timestamp = header.stamp.toSec();  //; initial_timestamp类成员变量，是上一次初始化的时间戳
                //; 下面把视觉的初始化时间设置成了无穷大，这样代表视觉还没有初始化？
                g_camera_lidar_queue.m_visual_init_time = 3e88; // Set state as uninitialized.
                g_camera_lidar_queue.m_camera_imu_td = 0;
            }

            //; VIO 初始化成功
            if (result)
            {
                solver_flag = NON_LINEAR;
                // doc: Step 4： 后端非线性优化，边缘化
                solveOdometry();

                // Step 5： 滑动窗口，移除边缘化的帧
                slideWindow();

                // Step 6： 移除无效地图点，就是被三角化失败的点
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }

    // doc: 进行初始化之后，后面都会进入else这个分支
    else
    {
        TicToc t_solve;
        // doc: Step 1 后端非线性优化，边缘化
        // doc: 注意这里面的后端优化是作者自己写的LM算法，没有调用ceres的库，所以这个地方还是和vins不同的
        solveOdometry();   


        ROS_DEBUG("Odom solver costs: %fms", t_solve.toc());
        // Step 1.5 检测VIO是否正常
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            g_camera_lidar_queue.m_visual_init_time = 3e88; // Set state as uninitialized.
            g_camera_lidar_queue.m_camera_imu_td = 0;

            return;
        }

        TicToc t_margin;
        // Step 2 滑窗，移除最老帧或者倒数第二帧
        slideWindow();

        // Step 3 移除无效地图点，就是被三角化失败的点
        f_manager.removeFailures();
        ROS_DEBUG("whole marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/*
    处理图像特征数据
    addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧
    判断并进行外参标定
    进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
    image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
    header 某帧图像的头信息
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header, const StatesGroup &state_prior)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // Step 1 将特征点信息加到f_manager这个特征点管理器中，同时进行是否关键帧的检查
    // 添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
    // 通过检测两帧之间的视差决定次新帧是否作为关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // 将图像数据、时间、临时预积分值存到图像帧类中
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;

    //; 注意这里和VINS不同，这里是new一个状态类传入，作为状态先验
    imageframe.m_state_prior = state_prior; 

    m_all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));

    // 更新临时预积分初始值
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //! 这里把VINS的后端全部都去掉了，因为VINS后端是基于非线性优化的
}

/*
    视觉的结构初始化
    确保IMU有充分运动激励
    relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
    sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
    visualInitialAlign()视觉惯性联合初始化
*/
bool Estimator::initialStructure()
{
    TicToc t_sfm;

    // 通过加速度标准差判断IMU是否有充分运动以初始化
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = m_all_image_frame.begin(), frame_it++; frame_it != m_all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }

        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)m_all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = m_all_image_frame.begin(), frame_it++; frame_it != m_all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }

        var = sqrt(var / ((int)m_all_image_frame.size() - 1)); // 标准差
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
        }
    }

    // 将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }

        sfm_f.push_back(tmp_feature);
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int l;

    // 保证具有足够的视差,由F矩阵恢复Rt
    // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    // 此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    // 对窗口中每个图像帧求解sfm问题
    // 得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        // 求解失败则边缘化最早一帧并滑动窗口
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = m_all_image_frame.begin();
    for (int i = 0; frame_it != m_all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }

        if ((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }

        // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        // 罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }

        // 保证特征点数大于5
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }

        /*
            bool cv::solvePnP(   求解pnp问题
                InputArray  objectPoints,   特征点的3D坐标数组
                InputArray  imagePoints,    特征点对应的图像坐标
                InputArray  cameraMatrix,   相机内参矩阵
                InputArray  distCoeffs,     失真系数的输入向量
                OutputArray     rvec,       旋转向量
                OutputArray     tvec,       平移向量
                bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
                int     flags = SOLVEPNP_ITERATIVE 采用LM优化
            )   
         */
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }

        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // 进行视觉惯性联合初始化
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

/*
    视觉惯性联合初始化
    陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
    更新了Bgs后，IMU测量量需要repropagate
    得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;

    // 计算陀螺仪偏置，尺度，重力加速度和速度
    bool result = VisualIMUAlignment(m_all_image_frame, Bgs, m_gravity, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = m_all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = m_all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        m_all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;

    f_manager.clearDepth(dep);

    // 重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();

    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);

    // 陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }

    // 将Ps、Vs、depth尺度s缩放
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); // Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换

    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = m_all_image_frame.begin(); frame_i != m_all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3); // Vs为优化得到的速度
        }
    }

    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth *= s;
    }

#if 1
    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(m_gravity);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    m_gravity = R0 * m_gravity;
#else
    Matrix3d R0 = Utility::g2R(Eigen::Vector3d(0, 0, 9.805));
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    m_gravity = R0 * m_gravity;
#endif

    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    // 所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }

    ROS_DEBUG_STREAM("g0     " << m_gravity.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/*
    判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
    solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
    relative_R 当前帧到第l帧之间的旋转矩阵R
    relative_T 当前帧到第l帧之间的平移向量T
    L 保存滑动窗口中与当前帧满足初始化条件的那一帧
    bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // 寻找第i帧到窗口最后一帧的对应特征点
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                // 第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }

            average_parallax = 1.0 * sum_parallax / int(corres.size());

            // 判断是否满足初始化条件：视差>30和内点数满足要求
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }

    return false;
}

// doc: 三角化求解所有特征点的深度，并进行非线性优化
void Estimator::solveOdometry()
{
    // doc: 窗口内的图像帧数小于窗口大小，则不进行优化
    if (frame_count < WINDOW_SIZE)
        return;

    if (solver_flag == NON_LINEAR)  // doc: 初始化成功
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());

        // doc: 注意：这里和原来vins中的不同，原来VINS中的处理很复杂，这里自己用了自己写的函数
        // doc:主要是手动构造大的残差和雅克比矩阵，然后利用LM方法求解
        optimization_LM();
    } 
}

// vector转换成double数组，因为ceres使用数值数组
// Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        m_para_Pose[i][0] = Ps[i].x();
        m_para_Pose[i][1] = Ps[i].y();
        m_para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        m_para_Pose[i][3] = q.x();
        m_para_Pose[i][4] = q.y();
        m_para_Pose[i][5] = q.z();
        m_para_Pose[i][6] = q.w();

        m_para_SpeedBias[i][0] = Vs[i].x();
        m_para_SpeedBias[i][1] = Vs[i].y();
        m_para_SpeedBias[i][2] = Vs[i].z();

        m_para_SpeedBias[i][3] = Bas[i].x();
        m_para_SpeedBias[i][4] = Bas[i].y();
        m_para_SpeedBias[i][5] = Bas[i].z();

        m_para_SpeedBias[i][6] = Bgs[i].x();
        m_para_SpeedBias[i][7] = Bgs[i].y();
        m_para_SpeedBias[i][8] = Bgs[i].z();
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        m_para_Ex_Pose[i][0] = tic[i].x();
        m_para_Ex_Pose[i][1] = tic[i].y();
        m_para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        m_para_Ex_Pose[i][3] = q.x();
        m_para_Ex_Pose[i][4] = q.y();
        m_para_Ex_Pose[i][5] = q.z();
        m_para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        m_para_Feature[i][0] = dep(i);

    if (ESTIMATE_TD)
        m_para_Td[0][0] = td;
}

// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正
void Estimator::double2vector()
{
    // 窗口第一帧之前的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    // 优化后的位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(m_para_Pose[0][6],
                                                     m_para_Pose[0][3],
                                                     m_para_Pose[0][4],
                                                     m_para_Pose[0][5])
                                             .toRotationMatrix());
    // 求得优化前后的姿态差
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(m_para_Pose[0][6],
                                       m_para_Pose[0][3],
                                       m_para_Pose[0][4],
                                       m_para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    // 根据位姿差做修正，即保证第一帧优化前后位姿不变
    //! 这个地方还是存在一点问题的，因为有了lidar先验就相当于有了6-DOF的可观性质，这里4-DOF的不可观就全部消失了
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = rot_diff * Quaterniond(m_para_Pose[i][6], m_para_Pose[i][3], m_para_Pose[i][4], m_para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(m_para_Pose[i][0] - m_para_Pose[0][0],
                                    m_para_Pose[i][1] - m_para_Pose[0][1],
                                    m_para_Pose[i][2] - m_para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(m_para_SpeedBias[i][0],
                                    m_para_SpeedBias[i][1],
                                    m_para_SpeedBias[i][2]);

        Bas[i] = Vector3d(m_para_SpeedBias[i][3],
                          m_para_SpeedBias[i][4],
                          m_para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(m_para_SpeedBias[i][6],
                          m_para_SpeedBias[i][7],
                          m_para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(m_para_Ex_Pose[i][0],
                          m_para_Ex_Pose[i][1],
                          m_para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(m_para_Ex_Pose[i][6],
                             m_para_Ex_Pose[i][3],
                             m_para_Ex_Pose[i][4],
                             m_para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = m_para_Feature[i][0];

    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = m_para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - m_para_Pose[0][0],
                                     relo_Pose[1] - m_para_Pose[0][1],
                                     relo_Pose[2] - m_para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());

        relocalization_info = 0;
    }
}

// 系统故障检测 -> Paper VI-G
bool Estimator::failureDetection()
{
    // 在最新帧中跟踪的特征数小于某一阈值
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        return true;
    }

    // 偏置或外部参数估计有较大的变化
    if (Bas[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }

    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }

    if ((tic[0] - READ_TIC[0]).norm() > 1.0)
    {
        ROS_INFO(" big Extrinsic T error %f", (tic[0] - TIC[0]).norm());
        return true;
    }

    double ext_R_diff = Eigen::Quaterniond(ric[0]).angularDistance(Eigen::Quaterniond(READ_RIC[0])) * 57.3;
    if (ext_R_diff > 20)
    {
        ROS_INFO(" big Extrinsic R error %f", ext_R_diff);
        return true;
    }

    // 最近两个估计器输出之间的位置或旋转有较大的不连续性
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 2.0)
    {
        ROS_INFO(" big translation");
        return true;
    }

    if (abs(tmp_P.z() - last_P.z()) > 2.0)
    {
        ROS_INFO(" big z translation");
        return true;
    }

    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 20)
    {
        ROS_INFO(" big delta_angle ");
        return true;
    }

    return false;
}

/*
    实现滑动窗口all_image_frame的函数
    如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
    如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
*/
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);

                //; 注意这里交换了lidar prior因子
                std::swap(m_lio_state_prediction_vec[i], m_lio_state_prediction_vec[i + 1]);
            }

            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
            //; 这里直接把滑窗中最新帧的lio先验赋值成当前帧问题也不大，因为一般只要lidar是连续的，下一次优化的时候
            //; 就会更新最新帧的lio状态了，不过还是有一定的风险。
            m_lio_state_prediction_vec[WINDOW_SIZE] = m_lio_state_prediction_vec[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = m_all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = m_all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;

                    it->second.pre_integration = NULL;
                }

                m_all_image_frame.erase(m_all_image_frame.begin(), it_0);
                m_all_image_frame.erase(t_0);
            }

            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            //; 状态值前移，但是滑窗中最新的状态值没有赋值，感觉也是不太严谨
            m_lio_state_prediction_vec[frame_count - 1] = m_lio_state_prediction_vec[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// 滑动窗口边缘化次新帧时处理特征点被观测的帧号
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// 滑动窗口边缘化最老帧时处理特征点被观测的帧号
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // back_R0、back_P0为窗口中最老帧的位姿
        // Rs、Ps为滑动窗口后第0帧的位姿，即原来的第1帧
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

/*
    进行重定位
    _frame_stamp    重定位帧时间戳
    _frame_index    重定位帧索引值
    _match_points   重定位帧的所有匹配点
    _relo_t     重定位帧平移向量
    _relo_r     重定位帧旋转矩阵
*/
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = m_para_Pose[i][j];
        }
    }
}
