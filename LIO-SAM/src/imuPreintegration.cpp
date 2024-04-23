#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

// info: 订阅激光里程计（来自MapOptimization）和IMU里程计，根据前一时刻激光里程计，和该时刻到当前时刻的IMU里程计变换增量，计算当前时刻IMU里程计；rviz展示IMU里程计轨迹（局部）。
class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
        // info: 如果lidar系与baselink系不同（激光系和载体系），需要外部提供二者之间的变换关系
        if (lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s", ex.what());
            }
        }

        // info: 订阅激光里程计，来自mapOptimization                                            低频
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        // info: 订阅imu里程计，来自IMUPreintegration   是本文件另一个class发布的话题               高频
        subImuOdometry = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());

        // info: 发布imu里程计，用于rviz展示
        // ?: 目前不是很清楚发布这个的用途
        pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        // info: 发布imu里程计轨迹,显示一条小拖尾
        pubImuPath = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
    }

    // info: 类型转换 nav_msgs::Odometry ---> Eigen::Affine3f
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    // info: 激光里程计回调函数，来自mapOptimization
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    /**
     * // info: imu里程计回调函数，来自IMUPreintegration   是本文件另一个class发布的话题
     * 1、以最近一帧激光里程计位姿为基础，计算该时刻与当前时刻间imu里程计增量位姿变换，相乘得到当前时刻imu里程计位姿
     * 2、发布当前时刻里程计位姿，用于rviz展示；发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
     */
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        // static tf
        // info: 发布tf，map与odom系设为同一个系(0,0,0,0,0,0,)
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        // info: 多线程需要加锁
        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg); // info: 数据类型 std::deque<nav_msgs::Odometry>

        // get latest odometry (at current IMU stamp)
        // info: 从imu里程计队列中删除比上面激光里程计时刻 lidarOdomTime 早的数据
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }

        // info: imuOdomQueue第一个数据
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        // info: imuOdomQueue最后一个数据
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        // info: imuOdom增量
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        // info: 最近的一帧激光里程计位姿 * imu里程计增量位姿变换 = 当前时刻imu里程计位姿
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;

        // info: 转化为便于发布的数据
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

        // publish latest odometry
        // info: 发布当前时刻里程计位姿
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // info: 发布tf，当前时刻odom与baselink系变换关系
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if (lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // info: 发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
        static nav_msgs::Path imuPath;     // info: 注意是静态变量
        static double last_path_time = -1; // info: 注意是静态变量
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // info: // 每隔0.1s添加一个
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            // info: 删除最近一帧激光里程计时刻之前的imu里程计
            imuPath.poses.push_back(pose_stamped);
            while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

// doc: 用激光里程计，两帧激光里程计之间的IMU预计分量构建因子图，优化当前帧的状态（包括位姿、速度、偏置）;
// doc: 以优化后的状态为基础，施加IMU预计分量，得到每一时刻的IMU里程计。
// doc: IMU信号积分使用gtsam的IMU预积分模块。里面共使用了两个队列，imuQueImu和imuQueOpt，以及两个预积分器imuPreintegratorImu和imuPreintegratorOpt;
// doc: imuQueOpt和imuPreintegratorOpt主要是根据历史信息计算IMU数据bias给真正的IMU里程计预积分器使用。
// doc: imuQueImu和imuPreintegratorImu是真正用来做IMU里程计的优化。
class IMUPreintegration : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;

    bool systemInitialized = false;

    // info: 噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    // info: imu预积分器
    // doc: 优化IMU参数：imuIntegratorOpt_ 负责预积分两帧激光里程计之间的IMU数据，计算IMU的bias
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    // doc: 输出位姿：imuIntegratorImu_ 根据最新的激光里程计，以及后续到到的IMU数据，预测从当前激光里程计往后的位姿（IMU里程计）
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    // info: imu数据队列
    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    // info: imu因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    // info: imu状态
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    // info: ISAM2优化器
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    // T_bl: tramsform points from lidar frame to imu frame
    // info: imu-lidar位姿变换
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    // T_lb: tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        // info: 订阅imu原始数据，用下面因子图优化的结果，施加两帧之间的imu预计分量，预测每一时刻（imu频率）的imu里程计
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // info: 订阅激光里程计，来自mapOptimization，用两帧之间的imu预计分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的imu里程计，以及下一次因子图优化）
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        // info: 发布imu里程计      这个话题就是 高频 的里程计
        pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000);

        // info: imu预积分参数 PreintegrationParams: 噪声协方差 + imu bias偏置
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);     // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);          // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
        ; // assume zero initial bias // info: 假设为0

        // info: 噪声先验
        // ???: 这是什么的噪声?
        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);                                                               // m/s
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                             // 1e-2 ~ 1e-3 seems to be good
        // info: 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
        // ???: ?
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());               // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        // info: imu预积分器，用于预测每一时刻（imu频率）的imu里程计（转到lidar系了，与激光里程计同一个系）
        // ???: 为什么需要two
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /**
     * info: 关键函数
     * info: 订阅激光里程计，来自 mapOptimization
     * info: 流程如下:
     * 1. 取出里程计的位置和方向
     * 2. 判断里程计是否退化，这个会在后端中说明，什么时候会退化，退化的精度会下降
     * 3. 这帧雷达的pose转为gtsam
     * 4. 第一次进入，初始化系统：
     *   4.1 GTSAM初始化
     *  ​ 4.2 添加先验约束
     *  ​ 4.3 添加实际的状态量
     *  ​ 4.4 更新isam优化器
     *  ​ 4.5 预积分接口重置
     * 5. 加入imu数据，自动实现预积分量的更新以及协方差矩阵的更新
     * 6. 添加ImuFactor，零偏约束
     * 7. 添加雷达的位姿约束
     * 8. 失败检测（速度大于30，零偏太大）
     * 9. 根据最新的imu状态进行传播
     */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        // doc: 确保imu优化队列中有imu数据进行预积分
        if (imuQueOpt.empty())
            return;

        // doc: 当前帧激光位姿，来自mapOptimization scan-to-map匹配、因子图优化后的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        // doc: 判断里程计是否退化，这个会在后端中说明，什么时候会退化，退化的精度会下降
        // ?
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // doc: 系统初始化，在第一帧或者优化出错的情况下重新初始化
        if (systemInitialized == false)
        {
            // doc: 重置ISAM2优化器
            resetOptimization();

            // doc: 从imu优化队列中删除当前帧激光里程计时刻之前的imu数据
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }


            /**
             * info: 下面的内容为，添加因子图的信息
             *  gtsam::NonlinearFactorGraph graphFactor： 类别为：gtsam::PriorFactor
             *  添加先验速度约束
             *  添加先验零偏约束
             *  添加先验位置约束
             * 
             * gtsam::Values graphValue
             *  插入位置
             *  插入速度
             *  插入零偏
            */
            // doc:  把雷达的位姿转移到imu坐标系下         // initial pose 
            prevPose_ = lidarPose.compose(lidar2Imu);
            // doc: X(0)表示第一个位姿，有一个先验的约束。约束内容为，lidar到imu下的prevPose_这么一个位姿
            // doc: 该约束的权重，置信度为priorPoseNoise，越小代表置信度越高
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            // doc: 加入 priorPose 因子
            graphFactors.add(priorPose);    // gtsam::NonlinearFactorGraph graphFactors; 因子图
            // doc: 初始化速度因子 priorVel 为 0 
            // doc： 感觉速度可以用 imu 初始化
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);

            // doc: 加入 priorBias 因子
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);

            // doc: 将初始状态设置为因子图变量的初始值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            // doc: 加入新的因子，并使用优化器对因子图做一次优化
            // doc: 可以多次调用update()对当前的因子图进行多次更新
            optimizer.update(graphFactors, graphValues);

            // doc: 进入优化器之后保存约束和状态量的变量就清零
            // doc: 清空因子图。因子已经已经被记录到优化器中。这是在gtsam的官方example的递增式优化流程中的用法
            // doc: example:gtsam/examples/VisualSAM2Example.cpp
            // ?
            graphFactors.resize(0);
            graphValues.clear();

            // doc: 重置两个预积分器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            key = 1;
            // doc: 设置系统已经初始化标志位
            systemInitialized = true;
            return;
        }

        // doc: 每100帧lidar里程计重置一下优化器
        // doc: 删除旧因子图，加快优化速度
        // INFO: 
        // doc: 讨论：Liosam这个预积分的子图不是滑窗，传统滑窗是新增一帧，边缘化掉旧帧，维持滑窗尺寸不变。这里直接累积到100帧再优化，
        // doc: 一次性边缘化只保留最新帧，然后reset了……这样会用旧的bias导致预测的旋转在前后几帧表现出来差异太大，可视化的效果就是imu
        // doc: 预测轨迹成折线，估计是作者考虑到滑窗实现的问题，才这么整的。尽管gtsam中有fix lag smoother，但估计是考虑用isam，其实
        // doc: 现方法还是有点问题，没有去同步处理旧的这些factor啥的，导致其内存占用会缓慢增长。
        if (key == 100)
        {
            // doc: 从优化器中先缓存一下当前优化出来的变量的方差的
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
            // doc: 重置优化器和因子图
            resetOptimization();
            // doc: 把上一次优化出的位姿作为重新初始化的priorPose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // doc: 把上一次优化出的速度作为重新初始化的priorVel
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // doc: 把上一次优化出的bias作为重新初始化的priorBias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // doc: 将prior状态设置成初始估计值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // doc: 进行一次迭代优化
            optimizer.update(graphFactors, graphValues);
            // doc: 清空因子图和值（已经被保存进优化器里了）
            graphFactors.resize(0);
            graphValues.clear();

            // doc: 重置因子索引
            key = 1;
        }

        // doc: 将imuQueOpt队列中，所有早于当前雷达里程计的数据进行积分，获取最新的IMU bias
        // info: 就是这里会出现断层，以前的imu信息只保留了结果，这里重新用新的进行更新，可能就会导致imu优化的出来的参数和原来的差距比较大
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            // info：先积分 然后删除比当前mapOptimization 位姿旧的imu数据
            if (imuTime < currentCorrectionTime - delta_t)
            {
                // 时间差
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                // doc: 这里是实际做积分的地方
                imuIntegratorOpt_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // info: 下面这一大部分是使用IMU预积分的结果加入因子图中，同时，加入雷达里程计因子，使用优化器优化因子图得出当前IMU对应的Bias。
        // doc: 使用IMU预积分的结果构建IMU因子，并加入因子图中
        const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
        // doc: 上一帧的速度和位姿，当前帧的速度和位姿还有上一帧的零偏，以及imu预积分测量
        // doc: imu 因子连接相邻两个状态，并且包括了这两帧之间的imu测量，很合理吧
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // doc: 添加IMU的bias的BetweenFactor
        // doc: 这里的 deltaTij()获取的是IMU预积分器从上一次积分输入到当前输入的时间
        // doc: noiseModelBetweenBias是我们提前标定的IMU的偏差的噪声
        // doc: 两者的乘积就等于这两个时刻之间IMU偏差的漂移
        // ?: deltaTij
        // doc：说白了，这里应该是告诉因子图，这两个imu bias状态已经连接了，记得给我优化
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

        // info：以下添加雷达的位姿约束
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        // doc: 作为先验因子，如果是退化的话，置信度小一点。这里的degenerate调整噪声的大小。correctionNoise2是大噪声
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        // doc: 说白了，我们只有imu是不行的，这里要用lidar（来自另后端优化）的pose初始化状态X(k), 然后才能联合优化bias
        graphFactors.add(pose_factor);

        // doc: imuIntegratorOpt_->predict输入之前时刻的状态和偏差，预测当前时刻的状态
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // doc: 将IMU预积分的结果作为当前时刻因子图变量的初值
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);

        // optimize
        // doc: 优化两次，两次调整会得到一个好的结果
        optimizer.update(graphFactors, graphValues);
        optimizer.update();

        // doc: 清空因子图和值（已经被保存进优化器里了）
        graphFactors.resize(0);
        graphValues.clear();

        // doc: 从优化器中获取当前经过优化后估计值
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_ = result.at<gtsam::Pose3>(X(key));
        prevVel_ = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));

        // Reset the optimization preintegration object.
        // doc: 重置预积分器
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // doc: 检查优化器优化结果，当优化出的速度过快（大于30），或者Bias过大，认为优化或者某个环节出现问题，重置优化器
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }

        /**
         * info:这一部分是将偏差优化器imuIntegratorOpt_/graphFactors优化出的结果传递到IMU里程计的预积分器。
         * 让IMU里程计预积分器使用最新估计出来的Bias进行积分
         *  1. 缓存最新的状态（作为IMU里程计预积分器预测时的上一时刻状态传入），和最新的偏差（重置IMU里程计预积分器时使用）
         *  2. 同步IMU数据队列和雷达里程计时间，去除当前雷达里程计时间之前的数据
         *  3. 对IMU队列中剩余的其他数据进行积分。（这样在新的IMU到来的时候就可以直接在这个基础上进行积分）
        */
        // 2. after optiization, re-propagate imu odometry preintegration
        // doc: 根据最新的imu状态进行传播      注意：这里imu队列是另外一个队列了，所以要重新刷新掉旧的数据
        prevStateOdom = prevState_;
        prevBiasOdom = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // info: 根据最新的imu状态进行传播
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // info：这里高频 imu_odom 会清空之前的预积分，利用新的 bias 重新计算队列里面的 imu 预积分      注意：两个回调函数都加了锁，不会重复计算的
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    /**
     * info：这个就是高频的 imu_odom 应该就是 imu 频率左右的吧，除了锁 mtx 的可能会导致频率降低
     * info: 该回调函数会把接收到的imu数据全部转换到lidar坐标系上，并且把收到的消息传入到两个队列中，一个用于预积分和位姿的优化，一个用于更新最新imu状态。
     * info: 只有在odometry 的回调函数处理过后doneFirstOpt才会true。下面先看，在设置为true之前会发生什么
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr &imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // info: 把imu转换到lidar坐标系上 
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // info: 只有odometry handler (另外一个回调函数) 之后才会true      
        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "roboat_loam");

    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}
