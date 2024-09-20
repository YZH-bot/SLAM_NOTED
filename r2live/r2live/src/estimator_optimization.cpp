//; 注意：这个文件是相对VINS中单独写的一个优化文件，应该是VINS中的处理太过复杂了，这里作者单独写了优化策略

#include "estimator.h"
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "tools_ceres.hpp"
#include "LM_Strategy.hpp"

#define USING_CERES_HUBER 0
extern Camera_Lidar_queue g_camera_lidar_queue;
extern MeasureGroup Measures;
extern StatesGroup g_lio_state;
Estimator *g_estimator;
ceres::LossFunction *g_loss_function;
int g_extra_iterations = 0;

// ANCHOR - huber_loss
void huber_loss(Eigen::Matrix<double, -1, 1> residual, double &residual_scale, double &jacobi_scale,
                double outlier_threshold = 1.0)
{
    // http://ceres-solver.org/nnls_modeling.html#lossfunction
    double res_norm = residual.norm();
    if (res_norm / outlier_threshold < 1.0)
    {
        residual_scale = 1.0;
        jacobi_scale = 1.0;
    }
    else
    {
        residual_scale = (2 * sqrt(res_norm) / sqrt(outlier_threshold) - 1.0) / res_norm;
        jacobi_scale = residual_scale;
    }
}

struct IMU_factor_res
{
    Eigen::Matrix<double, 15, 1> m_residual;   //; 残差值
    double data_buffer[4][150];
    //; 存储雅克比参数块
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor | Eigen::DontAlign>> m_jacobian_mat_vec;
    double *m_jacobian_addr_vector[4];  //; 雅克比存储地址，指向的是类成员变量 m_jacobian_mat_vec
    double *m_residual_addr;  //; 残差存储地址，指向的是这个类中的成员变量 m_residual
    double *m_parameters_vector[4];  //; 优化参数的存储地址，是添加IMU预积分约束的时候传入形参给定的
    int m_index_i;
    int m_index_j;
    IMUFactor *m_imu_factor;
    Estimator *m_estimator;  //; 这个IMU_factor属于哪个 estimator，最后是estimator来调用它

    void init()
    {
        if (m_jacobian_mat_vec.size() == 0)
        {
            m_jacobian_mat_vec.resize(4);
            m_jacobian_mat_vec[0].resize(15, 7);
            m_jacobian_mat_vec[1].resize(15, 9);
            m_jacobian_mat_vec[2].resize(15, 7);
            m_jacobian_mat_vec[3].resize(15, 9);
        }

        m_residual.setZero();
        //; matrix.data()方法就是把eigen对象转化成C++普通的数组，但是这里注意是把数组的指针传给他
        m_residual_addr = m_residual.data();  
        for (int i = 0; i < m_jacobian_mat_vec.size(); i++)
        {
            m_jacobian_mat_vec[i].setZero();
            m_jacobian_addr_vector[i] = m_jacobian_mat_vec[i].data();
        }
    };

    //; 构造函数
    IMU_factor_res()
    {
        init();
    };


    //; 添加图像关键帧之间的约束因子
    void add_keyframe_to_keyframe_factor(Estimator *estimator, IMUFactor *imu_factor, const int &index_i,
                                         const int &index_j)
    {
        m_index_i = index_i;
        m_index_j = index_j;
        m_estimator = estimator;

        //; 从这里可以看出来，VIO优化两帧的速度零偏，7+9=16
        //! 下面的操作就是把estimator中的优化参数的地址赋值给这里的成员变量
        m_parameters_vector[0] = m_estimator->m_para_Pose[m_index_i];
        m_parameters_vector[1] = m_estimator->m_para_SpeedBias[m_index_i];
        m_parameters_vector[2] = m_estimator->m_para_Pose[m_index_j];
        m_parameters_vector[3] = m_estimator->m_para_SpeedBias[m_index_j];

        m_imu_factor = imu_factor;  //; imu_factor是预积分因子，里面定义了如何计算残差和雅克比
        init();
    }

    void Evaluate()
    {
        init();
        //; 手动调用Evaluate函数（ceres优化中是自动调用），计算的残差和雅克比直接存到了成员内存m_residual_addr, m_jacobian_addr_vector中
        m_imu_factor->Evaluate(m_parameters_vector, m_residual_addr, m_jacobian_addr_vector);
    }
};

// ANCHOR - LiDAR prior factor
struct L_prior_factor
{
    Eigen::Matrix<double, -1, 1> m_residual;
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m_jacobian_mat_vec;
    int m_index_i;
    double *m_jacobian_addr_vector[5];
    double *m_residual_addr;
    double *m_parameters_vector[5];

    LiDAR_prior_factor_15 *m_lidar_prior_factor;
    Estimator *m_estimator;
    void init()
    {
        if (m_jacobian_mat_vec.size() == 0)
        {
            m_jacobian_mat_vec.resize(2);
            m_jacobian_mat_vec[0].resize(15, 7);
            m_jacobian_mat_vec[1].resize(15, 9);
            m_residual.resize(15);
        }

        m_residual.setZero();
        m_residual_addr = m_residual.data();
        for (int i = 0; i < m_jacobian_mat_vec.size(); i++)
        {
            m_jacobian_mat_vec[i].setZero();
            m_jacobian_addr_vector[i] = m_jacobian_mat_vec[i].data();
        }
    }

    L_prior_factor() = default;

    void add_lidar_prior_factor(Estimator *estimator, LiDAR_prior_factor_15 *lidar_prior_factor, const int &index_i)
    {
        m_estimator = estimator;
        m_lidar_prior_factor = lidar_prior_factor;
        m_index_i = index_i;
        m_parameters_vector[0] = m_estimator->m_para_Pose[m_index_i];
        m_parameters_vector[1] = m_estimator->m_para_SpeedBias[m_index_i];
    }

    void Evaluate()
    {
        init();
        m_lidar_prior_factor->Evaluate(m_parameters_vector, m_residual_addr, m_jacobian_addr_vector);
    }
};

// ANCHOR - Keypoint_projection_factor
struct Keypoint_projection_factor
{
    Eigen::Matrix<double, -1, 1> m_residual;
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m_jacobian_mat_vec;
    int m_index_i;
    int m_index_j;
    int m_feature_index;
    double *m_jacobian_addr_vector[5];
    double *m_residual_addr;
    double *m_parameters_vector[5];
    ProjectionTdFactor *m_projection_factor;
    Estimator *m_estimator;
    void init()
    {
        if (m_jacobian_mat_vec.size() == 0)
        {
            m_jacobian_mat_vec.resize(5);
            m_jacobian_mat_vec[0].resize(2, 7);
            m_jacobian_mat_vec[1].resize(2, 7);
            m_jacobian_mat_vec[2].resize(2, 7);
            m_jacobian_mat_vec[3].resize(2, 1);
            m_jacobian_mat_vec[4].resize(2, 1);
        }

        m_residual.resize(2);
        m_residual.setZero();
        m_residual_addr = m_residual.data();
        for (int i = 0; i < m_jacobian_mat_vec.size(); i++)
        {
            m_jacobian_mat_vec[i].setZero();
            m_jacobian_addr_vector[i] = m_jacobian_mat_vec[i].data();
        }
    };

    Keypoint_projection_factor() = default;
    ~Keypoint_projection_factor() = default;

    void add_projection_factor(Estimator *estimator, ProjectionTdFactor *projection_factor, const int &index_i,
                               const int &index_j, const int &feature_idx)
    {
        m_estimator = estimator;
        m_index_i = index_i;
        m_index_j = index_j;
        m_feature_index = feature_idx;
        m_projection_factor = projection_factor;
        m_parameters_vector[0] = m_estimator->m_para_Pose[m_index_i];
        m_parameters_vector[1] = m_estimator->m_para_Pose[m_index_j];
        m_parameters_vector[2] = m_estimator->m_para_Ex_Pose[0];
        m_parameters_vector[3] = m_estimator->m_para_Feature[m_feature_index];
        m_parameters_vector[4] = m_estimator->m_para_Td[0];
        init();
    }

    void Evaluate()
    {
        init();
        m_projection_factor->Evaluate(m_parameters_vector, m_residual_addr, m_jacobian_addr_vector);

        if (USING_CERES_HUBER)
        {
            Common_tools::apply_ceres_loss_fun(g_loss_function, m_residual, m_jacobian_mat_vec);
        }
        else
        {
            double res_scale, jacobi_scale;
            huber_loss(m_residual, res_scale, jacobi_scale, 0.5);
            if (res_scale != 1.0)
            {
                m_residual *= res_scale;
                for (int i = 0; i < m_jacobian_mat_vec.size(); i++)
                {
                    m_jacobian_mat_vec[i] *= jacobi_scale;
                }
            }
        }
    }
};

// ANCHOR -  Marginalization_factor
struct Marginalization_factor
{
    Eigen::Matrix<double, -1, 1> m_residual, m_residual_new;
    std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m_jacobian_mat_vec;
    int m_index_i;
    int m_index_j;
    int m_feature_index;
    double *m_jacobian_addr_vector[100];
    double *m_residual_addr;
    double *m_parameters_vector[100];
    int m_residual_block_size;
    std::vector<int> m_jacobian_pos;

    int m_margin_res_size;
    std::vector<int> m_margin_res_pos_vector;
    std::vector<int> m_margin_res_pos_parameters_size;
    vio_marginalization *m_vio_margin_ptr = nullptr;
    Marginalization_factor()
    {
        m_margin_res_size = 0;
    }

    ~Marginalization_factor() = default;

    // 从边缘化的约束计算雅克比
    void Evaluate_mine(Eigen::VectorXd &residual_vec, Eigen::MatrixXd &jacobian_matrix)
    {
        scope_color(ANSI_COLOR_CYAN_BOLD);
        int margin_residual_size = m_vio_margin_ptr->m_linearized_jacobians.rows();
        Eigen::VectorXd diff_x(margin_residual_size);
        Eigen::Quaterniond temp_Q;
        Eigen::Vector3d temp_t;

        m_residual_new.resize(margin_residual_size);
        m_residual_new.setZero();
        diff_x.setZero();
        
        //; 当前滑窗中第0帧的旋转
        temp_Q = Eigen::Quaterniond(g_estimator->m_para_Pose[0][6], g_estimator->m_para_Pose[0][3], g_estimator->m_para_Pose[0][4], g_estimator->m_para_Pose[0][5]).normalized();
        int pos = 15;
        if (m_vio_margin_ptr->m_margin_flag == 0) // mar oldest
        {
            //; 下面这几行就在计算(x0 - x0_marg)
            // pose[0] speed_bias[0]
            diff_x.block(0, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_Pose[0][0], g_estimator->m_para_Pose[0][1], g_estimator->m_para_Pose[0][2]) - m_vio_margin_ptr->m_Ps[1];
            diff_x.block(3, 0, 3, 1) = Sophus::SO3d(m_vio_margin_ptr->m_Rs[1].transpose() * temp_Q.toRotationMatrix()).log();
            diff_x.block(6, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][0], g_estimator->m_para_SpeedBias[0][1], g_estimator->m_para_SpeedBias[0][2]) - m_vio_margin_ptr->m_Vs[1];
            diff_x.block(9, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][3], g_estimator->m_para_SpeedBias[0][4], g_estimator->m_para_SpeedBias[0][5]) - m_vio_margin_ptr->m_Bas[1];
            diff_x.block(12, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][6], g_estimator->m_para_SpeedBias[0][7], g_estimator->m_para_SpeedBias[0][8]) - m_vio_margin_ptr->m_Bgs[1];
            //; 雅克比保持不变
            jacobian_matrix.block(0, 0, margin_residual_size, 15) = m_vio_margin_ptr->m_linearized_jacobians.block(0, 0, margin_residual_size, 15);

            //; 这里只索引到WINDOW_SIZE，因为上次的边缘化保留下的最新状态其实是本次优化的滑窗中倒数第2帧
            for (int i = 1; i < WINDOW_SIZE; i++)
            {
                temp_t = Eigen::Vector3d(g_estimator->m_para_Pose[i][0], g_estimator->m_para_Pose[i][1], g_estimator->m_para_Pose[i][2]);
                temp_Q = Eigen::Quaterniond(g_estimator->m_para_Pose[i][6], g_estimator->m_para_Pose[i][3], g_estimator->m_para_Pose[i][4], g_estimator->m_para_Pose[i][5]).normalized();
                diff_x.block(pos, 0, 3, 1) = temp_t - Eigen::Vector3d(m_vio_margin_ptr->m_Ps[i + 1]);
                diff_x.block(pos + 3, 0, 3, 1) = Eigen::Vector3d(Sophus::SO3d(m_vio_margin_ptr->m_Rs[i + 1].transpose() * temp_Q.toRotationMatrix()).log());
                jacobian_matrix.block(0, i * 15, margin_residual_size, 6) = m_vio_margin_ptr->m_linearized_jacobians.block(0, pos, margin_residual_size, 6);
                pos += 6;  //; 跳过这一帧的b_a, b_g，因为只有关于位姿的残差
            }
        }
        else if (m_vio_margin_ptr->m_margin_flag == 1) // mar second new
        {
            // pose[0] speed_bias[0]
            diff_x.block(0, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_Pose[0][0], g_estimator->m_para_Pose[0][1], g_estimator->m_para_Pose[0][2]) - m_vio_margin_ptr->m_Ps[0];
            diff_x.block(3, 0, 3, 1) = Sophus::SO3d(m_vio_margin_ptr->m_Rs[0].transpose() * temp_Q.toRotationMatrix()).log();
            diff_x.block(6, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][0], g_estimator->m_para_SpeedBias[0][1], g_estimator->m_para_SpeedBias[0][2]) - m_vio_margin_ptr->m_Vs[0];
            diff_x.block(9, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][3], g_estimator->m_para_SpeedBias[0][4], g_estimator->m_para_SpeedBias[0][5]) - m_vio_margin_ptr->m_Bas[0];
            diff_x.block(12, 0, 3, 1) = Eigen::Vector3d(g_estimator->m_para_SpeedBias[0][6], g_estimator->m_para_SpeedBias[0][7], g_estimator->m_para_SpeedBias[0][8]) - m_vio_margin_ptr->m_Bgs[0];
            jacobian_matrix.block(0, 0, margin_residual_size, 15) = m_vio_margin_ptr->m_linearized_jacobians.block(0, 0, margin_residual_size, 15);
            for (int i = 1; i < WINDOW_SIZE - 1; i++)
            {
                temp_t = Eigen::Vector3d(g_estimator->m_para_Pose[i][0], g_estimator->m_para_Pose[i][1], g_estimator->m_para_Pose[i][2]);
                temp_Q = Eigen::Quaterniond(g_estimator->m_para_Pose[i][6], g_estimator->m_para_Pose[i][3], g_estimator->m_para_Pose[i][4], g_estimator->m_para_Pose[i][5]).normalized();
                diff_x.block(pos, 0, 3, 1) = temp_t - Eigen::Vector3d(m_vio_margin_ptr->m_Ps[i]);
                diff_x.block(pos + 3, 0, 3, 1) = Eigen::Vector3d(Sophus::SO3d(m_vio_margin_ptr->m_Rs[i].transpose() * temp_Q.toRotationMatrix()).log());
                jacobian_matrix.block(0, i * 15, margin_residual_size, 6) = m_vio_margin_ptr->m_linearized_jacobians.block(0, pos, margin_residual_size, 6);
                pos += 6;
            }
        }

        //; 关于外参的雅克比和状态的变化量
        temp_t = Eigen::Vector3d(g_estimator->m_para_Ex_Pose[0][0], g_estimator->m_para_Ex_Pose[0][1], g_estimator->m_para_Ex_Pose[0][2]);
        temp_Q = Eigen::Quaterniond(g_estimator->m_para_Ex_Pose[0][6], g_estimator->m_para_Ex_Pose[0][3], g_estimator->m_para_Ex_Pose[0][4], g_estimator->m_para_Ex_Pose[0][5]);
        diff_x.block(pos, 0, 3, 1) = temp_t - m_vio_margin_ptr->m_tic[0];
        diff_x.block(pos + 3, 0, 3, 1) = Sophus::SO3d(m_vio_margin_ptr->m_ric[0].transpose() * temp_Q.toRotationMatrix()).log();
        //; 关于时间延时的雅克比和状态的变化量
        diff_x(pos + 6, 0) = g_estimator->m_para_Td[0][0] - m_vio_margin_ptr->m_td;
        jacobian_matrix.block(0, (WINDOW_SIZE + 1) * 15, margin_residual_size, 7) = m_vio_margin_ptr->m_linearized_jacobians.block(0, pos, margin_residual_size, 7);
        // !!!: FEJ，更新残差，但是雅克比不更新
        m_residual_new = m_vio_margin_ptr->m_linearized_residuals + (m_vio_margin_ptr->m_linearized_jacobians * diff_x);
        residual_vec.block(0, 0, margin_residual_size, 1) = m_residual_new;
        if (m_vio_margin_ptr->m_if_enable_debug == 1)
        {
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_res_new.txt", m_vio_margin_ptr->m_linearized_residuals);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_jac_new.txt", m_vio_margin_ptr->m_linearized_jacobians);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_residual_new.txt", m_residual_new);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_dx.txt", diff_x);
        }
    }
};

Common_tools::Timer LM_timer_tictoc;
double total_visual_res = 0;
void update_delta_vector(Estimator *estimator, Eigen::Matrix<double, -1, 1> &delta_vector)
{
    if (std::isnan(delta_vector.sum()))
    {
        return;
    }
    
    //; 得到参与优化的视觉特征点的个数
    int feature_residual_size = delta_vector.rows() - (WINDOW_SIZE + 1) * 15 - 1 - 6;
    int if_update_para = 1;

    for (int idx = 0; idx < WINDOW_SIZE + 1; idx++)
    {
        //; 优化之前的Q
        Eigen::Quaterniond q_ori = Eigen::Quaterniond(estimator->m_para_Pose[idx][6], estimator->m_para_Pose[idx][3], estimator->m_para_Pose[idx][4], estimator->m_para_Pose[idx][5]);
        //; 优化更新量 δQ
        Eigen::Quaterniond q_delta = Sophus::SO3d::exp(delta_vector.block(idx * 15 + 3, 0, 3, 1)).unit_quaternion();
        Eigen::Quaterniond q_res = (q_ori * q_delta).normalized();  //; 更新之后的Q
        
        //; 更新位置，直接加就行
        for (int element = 0; element < 3; element++)
        {
            estimator->m_para_Pose[idx][element] += delta_vector(idx * 15 + element);
        }
        
        //; 更新四元数
        estimator->m_para_Pose[idx][6] = q_res.w();
        estimator->m_para_Pose[idx][3] = q_res.x();
        estimator->m_para_Pose[idx][4] = q_res.y();
        estimator->m_para_Pose[idx][5] = q_res.z();

        //; 更新速度零偏，也是直接加就行
        for (int element = 0; element < 9; element++)
        {
            estimator->m_para_SpeedBias[idx][element] += delta_vector(idx * 15 + 6 + element);
        }
    }

    //; 更新相机和IMU之间的外参
    if (ESTIMATE_EXTRINSIC)
    {
        Eigen::Quaterniond q_ori = Eigen::Quaterniond(estimator->m_para_Ex_Pose[0][6], estimator->m_para_Ex_Pose[0][3], estimator->m_para_Ex_Pose[0][4], estimator->m_para_Ex_Pose[0][5]);
        Eigen::Quaterniond q_delta = Sophus::SO3d::exp(delta_vector.block((WINDOW_SIZE + 1) * 15 + 3, 0, 3, 1)).unit_quaternion();
        Eigen::Quaterniond q_res = (q_ori * q_delta).normalized();
        for (int element = 0; element < 3; element++)
        {
            estimator->m_para_Ex_Pose[0][element] += delta_vector[(WINDOW_SIZE + 1) * 15 + element];
        }

        estimator->m_para_Ex_Pose[0][6] = q_res.w();
        estimator->m_para_Ex_Pose[0][3] = q_res.x();
        estimator->m_para_Ex_Pose[0][4] = q_res.y();
        estimator->m_para_Ex_Pose[0][5] = q_res.z();
    }

    //; 更新时间延时
    estimator->m_para_Td[0][0] += delta_vector[(WINDOW_SIZE + 1) * 15 + 6];
    for (int element = 0; element < feature_residual_size; element++)
    {
        estimator->m_para_Feature[element][0] += delta_vector[(WINDOW_SIZE + 1) * 15 + 6 + 1 + element];
    }
}

// doc: 这个Evaluate函数和ceres中手动定义残差和雅克比的函数类似，就是把IMU、LiDAR和Camera三种约束的残差
// doc: 和雅克比都拿出来，放到最后总的大的残差和雅克比矩阵中
void Evaluate(Estimator *estimator, std::vector<IMU_factor_res> &imu_factor_res_vec,
              std::vector<Keypoint_projection_factor> &projection_factor_res_vec,
              std::vector<L_prior_factor> &lidar_prior_factor_vec,
              Marginalization_factor &margin_factor,
              const int &feature_residual_size,
              Eigen::SparseMatrix<double> &jacobian_mat_sparse,     // doc: output
              Eigen::SparseMatrix<double> &residual_sparse,         // doc: output
              int marginalization_flag = -1) // Flag = 0, evaluate all, flag = 1, marginalize old, flag = 2, marginalize last.
{
    int number_of_imu_res = imu_factor_res_vec.size();
    int number_of_projection_res = projection_factor_res_vec.size();

    //; 下面统计要优化的参数的总维度
    // pose[0], speed_bias[0],..., pose[Win+1], speed_bias[Win+1], I_CAM_E, Td, Feature_size
    int parameter_size = (WINDOW_SIZE + 1) * 15 + 6 + 1 + feature_residual_size + 1;   //; 这里为什么要 +1 ？
    int margin_res_size = 0; 
    int lidar_prior_res_size = lidar_prior_factor_vec.size();

    //; 残差的维度，imu预积分约束是15维，视觉重投影约束是2维，lidar先验约束也是15维
    int res_size = number_of_imu_res * 15 + number_of_projection_res * 2 + lidar_prior_res_size * 15;
    Eigen::Matrix<double, -1, 1> residual;
    Eigen::Matrix<double, -1, -1> jacobian_mat;
    Eigen::Matrix<double, -1, -1> hessian_mat, mat_I;

    Eigen::SparseMatrix<double> mat_I_sparse;

    //; 如果有边缘化约束，那么雅克比和残差维度都要增加
    if (margin_factor.m_vio_margin_ptr != nullptr)
    {
        margin_res_size = margin_factor.m_vio_margin_ptr->m_linearized_jacobians.rows();
        res_size += margin_res_size;
    }

    residual.resize(res_size);  //; 残差 nx1维
    jacobian_mat.resize(res_size, parameter_size);  //; 雅克比nxm维，
    hessian_mat.resize(parameter_size, parameter_size);  //; hessian矩阵mxm维
    mat_I.resize(hessian_mat.rows(), hessian_mat.cols());
    mat_I.setIdentity();
    mat_I_sparse = mat_I.sparseView();
    residual.setZero();
    jacobian_mat.setZero();
    int jacobian_pos_col;
    int res_pos_ros = 0;

    // doc: Step 0 : VIO的边缘化约束
    if (margin_factor.m_vio_margin_ptr != nullptr)
    {
        margin_factor.Evaluate_mine(residual, jacobian_mat);
    }

    // Add IMU constrain factor
    // Step 1 : 计算IMU预积分约束的残差和雅克比，并添加到最后的总的大矩阵中
    for (int idx = 0; idx < number_of_imu_res; idx++)
    {
        if (marginalization_flag == Estimator::MARGIN_OLD && idx >= 1) // Margin old
        {
            continue;
        }

        if (marginalization_flag == Estimator::MARGIN_SECOND_NEW) // Margin second new
        {
            continue;
        }
        //; 手动调用IMU的Evaluate函数，计算IMU残差和雅克比
        imu_factor_res_vec[idx].Evaluate();
        //; 当前IMU残差的位置。 可见总体的大的雅克比矩阵中，上面几行存储的是边缘化的残差
        res_pos_ros = margin_res_size + 15 * idx;  
        residual.block(res_pos_ros, 0, 15, 1) = imu_factor_res_vec[idx].m_residual;  //; 把这帧IMU的预积分约束残差加入大的残差矩阵中
       
        //; 然后在总的雅克比矩阵中添加这部分雅克比
        //; 在雅克比矩阵的列数中，前面的存储各个图像帧的参数，都是6+9 = 15维，所以位姿部分的优化参数索引是index*15, 速度零偏部分的索引是index*15+6
        jacobian_pos_col = imu_factor_res_vec[idx].m_index_i * 15; // Pos[i]
        //; 注意这里赋值的时候特地赋值是第一个雅克比矩阵块的前6列，是因为位姿部分global参数是7维，而雅克比只在前6维，就是因为四元数过参数化的问题
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 6) = imu_factor_res_vec[idx].m_jacobian_mat_vec[0].block(0, 0, 15, 6);

        jacobian_pos_col = imu_factor_res_vec[idx].m_index_i * 15 + 6; // speed_bias[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 9) = imu_factor_res_vec[idx].m_jacobian_mat_vec[1];

        jacobian_pos_col = imu_factor_res_vec[idx].m_index_j * 15; // Pos[j]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 6) = imu_factor_res_vec[idx].m_jacobian_mat_vec[2].block(0, 0, 15, 6);

        jacobian_pos_col = imu_factor_res_vec[idx].m_index_j * 15 + 6; // speed_bias[j]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 9) = imu_factor_res_vec[idx].m_jacobian_mat_vec[3];
    }

    // Add LiDAR prior residual
    // Step 2 : 计算LiDAR先验约束的残差和雅克比，并添加到最后的总的大矩阵中
    for (int idx = 0; idx < lidar_prior_factor_vec.size(); idx++)
    {
        //; 手动调用Evaluate函数，计算LiDAR先验的残差和雅克比
        lidar_prior_factor_vec[idx].Evaluate();
        //; 这里可以看到，lidar先验残差排在 margin + imu 之后
        res_pos_ros = margin_res_size + 15 * (number_of_imu_res) + idx * 15;
        //; 把残差和雅克比更新到总的矩阵中
        residual.block(res_pos_ros, 0, 15, 1) = lidar_prior_factor_vec[idx].m_residual;

        jacobian_pos_col = lidar_prior_factor_vec[idx].m_index_i * 15; // Pos[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 6, 6) = lidar_prior_factor_vec[idx].m_jacobian_mat_vec[0].block(0, 0, 6, 6);

        jacobian_pos_col = lidar_prior_factor_vec[idx].m_index_i * 15 + 6; // speed_bias[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 15, 9) = lidar_prior_factor_vec[idx].m_jacobian_mat_vec[1].block(0, 0, 15, 9);
    }

    // Add projection factor
    // Step 3 : 计算视觉重投影约束的残差和雅克比，并添加到最后的总的大矩阵中
    for (int idx = 0; idx < number_of_projection_res; idx++)
    {
        if (marginalization_flag == Estimator::MARGIN_OLD && projection_factor_res_vec[idx].m_index_i != 0)
        {
            continue;
        }

        if (marginalization_flag == Estimator::MARGIN_SECOND_NEW)
        {
            continue;
        }

        projection_factor_res_vec[idx].Evaluate();
        if (fabs((projection_factor_res_vec[idx].m_jacobian_mat_vec[3].transpose() * projection_factor_res_vec[idx].m_jacobian_mat_vec[3]).coeff(0, 0)) <= MIMIMUM_DELTA)
        {
            cout << "Visual [" << idx << "] unavailable!" << endl;
            continue;
        }

        //; 添加到最后的大矩阵中
        //; 视觉重投影的约束放在残差的最后一块，即在margin + imu + lidar 之后
        res_pos_ros = margin_res_size + 15 * (number_of_imu_res) + lidar_prior_res_size * 15 + idx * 2;
        residual.block(res_pos_ros, 0, 2, 1) = projection_factor_res_vec[idx].m_residual;
        
        //; 注意视觉重投影的雅克比是对imu_i imu_j T_ic lambda，即i j IMU位姿，相机和imu外参， 特征点逆深度这四个变量的雅克比
        jacobian_pos_col = projection_factor_res_vec[idx].m_index_i * 15; // Pos[i]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 6) = projection_factor_res_vec[idx].m_jacobian_mat_vec[0].block(0, 0, 2, 6);

        jacobian_pos_col = projection_factor_res_vec[idx].m_index_j * 15; // Pos[j]
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 6) = projection_factor_res_vec[idx].m_jacobian_mat_vec[1].block(0, 0, 2, 6);
       
        // Cam_IMU_extrinsic
        //; 添加IMU和相机外参，时间延时相关的残差和雅克比部分
        if (ESTIMATE_EXTRINSIC)
        {
            //; 外参位置是出了滑窗之后的第一个参数
            jacobian_pos_col = 15 * (WINDOW_SIZE + 1);
            jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 6) = projection_factor_res_vec[idx].m_jacobian_mat_vec[2].block(0, 0, 2, 6);
        }
        
        //; 时间延时的雅克比
        jacobian_pos_col = 15 * (WINDOW_SIZE + 1) + 6; // Time offset
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 1) = projection_factor_res_vec[idx].m_jacobian_mat_vec[4].block(0, 0, 2, 1);

        //; 逆深度的雅克比
        jacobian_pos_col = 15 * (WINDOW_SIZE + 1) + 6 + 1 + projection_factor_res_vec[idx].m_feature_index; // Keypoint res
        jacobian_mat.block(res_pos_ros, jacobian_pos_col, 2, 1) = projection_factor_res_vec[idx].m_jacobian_mat_vec[3].block(0, 0, 2, 1);
    }

    //; 把构造的残差和雅克比，转成spare矩阵形式？
    jacobian_mat_sparse = jacobian_mat.sparseView();
    residual_sparse = residual.sparseView();

    double current_cost = residual.array().abs().sum();

    return;
}

Eigen::Matrix<double, -1, 1> solve_LM(
    Eigen::SparseMatrix<double> &jacobian_mat_sparse,
    Eigen::SparseMatrix<double> &residual_sparse)
{
    int res_size = jacobian_mat_sparse.cols();
    int feature_residual_size = res_size - ((WINDOW_SIZE + 1) * 15 + 6 + 1);
    Eigen::Matrix<double, -1, 1> delta_vector;
    Eigen::SparseMatrix<double> hessian_mat_sparse, delta_vector_sparse, gradient_sparse, hessian_inv_sparse, hessian_temp_sparse;
    Eigen::Matrix<double, -1, -1> hessian_inv, gradient_dense, hessian_temp_dense;
    delta_vector.resize(res_size);
    delta_vector.setZero();
    jacobian_mat_sparse.makeCompressed();
    residual_sparse.makeCompressed();
    hessian_mat_sparse = jacobian_mat_sparse.transpose() * jacobian_mat_sparse;
    gradient_sparse = -jacobian_mat_sparse.transpose() * residual_sparse;

    LM_timer_tictoc.tic();
    hessian_temp_sparse = (hessian_mat_sparse);
    int solver_status = 0;

    delta_vector = sparse_schur_solver(hessian_temp_sparse * 1000.0, gradient_sparse * 1000.0, (WINDOW_SIZE + 1) * 15 + 6 + 1).toDense();

    double delta_vector_norm = delta_vector.block(0, 0, (WINDOW_SIZE + 1) * 15 + 6 + 1, 1).norm();

    if (delta_vector_norm > 1.0)
    {
        g_extra_iterations = 1;
    }

    return delta_vector;
}


// doc: 作者自己写的后端优化函数，没有调用ceres的库
void Estimator::optimization_LM()
{
    vector2double();
    double t_LM_cost = 0;
    double t_build_cost = 0;
    g_estimator = this;
    g_extra_iterations = 0;
    Common_tools::Timer timer_tictoc;
    timer_tictoc.tic();
    std::vector<IMU_factor_res> imu_factor_res_vec;
    std::vector<Keypoint_projection_factor> projection_factor_res_vec;
    std::vector<L_prior_factor> lidar_prior_factor_vec;
    Marginalization_factor margin_factor;
    imu_factor_res_vec.clear();
    projection_factor_res_vec.clear();
    lidar_prior_factor_vec.clear();

    g_loss_function = new ceres::HuberLoss(0.5);
    TicToc t_whole, t_prepare, t_solver;

    // doc: Step 1 : 添加约束
    // doc: Step 1.0 : 上次边缘化得到的先验约束
    if (m_vio_margin_ptr)
    {
        margin_factor.m_vio_margin_ptr = m_vio_margin_ptr;
    }

    // doc: Step 1.1 : 遍历所有图像帧，添加帧与帧之间的imu预积分约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;

        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);

        // doc: 下面如果使用ceres，直接调用AddResidualBlock即可。这里是自己写的
        IMU_factor_res imu_factor_res;
        imu_factor_res.add_keyframe_to_keyframe_factor(this, imu_factor, i, j);
        imu_factor_res_vec.push_back(imu_factor_res);
    }

    // doc: Step 1.2 : 遍历所有关键帧，添加LIO先验约束
    // doc: bug? lidar先验因子可能是没有的？--> 在lidar没有出问题的情况下数据一般都是连续的，所以lidar先验因子一般都是有的
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        LiDAR_prior_factor_15 *lidar_prior_factor = new LiDAR_prior_factor_15(&m_lio_state_prediction_vec[i]);
        L_prior_factor l_prior_factor;
        l_prior_factor.add_lidar_prior_factor(this, lidar_prior_factor, i);
        lidar_prior_factor_vec.push_back(l_prior_factor);
    }

    int f_m_cnt = 0;
    int feature_index = -1;

    // doc: Step 1.3 : 遍历所有图像特征点，添加图像特征点构成的约束
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }

            Vector3d pts_j = it_per_frame.point;

            ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                              it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                              it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
            Keypoint_projection_factor key_pt_projection_factor;
            key_pt_projection_factor.add_projection_factor(this, f_td, imu_i, imu_j, feature_index);
            projection_factor_res_vec.push_back(key_pt_projection_factor);
            f_m_cnt++;
        }
    }

    Eigen::SparseMatrix<double> residual_sparse, jacobian_sparse, hessian_sparse;
    LM_trust_region_strategy lm_trust_region;

    // doc: Step 2 进行迭代优化。配置文件中 NUM_ITERATIONS 写的是4次， g_extra_iterations会根据优化结果的不同设置成0或者1
    for (int iter_count = 0; iter_count < NUM_ITERATIONS + g_extra_iterations; iter_count++)
    {
        t_build_cost += timer_tictoc.toc();
        timer_tictoc.tic();

        //; 重要：显示的调用Evaluate函数，计算总的雅克比矩阵 jacobian_sparse,  和残差 residual_sparse
        Evaluate(this, imu_factor_res_vec, projection_factor_res_vec, lidar_prior_factor_vec, margin_factor, feature_index, jacobian_sparse, residual_sparse);
        
        Eigen::VectorXd delta_vector;
        //; 调用自己写的LM算法，计算 H * δx = -b，得到状态变量的增量δx。实际上这里给了J和e, 最后就在算 J^T * J * x = -J^T *e
        //; 最后一个变量给的是稠密块的维度，就是左上角 状态变量+外参+时间延时
        delta_vector = lm_trust_region.compute_step(jacobian_sparse, residual_sparse, (WINDOW_SIZE + 1) * 15 + 6 + 1).toDense();
        
        update_delta_vector(this, delta_vector);  //; 更新状态变量
        t_LM_cost += timer_tictoc.toc();
    }

    double2vector();
    
    // ?: 再计算一次残差和雅克比，为什么？
    // doc: 解答：是为了后面进行滑窗边缘化，上面求解得到最后优化的状态之后，并没有计算最后优化后的状态的雅克比和残差，这里计算一下为了下面的滑窗边缘化准备
    Evaluate(this, imu_factor_res_vec, projection_factor_res_vec, lidar_prior_factor_vec, margin_factor, feature_index,
             jacobian_sparse, residual_sparse, marginalization_flag);

    // doc: 最后进行VIO的边缘化操作
    if (m_vio_margin_ptr)
    {
        delete m_vio_margin_ptr;
    }

    m_vio_margin_ptr = new vio_marginalization();   //; 没有构造函数

    //; 先把滑窗中所有状态变量赋值
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        m_vio_margin_ptr->m_Ps[i] = Ps[i];
        m_vio_margin_ptr->m_Vs[i] = Vs[i];
        m_vio_margin_ptr->m_Rs[i] = Rs[i];
        m_vio_margin_ptr->m_Bas[i] = Bas[i];
        m_vio_margin_ptr->m_Bgs[i] = Bgs[i];
        m_vio_margin_ptr->m_ric[0] = ric[0];
        m_vio_margin_ptr->m_tic[0] = tic[0];
        m_vio_margin_ptr->m_td = td;
    }

    if (marginalization_flag == MARGIN_OLD)
    {
        int visual_size = jacobian_sparse.cols() - (15 * (WINDOW_SIZE + 1) + 6 + 1); // Extrinsic, Td
        hessian_sparse = jacobian_sparse.transpose() * jacobian_sparse;
        // doc: 执行边缘化操作，得到边缘化约束的残差和雅克比
        // doc: 这里传入的变量就是要进行舒尔补边缘化的方程 H * δx = b, 即J'*J * δx = -J'*e 
        m_vio_margin_ptr->margin_oldest_frame(hessian_sparse.toDense(), (jacobian_sparse.transpose() * residual_sparse).toDense(), visual_size);
    }
    else if (marginalization_flag == MARGIN_SECOND_NEW)
    {
        int visual_size = jacobian_sparse.cols() - (15 * (WINDOW_SIZE + 1) + 6 + 1); // Extrinsic, Td
        hessian_sparse = jacobian_sparse.transpose() * jacobian_sparse;
        m_vio_margin_ptr->margin_second_new_frame(hessian_sparse.toDense(), (jacobian_sparse.transpose() * residual_sparse).toDense(), visual_size);
    }
}