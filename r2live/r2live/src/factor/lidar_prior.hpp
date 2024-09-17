#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>
#include "common_lib.h"

// ANCHOR:  LiDAR prior factor here.
class LiDAR_prior_factor : public ceres::SizedCostFunction<6, 7>
{
public:
    eigen_q m_prior_q = eigen_q::Identity();
    vec_3 m_prior_t = vec_3::Zero();
    LiDAR_prior_factor() = delete;
    StatesGroup *m_lio_prior_state;

    LiDAR_prior_factor(StatesGroup *lio_prior_state) : m_lio_prior_state(lio_prior_state)
    {
        m_prior_q = eigen_q(m_lio_prior_state->rot_end);
        m_prior_t = m_lio_prior_state->pos_end;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        double w_s = 0.1;

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        Eigen::Matrix<double, 6, 6> cov_mat_temp = m_lio_prior_state->cov.block(0, 0, 6, 6);
        cov_mat_temp.block(0, 0, 3, 6).swap(cov_mat_temp.block(3, 0, 3, 6));
        cov_mat_temp.block(0, 0, 6, 3).swap(cov_mat_temp.block(0, 3, 6, 3));

        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(cov_mat_temp.inverse()).matrixL().transpose();

        residual.block(0, 0, 3, 1) = (Pi - m_prior_t) * w_s;
        residual.block(3, 0, 3, 1) = Sophus::SO3d((m_prior_q.inverse() * (Qi))).log() * w_s;

        residual = sqrt_info * residual;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) = mat_3_3::Identity();
                jacobian_pose_i.block<3, 3>(O_R, O_R) = inverse_right_jacobian_of_rotion_matrix(Sophus::SO3d((m_prior_q.inverse() * (Qi))).log());
                jacobian_pose_i = w_s * sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }
        }

        return true;
    }
};

#if 1
// ANCHOR:  LiDAR prior factor here.
//; 残差维度15，优化参数维度：第一个参数块7，第二个参数块9
class LiDAR_prior_factor_15 : public ceres::SizedCostFunction<15, 7, 9>
{
public:
    eigen_q m_prior_q = eigen_q::Identity();
    vec_3 m_prior_t = vec_3::Zero();
    Eigen::Matrix<double, 9, 1> m_prior_speed_bias;
    LiDAR_prior_factor_15() = delete;
    StatesGroup *m_lio_prior_state;

    LiDAR_prior_factor_15(StatesGroup *lio_prior_state) : m_lio_prior_state(lio_prior_state)
    {
        m_prior_q = eigen_q(m_lio_prior_state->rot_end);
        m_prior_t = m_lio_prior_state->pos_end;
        m_prior_speed_bias.block(0, 0, 3, 1) = m_lio_prior_state->vel_end;
        m_prior_speed_bias.block(3, 0, 3, 1) = m_lio_prior_state->bias_a;
        m_prior_speed_bias.block(6, 0, 3, 1) = m_lio_prior_state->bias_g;
    }

    //; 计算LiDAR先验的残差和雅克比
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //; 参数块位姿（Pi Qi）是相机的位姿
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix<double, 9, 1> speed_bias;
        speed_bias << parameters[1][0], parameters[1][1], parameters[1][2],
            parameters[1][3], parameters[1][4], parameters[1][5],
            parameters[1][6], parameters[1][7], parameters[1][8];

        //! 这个ws是干嘛的？这样不是相当于又手动给lidar先验的残差乘了一个权重，让lidar先验的权重更加低了？
        double w_s = 0.1;

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        Eigen::Matrix<double, 15, 15> cov_mat_temp;

        cov_mat_temp = m_lio_prior_state->cov.block(0, 0, 15, 15);

        //; 下面的操作就是把lio中维护的协方差矩阵按照vio中用到的协方差矩阵顺序进行排列，
        //; lio: Q P V bg ba g       vio: P Q V ba bg,  所以需要交换P和Q，bg和ba, v保持不动
        cov_mat_temp.block(0, 0, 3, 15).swap(cov_mat_temp.block(3, 0, 3, 15));  //; 按行，P和Q交换
        cov_mat_temp.block(9, 0, 3, 15).swap(cov_mat_temp.block(12, 0, 3, 15)); //; 按行，bg和ba交换
        cov_mat_temp.block(0, 0, 15, 3).swap(cov_mat_temp.block(0, 3, 15, 3));  //; 按列，P和Q交换
        cov_mat_temp.block(0, 9, 15, 3).swap(cov_mat_temp.block(0, 12, 15, 3)); //; 按列，bg和ba交换

        //; 协方差矩阵求逆变成信息矩阵，然后开根号，当做残差的权重乘到残差上面
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(cov_mat_temp.inverse()).matrixL().transpose();

        residual.block(0, 0, 3, 1) = (Pi - m_prior_t) * w_s;  //; m_prior_t 就是lidar先验的位置
        residual.block(3, 0, 3, 1) = Sophus::SO3d((m_prior_q.inverse() * (Qi))).log() * w_s;  //; 姿态误差求李代数
        residual.block(6, 0, 9, 1) = (speed_bias - m_prior_speed_bias) * w_s;  //; 速度和零偏

        residual = sqrt_info * residual;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                //; 关于相机位姿的平移雅克比，就是单位阵
                jacobian_pose_i.block<3, 3>(0, 0) = mat_3_3::Identity();
                //; 关于相机位姿的旋转雅克比，用李代数求导的方式
                jacobian_pose_i.block<3, 3>(3, 3) = inverse_right_jacobian_of_rotion_matrix(Sophus::SO3d((m_prior_q.inverse() * (Qi))).log());

                jacobian_pose_i = w_s * sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
                //; 15维的残差，只有后面的v、ba、bg对自己才有导数，而PQ对他们都没有导数，所以前面的置为0，对他们自己的导数置为单位帧
                jacobian_pose_i.setZero();
                jacobian_pose_i.block(6, 0, 9, 9).setIdentity();
                jacobian_pose_i = w_s * sqrt_info * jacobian_pose_i;
                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                }
            }
        }

        return true;
    }
};
#endif;