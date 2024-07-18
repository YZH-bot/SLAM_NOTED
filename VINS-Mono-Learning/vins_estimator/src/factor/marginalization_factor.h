#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

// info: 解释：https://blog.csdn.net/hltt3838/article/details/109649675
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;// doc: 优化变量数据 该约束因子相关联的参数块变量
    std::vector<int> drop_set;// doc: 待边缘化的优化变量id      parameter_blocks中待marg变量的索引

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;// ? doc: 残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// doc: 这个类保存了优化时上一步边缘化后保留下来的先验信息
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;// doc: 所有观测项
    // doc: 这里将参数块分为Xm,Xb,Xr,Xm表示被marg掉的参数块，Xb表示与Xm相连接的参数块，Xr表示剩余的参数块
    // doc: 那么m=Xm的localsize之和，n为Xb的localsize之和， pos 为（Xm+Xb）localsize之和
    int m, n;// doc: m为要边缘化的变量个数，n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size; // doc: <优化变量内存地址,localSize>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; // doc: <待边缘化的优化变量内存地址,在parameter_block_size中的id> 将被marg掉的约束边相关联的参数块，即将被marg掉的参数块以及与它们直接相连的参数快
    std::unordered_map<long, double *> parameter_block_data;// doc: <优化变量内存地址,数据>

    std::vector<int> keep_block_size; //global size  
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

// info: 类描述：该类是优化时表示上一步边缘化后保留下来的先验信息代价因子，变量marginalization_info保存了类似约束测量信息
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
