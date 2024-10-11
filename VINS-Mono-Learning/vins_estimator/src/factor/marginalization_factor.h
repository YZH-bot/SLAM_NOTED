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
// doc: 这个类就是边缘化因子: 残差
struct ResidualBlockInfo
{
    // doc: 构造函数： 传入 cost_function, loss_function, parameter_blocks 相关优化变量, drop_set待边缘化的变量
    // doc: ceres::CostFunction 是 ceres::SizedCostFunction 的父类，所以可以传入 IMUFactor、ProjectionTdFactor 等因子
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;// doc: 优化变量数据 该约束因子相关联的参数块变量
    std::vector<int> drop_set;// doc: 待边缘化的优化变量id      parameter_blocks中待marg变量的索引

    double **raw_jacobians; // doc: 雅可比矩阵
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; // doc: 雅可比矩阵
    Eigen::VectorXd residuals;// ? doc: 误差项：残差 IMU:15X1 视觉2X1

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
    // doc: 将所有参与到边缘化的因子信息加进来，即 ResidualBlockInfo
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    // doc: 预处理
    void preMarginalize();
    // doc: 边缘化操作
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;// doc: 所有观测项
    // doc: 这里将参数块分为Xm,Xb,Xr,Xm表示被marg掉的参数块，Xb表示与Xm相连接的参数块，Xr表示剩余的参数块
    // doc: 那么m=Xm的localsize之和，n为Xb的localsize之和，
    int m, n;// doc: m 为要边缘化变量的总维度，n 为要保留下来变量的总维度
    // info: 这里三个 umordered_map 的 key 的 long 类型是一样的，都表示变量的内存地址转换成的 long 类型
    std::unordered_map<long, int> parameter_block_size; // doc: <优化变量内存地址,localSize>: 每个优化变量的 Global 维度
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; // doc: 每个变量在 Hessian 矩阵中的索引，将要 marg 的变量放在前面
    std::unordered_map<long, double *> parameter_block_data;// doc: <优化变量内存地址,数据>：每个变量存储的数据

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
