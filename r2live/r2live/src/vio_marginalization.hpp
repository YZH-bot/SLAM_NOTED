#pragma once

#include "tools_data_io.hpp"
#include "tools_eigen.hpp"
#include "tools_timer.hpp"
#include "parameters.h"

/* 
    VIO边缘化
    如果仅仅从前后两帧图像计算相机位姿，速度快但是精度低；但是采用全局优化BA，连接所有图像帧，精度高但是复杂度高。
    采用滑动窗，固定数量的帧进行优化，这样能够保证精度和速度。既然是滑动窗，在滑动的过程中会有新的图像进来，旧的图像离开，所谓边缘化就是为了删除图像，但是把图像信息约束保留下来。
    边缘化主要利用的技巧是Schur Complement（舒尔补）
 */
class vio_marginalization
{
public:
    Eigen::MatrixXd A, b;
    Eigen::Matrix<double, -1, -1, Eigen::DontAlign> m_linearized_jacobians, m_linearized_residuals;
    std::vector<int> idx_to_margin, idx_to_keep;
    int m_if_enable_debug = 0;
    Eigen::Vector3d m_Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d m_Vs[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d m_Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d m_Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d m_Bgs[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d m_ric[1];
    Eigen::Vector3d m_tic[1];
    int m_margin_flag = 0;
    double m_td;

    void printf_vector(std::vector<int> &vec_int, std::string str = std::string(" "))
    {
        cout << str << " ";
        for (int i = 0; i < vec_int.size(); i++)
        {
            cout << vec_int[i] << ", ";
        }

        cout << endl;
    }

    std::vector<int> find_related_visual_landmark_idx(const Eigen::MatrixXd &mat_hessian, std::vector<int> &idx_to_margin, int mar_pos, int visual_size)
    {
        std::vector<int> res_idx_vector;
        int para_size = mat_hessian.cols();
        int visual_start_idx = para_size - visual_size;
        for (int idx = visual_start_idx; idx < para_size; idx++)
        {
            //; 对最老帧来说，这里就是(0,index)开始，选15,1的块
            if (mat_hessian.block(mar_pos, idx, 15, 1).isZero() == false)
            {
                idx_to_margin.push_back(idx);
            }
        }

        return idx_to_margin;
    }

    // 执行边缘化
    //! bug? 这里的边缘化操作我感觉应该还是存在一些问题的，因为这里直接用了最后的H矩阵，就视觉重投影包括了所有的
    //!      滑窗中的关键帧，那么最后的H矩阵中每个位姿的地方也是由很多不同的雅克比块相乘再相加得到的，这样直接用H
    //!      进行边缘化，就不是纯第0帧构成的残差进行边缘化了。
    //! 再思考：如果我用的就是第0帧的残差的雅克比矩阵构成的H矩阵，没有其他残差的雅克比的污染，但是仍然使用总的H的维度
    //!        会有问题吗？
    //! 解答：不会有问题！因为舒尔补之后的H矩阵中和这些状态变量所在的维度都是0。
    void marginalize(const Eigen::MatrixXd &mat_hessian, const Eigen::MatrixXd &mat_residual, std::vector<int> &idx_to_margin, std::vector<int> &idx_to_keep)
    {
        // sorting matrices
        int to_keep_size = idx_to_keep.size();
        int to_margin_size = idx_to_margin.size();
        int raw_hessian_size = mat_hessian.rows();
        Eigen::VectorXd res_residual(to_keep_size + to_margin_size);
        //; 这个变量是hessian的临时变量，用来存储交换了行之后，还没有交换列得到的hessain，所以行是最终的维度，但是列仍然是原来的hessian的维度
        Eigen::MatrixXd temp_hessian(to_keep_size + to_margin_size, raw_hessian_size);
        Eigen::MatrixXd res_hessian(to_keep_size + to_margin_size, to_keep_size + to_margin_size);
        
        //; 把要边缘化掉的变量所在的行调整到最前面，方便后面schur消元
        for (int i = 0; i < idx_to_margin.size(); i++)
        {
            res_residual.row(i) = mat_residual.row(idx_to_margin[i]);
            temp_hessian.row(i) = mat_hessian.row(idx_to_margin[i]);
        }

        //; 把要保留下来的行依次放到后面
        for (int i = 0; i < idx_to_keep.size(); i++)
        {
            res_residual.row(i + to_margin_size) = mat_residual.row(idx_to_keep[i]);
            temp_hessian.row(i + to_margin_size) = mat_hessian.row(idx_to_keep[i]);
        }

        //; 这里又赋值列，可以看出来上面的temp_hessian就是为了调整行的顺序使用的
        for (int i = 0; i < idx_to_margin.size(); i++)
        {
            res_hessian.col(i) = temp_hessian.col(idx_to_margin[i]);
        }

        for (int i = 0; i < idx_to_keep.size(); i++)
        {
            res_hessian.col(i + to_margin_size) = temp_hessian.col(idx_to_keep[i]);
        }

        if (m_if_enable_debug)
        {
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_hessian_A_new.txt", res_hessian);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_residual_B_new.txt", res_residual);
        }

        int m = to_margin_size;
        int n = to_keep_size;

        // 舒尔补
        Eigen::MatrixXd Amm_inv, Amm;
        Eigen::MatrixXd A, b;
        Amm = 0.5 * (res_hessian.block(0, 0, m, m) + res_hessian.block(0, 0, m, m).transpose());
        Amm_inv = Amm.ldlt().solve(Eigen::MatrixXd::Identity(m, m));

        Eigen::SparseMatrix<double> bmm = res_residual.segment(0, m).sparseView();
        Eigen::SparseMatrix<double> Amr = res_hessian.block(0, m, m, n).sparseView();
        Eigen::SparseMatrix<double> Arm = res_hessian.block(m, 0, n, m).sparseView();
        Eigen::SparseMatrix<double> Arr = res_hessian.block(m, m, n, n).sparseView();
        Eigen::SparseMatrix<double> brr = res_residual.segment(m, n).sparseView();
        Eigen::SparseMatrix<double> Amm_inv_spm = Amm_inv.sparseView();
        Eigen::SparseMatrix<double> Arm_Amm_inv = (Arm)*Amm_inv_spm;

        A = (Arr - Arm_Amm_inv * Amr).toDense();
        b = (brr - Arm_Amm_inv * bmm).toDense();
        //; 这里作者用的方式更加简单了，用了llt的方法，即 A = L*L', 而我们要的是A = J'*J, 所以matrixL = L = J'
        m_linearized_jacobians = A.llt().matrixL().transpose();
        if (1)
        {
            const double eps = 1e-15;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);  //; 又用了特征值分解
            Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
            Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

            Eigen::VectorXd S_sqrt = S.cwiseSqrt();
            Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

            //; A = J'*J = V*S*V', 对S进行LLT分解有S = L*L', 则A = V*L*L'*V' = (VL)*(VL)'= J'*J, 即J = (VL)' = L'*V' = L*V'
            //; 边缘化为了实现对剩下参数块的约束，为了便于一起优化，就抽象成了残差和雅克比的形式，这样也形成了一种残差约束
            m_linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
            m_linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
        }

        if (m_if_enable_debug)
        {
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_jac_new.txt", m_linearized_jacobians);
            Common_tools::save_matrix_to_txt("/home/ziv/temp/mar_linearized_res_new.txt", m_linearized_residuals);
        }
    }

    /**
     * @brief 
     * 
     * @param[in] mat_hessian 
     * @param[in] mat_res 
     * @param[in] visual_size : 视觉路标点的维数
     */
    void margin_oldest_frame(const Eigen::MatrixXd &mat_hessian, const Eigen::MatrixXd &mat_res, int visual_size)
    {
        m_margin_flag = 0;
        int para_size = mat_hessian.cols();   //; 所有状态变量的维度
        int visual_start_idx = para_size - visual_size;  //; 视觉路标点的起始索引
        idx_to_margin.clear();
        idx_to_keep.clear();
        int mar_pos = 0;

        idx_to_margin.reserve(1000);
        idx_to_keep.reserve(1000);

        // Step 1 划掉前15的变量，也就是最老帧的状态变量
        for (int idx = mar_pos; idx < mar_pos + 15; idx++)
        {
            idx_to_margin.push_back(idx);
        }

        // Step 2 要保留下的变量，注意：边缘化求解的时候保留下的变量一定是和要边缘化掉的状态变量有关系的哪些变量
        // 实际上是边缘化的结果对这些变量有约束，所以要把这些变量放到边缘化的方程组中。而其他变量和最老帧没有关系，
        // 那么即使放到边缘化的方程组中，最后求解的结果对这些状态变量的约束也是0，所以这样只会徒增计算量，并没有其他帮助
        // Pose[1] Speed_bias[1]
        //; 思考：为什么保留下的变量第一帧是15维？
        //; 解答：因为第0帧通过IMU预积分和第1帧之间构成的是位姿速度、零偏两种约束，所以一共是15维
        for (int idx = mar_pos + 15; idx < mar_pos + 30; idx++)
        {
            //; index : 15 - 30
            idx_to_keep.push_back(idx);
        }

        //! 问题：lidar先验位姿为什么全都加进去？边缘化掉最老帧，这个lidar先验只会对最老帧有约束啊？所以感觉这里的操作是没有必要的
        //! 解答：其实并不是考虑这个原因，而是考虑和第0帧有共视关系的那些关键帧。原来的VINS的写法是利用第0帧的地图管理器
        //!      遍历寻找和它共视的关键帧，但是这样比较麻烦，而且一般运动不是很剧烈的话，第0帧基本上和滑窗中的其他帧都有共视
        //!      关系，所以这里作者就简化了写法，不判断是否和第0帧有共视关系了，直接一股脑全部加进去。
        // Pose[2], Pose[3], ... , Pose[WINDOW_SIZE]
        for (int pose_idx = 2; pose_idx < WINDOW_SIZE + 1; pose_idx++)
        {
            //; 30-36, 45-51, ...
            for (int idx = pose_idx * 15; idx < pose_idx * 15 + 6; idx++)
            {
                idx_to_keep.push_back(idx);
            }
        }

        // Extrinsic + Td
        //; 外参和时间延时：通过视觉重投影、以及IMU预积分起到作用
        for (int idx = visual_start_idx - 7; idx < visual_start_idx; idx++)
        {
            idx_to_keep.push_back(idx);
        }

        //; 这个函数通过判断hessian矩阵中哪一块不是0，来判断3d点是否是和第0帧的视觉共视点，然后会更新index_to_margin这个变量
        std::vector<int> related_visual_idx = find_related_visual_landmark_idx(mat_hessian, idx_to_margin, mar_pos, visual_size);
        if (m_if_enable_debug)
        {
            cout << "=======Mar Old==========" << endl;
            cout << "Related total visual landmark: " << idx_to_margin.size() - 15 << endl;
            cout << "Total margin size = " << idx_to_margin.size() << endl;
            cout << "Total keep size = " << idx_to_keep.size() << endl;
            printf_vector(idx_to_margin, "To margins: ");
            printf_vector(idx_to_keep, "To keep: ");
        }

        //; idx_to_margin:被边缘化掉的变量的索引，包括最老帧、最老帧看到的地图点
        //; idx_to_keep:和最老帧相关的保留下来状态，包括第1帧位姿和bias、其他帧位姿、外参、时间延时
        marginalize(mat_hessian, mat_res, idx_to_margin, idx_to_keep);
    }

    void margin_second_new_frame(const Eigen::MatrixXd &mat_hessian, const Eigen::MatrixXd &mat_res, int visual_size)
    {
        m_margin_flag = 1;
        int para_size = mat_hessian.cols();
        int visual_start_idx = para_size - visual_size;
        int mar_pos = (WINDOW_SIZE - 1) * 15;
        std::vector<int> idx_to_margin, idx_to_keep;
        idx_to_margin.clear();
        idx_to_keep.clear();
        idx_to_margin.reserve(1000);
        idx_to_keep.reserve(1000);
        // Pose[WINDOW_SIZE-1]
        for (int idx = mar_pos; idx < mar_pos + 6; idx++)
        {
            idx_to_margin.push_back(idx);
        }

        // Pose[0] Speed_bias[0]
        for (int idx = 0; idx < 15; idx++)
        {
            idx_to_keep.push_back(idx);
        }

        // Pose[1], Pose[2], ... , Pose[WINDOW_SIZE-1]
        for (int pose_idx = 1; pose_idx < WINDOW_SIZE - 1; pose_idx++)
        {
            for (int idx = pose_idx * 15; idx < pose_idx * 15 + 6; idx++)
            {
                idx_to_keep.push_back(idx);
            }
        }

        for (int idx = visual_start_idx - 7; idx < visual_start_idx; idx++)
        {
            idx_to_keep.push_back(idx);
        }

        if (m_if_enable_debug)
        {
            cout << "=======Mar Last==========" << endl;
            cout << "Related total visual landmark: " << 0 << endl;
            cout << "Total margin size = " << idx_to_margin.size() << endl;
            cout << "Total keep size = " << idx_to_keep.size() << endl;
            printf_vector(idx_to_margin, "To margins: ");
            printf_vector(idx_to_keep, "To keep: ");
        }

        marginalize(mat_hessian, mat_res, idx_to_margin, idx_to_keep);
    }
};