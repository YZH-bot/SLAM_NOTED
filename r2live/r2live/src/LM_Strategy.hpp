#pragma once
#include "tools_ceres.hpp"
#include "tools_eigen.hpp"

const double MIMIMUM_DELTA = 1e-15;

inline Eigen::SparseMatrix<double> schur_complement_woodbury_matrix(Eigen::SparseMatrix<double> &mat_A, Eigen::SparseMatrix<double> &mat_U,
                                                                    Eigen::SparseMatrix<double> &mat_C_inv, Eigen::SparseMatrix<double> &mat_V)
{
    // https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    Eigen::SparseMatrix<double> mat_A_inv = mat_A.toDense().completeOrthogonalDecomposition().pseudoInverse().sparseView();
    Eigen::SparseMatrix<double> mat_mid_inv = (mat_C_inv + mat_V * mat_A_inv * mat_U).toDense().completeOrthogonalDecomposition().pseudoInverse().sparseView();
    return mat_A_inv - mat_A_inv * mat_U * mat_mid_inv * mat_V * mat_A_inv;
}

//; 使用schur补求解大型稀疏矩阵，方法和十四讲P249-250一样，就是schur消掉H矩阵右上角，先求稠密矩阵构成的方程
inline Eigen::SparseMatrix<double> sparse_schur_solver(const Eigen::SparseMatrix<double> &mat_H, const Eigen::SparseMatrix<double> &vec_JtX, int dense_block_size, int solver_type = 0)
{
    //Website: http://ceres-solver.org/nnls_solving.html#equation-hblock
    Eigen::SparseMatrix<double> vec_X;
    int mat_C_blk_size = mat_H.cols() - dense_block_size;  //; 稀疏部分的大小

    //; 下面这些矩阵的赋值和十四讲中写的一样，见十四讲第249-250页
    Eigen::SparseMatrix<double> mat_B, mat_E, mat_Et, mat_C, mat_v, mat_w, mat_C_inv, mat_I, mat_S, mat_S_inv, mat_E_C_inv;
    Eigen::SparseMatrix<double> mat_dy, mat_dz;
    mat_B = mat_H.block(0, 0, dense_block_size, dense_block_size);  //; 左上角稠密块  
    mat_E = mat_H.block(0, dense_block_size, dense_block_size, mat_C_blk_size);  //; 右上角即将被消除的块
    mat_C = mat_H.block(dense_block_size, dense_block_size, mat_C_blk_size, mat_C_blk_size);  //; 右下角稀疏块
Start_mat_c:  //! 这是什么神仙写法？
    mat_v = vec_JtX.block(0, 0, dense_block_size, 1);  //; 右侧b向量的对应稠密块的部分v
    mat_w = vec_JtX.block(dense_block_size, 0, mat_C_blk_size, 1);  //; 右侧b向量的对应稀疏块的部分w
    mat_Et = mat_E.transpose();
    mat_C_inv.resize(mat_C_blk_size, mat_C_blk_size);  //; 右下角稀疏矩阵的逆
    
    //; 共轭梯度法最小二乘？
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    int if_mat_C_invertable = 1;
    for (int i = 0; i < mat_C_blk_size; i++)
    {
        //; 如果右下角稀疏矩阵块的有的数值太小，那么它求逆的结果会非常大，导致数值不稳定，所以这里直接把逆赋值为0，并且置位不稳定标志
        if (fabs(mat_C.coeff(i, i)) < MIMIMUM_DELTA)
        {
            mat_C_inv.insert(i, i) = 0.0;
            if_mat_C_invertable = 0;
        }
        else
        {
            //; 求逆直接取倒数
            mat_C_inv.insert(i, i) = 1.0 / mat_C.coeff(i, i);
        }
    }

    mat_E_C_inv = mat_E * mat_C_inv;  //; E * C^-1
    if (if_mat_C_invertable)
    {
        mat_S = mat_B - mat_E_C_inv * mat_Et;
        // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
        //; LDLT分解就是L * D * L^T, 其中L是单位下三角矩阵，D是对角矩阵，就是数值计算方法中的求解线性方程的方法
        Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
        solver.compute(mat_S);
        //; 如果LTLT分解成功了，那么就可以用分解后的结果求解方程组
        if (solver.info() == Eigen::Success)
        {
            mat_dy = solver.solve((mat_v - mat_E_C_inv * mat_w));
        }
        else
        {
            mat_dy = mat_S.toDense().completeOrthogonalDecomposition().solve((mat_v - mat_E_C_inv * mat_w).toDense()).sparseView();
        }
    }
    else
    {
        cout << ANSI_COLOR_CYAN_BOLD << "Unstable update, perfrom schur_complement_woodbury_matrix" << ANSI_COLOR_RESET << endl;
        mat_S_inv = schur_complement_woodbury_matrix(mat_B, mat_E, mat_C, mat_Et);
        mat_dy = mat_S_inv * (mat_v - mat_E_C_inv * mat_w);
    }

    //; 求解稀疏部分的解，在十四讲第250页
    mat_dz = mat_C_inv * (mat_w - mat_Et * mat_dy);
    Eigen::Matrix<double, -1, -1> vec_X_dense;
    vec_X_dense.resize(vec_JtX.rows(), vec_JtX.cols());  //; vec_JtX.cols()就是1，这里也可以直接写成1吧
    vec_X_dense.block(0, 0, dense_block_size, 1) = mat_dy.toDense();
    vec_X_dense.block(dense_block_size, 0, mat_C_blk_size, 1) = mat_dz.toDense();
    vec_X = vec_X_dense.sparseView();  //; 再把最后求解的结果转成spare的形式，也就是系数表示

    return vec_X;
}

//; LM算法求解
struct LM_trust_region_strategy
{
public:
    double radius_ = 1e-4;
    double max_radius_;
    const double min_diagonal_ = 1e-15;
    const double max_diagonal_ = 1e15;
    double decrease_factor_ = 2.0;
    bool reuse_diagonal_;
    Eigen::SparseMatrix<double> gradient, hessian, last_step;
    std::vector<double> cost_history, model_cost_changes_history, real_changes_history, step_quality_history, radius_history;

    LM_trust_region_strategy()
    {
        cost_history.reserve(10);
        model_cost_changes_history.reserve(10);
        real_changes_history.reserve(10);
        step_quality_history.reserve(10);
        radius_history.reserve(10);
    };

    ~LM_trust_region_strategy() = default;

    //; 传入残差和雅克比进行一次梯度下降的计算。这里声明为内联函数比避免函数调用，可以提高执行效率
    inline Eigen::SparseMatrix<double> compute_step(Eigen::SparseMatrix<double> &jacobian, Eigen::SparseMatrix<double> &residuals, int dense_block_size = 0)
    {
        Eigen::SparseMatrix<double> current_step;
        int residual_size = jacobian.rows();
        int parameter_size = jacobian.cols();
        double current_cost = residuals.norm();
        //; 计算 J^T*J*δx = -J^T*b
        gradient = jacobian.transpose() * residuals; //; J^T*b
        hessian = jacobian.transpose() * jacobian;   //; J^T*J

        // Step 下面这部分可以看从零手写VIO的课件中关于LM算法的部分，实现VIO课件中讲的是一样的，这里LM算法的阻尼因子系数采用的是Nielsen策略，
        // Step   也是g2o和ceres使用的策略
        if (cost_history.size() && model_cost_changes_history.size())
        {
            double real_cost_change = cost_history.back() - current_cost;
            //; step_quality就是比例因子，即上次迭代后真实的损失函数的下降量和线性化计算的损失函数下降量之间的比值
            double step_quality = real_cost_change / model_cost_changes_history.back();  
            real_changes_history.push_back(real_cost_change);
            step_quality_history.push_back(step_quality);

            //; 如果比例因子>0，说明真实损失函数确实下降了，那么要减小阻尼因子，增大步长，加快收敛
            if (step_quality > 0) // Step is good,
            {
                //; radius_就是阻尼因子，Nielsen策略
                radius_ = radius_ * std::max(1.0 / 3.0, 1.0 - pow(2.0 * step_quality - 1.0, 3));
                decrease_factor_ = 2.0;
            }
            else
            {
                radius_ = radius_ * decrease_factor_;
                decrease_factor_ *= 2;
                current_step = -1 * last_step;
                last_step.setZero();
                radius_history.push_back(radius_);
                cost_history.push_back(current_cost);
            }
        }
        //; 如果是第一次执行的话，那么信赖域半径按照如下取值
        else
        {
            //; 1.这里取值方法就是VIO课件中写的方法，也就是找J^T*J对角线上的最大值，这个值和J^T*J的特征值是同数量级的，此时再乘以一个比例因子1e-6
            //;   作为最后信赖与半径的取值。但是这里又开了根号，和VIO课件中有点不一样
            //; 2.另外下面的min和max就是为了增加数值稳定性，防止太大或者太小的数出现
            radius_ = 1e-6 * sqrt(std::min(std::max(hessian.diagonal().maxCoeff(), min_diagonal_), max_diagonal_));
        }

        radius_history.push_back(radius_);
        cost_history.push_back(current_cost);

        Eigen::SparseMatrix<double> mat_D(parameter_size, parameter_size);

        //; mat_D应该是LM算法由于存在信赖区域半径，从而在H矩阵中附加的部分
        mat_D.setZero();
        for (int i = 0; i < parameter_size; i++)
        {
            //; 这里的操作还不是简单的加了 miu_I，而是每一个位置的系数都不同？
            mat_D.coeffRef(i, i) = sqrt(std::min(std::max(hessian.coeff(i, i), min_diagonal_), max_diagonal_) * radius_);
        }

        Eigen::SparseMatrix<double> hessian_temp = hessian + mat_D; //; hessian矩阵增加信赖区域部分
        Eigen::SparseMatrix<double> jacobian_mul_step, model_cost_mat;
        //; 如果说指定了稠密块，那么说明这是一个稀疏矩阵，那么下面就要使用舒尔补的算法加速稀疏矩阵的求解
        if (dense_block_size)
        {
            //; 对大型系数矩阵，利用系数矩阵舒尔补的解法就求解 H * δx = -b的问题
            //; 解得的current_step就是δx，也就是状态变量的变化值
            current_step = sparse_schur_solver(hessian_temp, -gradient, dense_block_size);
        }
        else
        {
            current_step = (hessian_temp.toDense().completeOrthogonalDecomposition().solve(-gradient.toDense())).sparseView();
        }

        jacobian_mul_step = jacobian * current_step;

        //; 下面这部分就是在算模型的残差变化，VIO的PPT中第17页的公式第一行，所以它并没有怎么进行化简, 也就是没有化成SBA的公式中的写法
        //; 但是实际上SBA公式只是看起来简单点，实际计算还是要算 J^T * b, 所以计算量没有下降
        model_cost_mat = (-jacobian_mul_step.transpose() * (residuals + jacobian_mul_step / 2.0));
        double model_cost_chanage = model_cost_mat.norm(); 
        model_cost_changes_history.push_back(model_cost_chanage);

        if (model_cost_chanage >= 0)
        {
            last_step = current_step;
            return current_step;
        }
        //; 模型损失必然会 > 0!
        else
        {
            cout << ANSI_COLOR_RED_BOLD << "Model_cost_chanage = " << model_cost_chanage << " is negative, something error here, please check!!!" << endl;
            return current_step;
        }
    }

    void printf_history()
    {
        if (cost_history.size())
        {
            cout << "===== History cost =====" << endl;
            cout << " Iter | Cost | M_cost | R_cost | S_Q | Rad " << endl;
            cout << "[ " << 0 << "] " << cost_history[0] << " --- " << endl;
            for (int i = 0; i < cost_history.size() - 1; i++)
            {
                cout << "[ " << i << "] " << cost_history[i + 1] << " | "
                     << model_cost_changes_history[i] << " | "
                     << real_changes_history[i] << " | "
                     << step_quality_history[i] << " | "
                     << radius_history[i]
                     << endl;
            }
        }
    }
};
