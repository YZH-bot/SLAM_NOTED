# GICP 算法（ICP汇总）

[TOC]

## ICP 算法介绍 
|                 point-to-point icp                  |                 point-to-plane icp                  |                                      point-to-plane icp                                      |
| :-------------------------------------------------: | :-------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| <img src="./imgs/point2point_icp.png" height="200"> | <img src="./imgs/point2plane_icp.png" height="200"> |                           <img src="./imgs/gicp.png" height="200">                           |
|     优化源点集与目标点集中对应点之间位置的偏差      |       优化源点集与目标点集中对应点平面的偏差        | 每个点都服从高斯分布，<br> 也就是说每对对应点不需要完美对应上，<br> 尽可能提高对应的概率即可 |

## GICP 原理：
假定我们找到了一对匹配点 $\pmb{a}_i$ 和 $\pmb{b}_i$ 的最优匹配为 $\pmb{T}$，那么：

$$
\begin{equation}
    \hat{\pmb{b}}_i = \pmb{T}\hat{\pmb{a}}_i
\end{equation}
$$

定义它们之间的距离残差为 $d_i^{(\pmb{T})} = \hat{\pmb{b}}_i - \pmb{T^*}\hat{\pmb{a}}_i$，假设 $\pmb{a}_i$ 和 $\pmb{b}_i$ 独立，则：

$$
\begin{equation}\begin{split}
d_{i}^{(\mathbf{T})}& \sim\mathcal{N}\left(\hat{b_i}-\left(\mathbf{T}\right)\hat{a_i},C_i^B+\left(\mathbf{T}\right)C_i^A\left(\mathbf{T}\right)^T\right)  \\
&=\mathcal{N}\left(0,C_i^B+\left(\mathbf{T}\right)C_i^A\left(\mathbf{T}\right)^T\right)
\end{split}\end{equation}
$$

GICP通过最大似然估计，找到置信最高的变换矩阵 $\pmb{T}$：

$$
\begin{equation}\begin{aligned}
\mathbf{T}& =\mathop{\arg\max}\limits_{\mathbf{T}}\prod_ip\left(d_i^{(\mathbf{T})}\right)  \\
&={\mathop{\arg\max}\limits_{\mathbf{T}}}\sum_i\log\left(p\left(d_i^{(\mathbf{T})}\right)\right) \\
&=\mathop{\arg\max}\limits_{\mathbf{T}}\sum_i {d_i^{(\mathbf{T})}}^T\left(C_i^B+\mathbf{T}C_i^A\mathbf{T}^T\right)^{-1}d_i^{(\mathbf{T})}
\end{aligned}\end{equation}
$$

这是GICP中最核心的优化函数。 $\left(C_i^B+\mathbf{T}C_i^A\mathbf{T}^T\right)^{-1}$ 是source点云与target点云对应点的局部协方差矩阵，这个项会影响对应残差在优化过程中的权重，加快收敛速度和精度。<br>
此外:
* 点到点匹配 ➡ $C_i^B = I,\space C_i^A = 0$
* 点到面匹配 ➡ $C_i^B = (\pmb{n}_i\pmb{n}_i^T)^{-1},\space C_i^A = 0$
  * 其中 $\pmb{n}_i\pmb{n}_i^T$ 是一个正交投影矩阵 ( $P = P^T, P^2 = P$)  ，相当于把点与点之间的距离投影到法向量上，得到点到面的距离。
* 面到面匹配 ➡ GICP

## 附录
前置知识，多元高斯分布的概率密度函数：

$$
\begin{equation}\mathrm{f_x(x_1,...,x_k)=\frac1{\sqrt{(2\pi)^k|\Sigma|}}e^{-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)},|\Sigma|\triangleq det\Sigma}\end{equation}
$$

公式(3)目标函数的推导:

$$
\begin{equation}
\begin{aligned}
\text{T}&=\mathop{\arg\max}\limits_{\mathbf{T}}\prod_{\mathrm{i}}\text{p}({d}_{\mathrm{i}}^{(\mathbf{T})})  \\
&=\mathop{\arg\max}\limits_{\mathbf{T}}\sum_{\mathrm{i}}\log({p(d_i^\mathbf{(T)})}) \\
&=\mathop{\arg\max}\limits_{\mathbf{T}}\sum_{\mathrm{i}}{ \log ( \frac 1 { \sqrt { ( 2 \pi ) ^ \mathrm{k}|\mathrm{C}_i^\mathrm{B}+\mathbf{T}\mathrm{C}_{i}^\mathrm{A}\mathbf{T}^\mathrm{T}|}})} \\
&-\frac12({d_i^\mathbf{(T)}-(\hat{b_i}-\mathbf{T}\hat{a_i})})^{{T}}({\mathrm{C_i^B}+\mathbf{T}\mathrm{C_i^A}\mathbf{T}^T})^{-1}({d_i^{(T)}-(\hat{b_i}-\mathbf{T}\hat{a_i})}) \\
&={\mathop{\arg\max}\limits_{\mathbf{T}}}\sum_\text{i}{ \log ( \frac 1 { \sqrt { ( 2 \pi ) ^ \mathrm{k}|\mathrm{C}_{i}^\mathrm{B}+\mathbf{T}\mathrm{C}_{i}^\mathrm{A}\mathbf{T}^\mathrm{T}|}})} \\
&-\frac12{d_i^{(\mathbf{T})^T}({\mathrm{C_i^B}+\mathbf{T}\mathrm{C_i^A}\mathbf{T}^T})^{-1}d_i^{(\mathbf{T})}} \\
&={\mathop{\arg\max}\limits_{\mathbf{T}}}\sum_\text{i}{ - \frac 1 2 }{d_i^{(\mathbf{T})}}^{{T}}({\mathrm{C_i^B}+\mathbf{T}\mathrm{C_i^A}\mathbf{T}^T})^{-1}{d_i^{(\mathbf{T})}} \\
&=\mathop{\arg\min}\limits_{\mathbf{T}}\sum_\text{i}{ {d_i^{(\mathbf{T})}}}^{\mathrm{T}}(\mathrm{C_i^B}+\mathbf{T}\mathrm{C_i^A}\mathbf{T}^T)^{-1}{d_i^{(\mathbf{T})}}
\end{aligned}
\end{equation}
$$

## 优化推导
- 为了方便求导和加速运算，作者代码中将协方差权重部分 $(\mathrm{C_i^B}+\mathbf{T}\mathrm{C_i^A}\mathbf{T}^T)^{-1}$ 当作一个常数矩阵。 
<br>

- 求导部分使用的是左扰动形式，并且是用的是 $SE(3)$ 进行推导：

$$
\begin{equation}\begin{aligned}
\frac{\partial\left(\mathbf{Tp}\right)}{\partial\delta\boldsymbol{\xi}}& =\lim_{\delta\boldsymbol{\xi}\to\boldsymbol{0}}\frac{\exp\left(\delta\boldsymbol{\xi}^{\wedge}\right)\exp\left(\boldsymbol{\xi}^{\wedge}\right)\mathbf{p}-\exp\left(\boldsymbol{\xi}^{\wedge}\right)\mathbf{p}}{\delta\boldsymbol{\xi}}  \\
&=\lim_{\delta\boldsymbol{\xi}\to\boldsymbol{0}}\frac{\left(\mathbf{I}+\delta\boldsymbol{\xi}^{\wedge}\right)\exp\left(\boldsymbol{\xi}^{\wedge}\right)\mathbf{p}-\exp\left(\boldsymbol{\xi}^{\wedge}\right)\mathbf{p}}{\delta\boldsymbol{\xi}} \\
&=\lim_{\delta\boldsymbol{\xi}\to\boldsymbol{0}}\frac{\delta\boldsymbol{\xi}^{\wedge}\exp\left(\boldsymbol{\xi}^{\wedge}\right)\mathbf{p}}{\delta\boldsymbol{\xi}} \\
&=\lim_{\delta\boldsymbol{\xi}\to\boldsymbol{0}}\frac{\left[\begin{array}{cc}\delta\boldsymbol{\phi}^{\wedge}&\delta\boldsymbol{\rho}\\\mathbf{0}^{T}&0\end{array}\right]\left[\begin{array}{c}\mathbf{R}\mathbf{p}+\mathbf{t}\\1\end{array}\right]}{\delta\boldsymbol{\xi}} \\
&=\lim_{\delta\boldsymbol{\xi}\to\boldsymbol{0}}\frac{\left[\begin{array}{c}\delta\boldsymbol{\phi}^{\wedge}\left(\mathbf{R}\mathbf{p}+\mathbf{t}\right)+\delta\boldsymbol{\rho}\\\mathbf{0}^{T}\end{array}\right. ]}{[\delta\boldsymbol{\rho},\delta\boldsymbol{\phi}]^{T}}=\left[\begin{array}{cc}\mathbf{I}&-(\mathbf{R}\mathbf{p}+\mathbf{t})^{\wedge}\\\mathbf{0}^{T}&\mathbf{0}^{T}\end{array}\right]\triangleq(\mathbf{T}\mathbf{p})^{\odot}
\end{aligned}\end{equation}
$$

### Gaussian-Newton 优化
- **代码实现 FastGICP::linearize 函数**：通过多线程对H矩阵和b矩阵进行计算   此外，这是个虚函数，每个子类需要自己进行实现
![proctime](../../fast_gicp/doc/fast_gicp_linearize_func.png)
## REFERENCE
[[1] https://blog.csdn.net/keineahnung2345/article/details/122836492](https://blog.csdn.net/keineahnung2345/article/details/122836492)
[[2] https://blog.csdn.net/xinxiangwangzhi_/article/details/125236953](https://blog.csdn.net/xinxiangwangzhi_/article/details/125236953)