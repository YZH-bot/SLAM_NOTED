# SLAM_NOTED
该repo主要记录对目前SLAM算法的注释版本，包括：
- ✔ **FLOAM**：[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/floam)

主要流程：
<img src="./notes/floam/floam.png" height="360">

- ✔ **direct_lidar_odemetry**：[📖[文献解读]](https://zhuanlan.zhihu.com/p/677991232)$~~~$[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/direct_lidar_odometry)
- ✔ **COIN-LIO**：[📖[文献解读]](https://zhuanlan.zhihu.com/p/697885897)$~~~$[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/coin-lio)
  - **优点**：有效的隧道退化对抗方式，利用 LiDAR 的 intensity image 提取 patch 最小化光度误差来补偿退化方向上的信息缺失;
  - **缺点**：对 LiDAR 线束要求比较高，原作者使用的是128线Ouster雷达，可以提供比较密集的强度图像，本人使用40线雷达部署确实能有效抑制退化，但是垂直 LiDAR 方向， 由于 LiDAR 本身线束的稀疏性，会引入些许误差，并且随着距离增加，这个误差会越大。

- **fast_gicp**: [code解析详情：fastgicp](https://github.com/YZH-bot/SLAM_NOTED/tree/master/fast_gicp) [working on]
- **LIO-SAM**：[working on]
- **direct_inertial_lidar_odemetry**：[todo]
- **VINS-mono**：[todo]

