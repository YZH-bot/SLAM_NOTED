# SLAM_NOTED
- [SLAM\_NOTED](#slam_noted)
  - [FLOAM](#floam)
  - [FAST-LIO2](#fast-lio2)
  - [direct\_lidar\_odemetry](#direct_lidar_odemetry)
  - [COIN-LIO](#coin-lio)
  - [fast\_gicp](#fast_gicp)
  - [LIO-SAM](#lio-sam)
  - [direct\_inertial\_lidar\_odemetry](#direct_inertial_lidar_odemetry)
  - [VINS-mono](#vins-mono)
  - [FAST\_LIO\_LOCALIZATION](#fast_lio_localization)
  - [FAST\_LIO\_Localization\_QN](#fast_lio_localization_qn)
  - [open\_vins](#open_vins)


该repo主要记录对目前SLAM算法的注释版本，包括：
## FLOAM
- ✔ **FLOAM**：[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/floam)

主要流程：
<img src="./notes/floam/floam.png" height="360">

## FAST-LIO2
- ✔ **FAST-LIO2**：[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/FAST_LIO)

## direct_lidar_odemetry
- ✔ **direct_lidar_odemetry**：[📖[文献解读]](https://zhuanlan.zhihu.com/p/677991232)$~~~$[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/direct_lidar_odometry)
  
## COIN-LIO
✔ **COIN-LIO**：[📖[文献解读]](https://zhuanlan.zhihu.com/p/697885897)$~~~$[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/coin-lio)$~~~$[📖[Original Repository ICRA 2024]](https://github.com/ethz-asl/COIN-LIO)
- **总结分析**：
  - **优点**：有效的隧道退化对抗方式，利用 LiDAR 的 intensity image 提取 patch 最小化光度误差来补偿退化方向上的信息缺失;
  - **缺点**：对 LiDAR 线束要求比较高，原作者使用的是128线Ouster雷达，可以提供比较密集的强度图像，本人使用40线雷达部署确实能有效抑制退化，但是垂直 LiDAR 方向， 由于 LiDAR 本身线束的稀疏性，会引入些许误差，并且随着距离增加，这个误差会越大。

- **退化检测效果**：
<br>
<p align="center">
    <img src="./notes/coin-lio/degeneration direction.gif" alt="drawing" width="360"/>
    <img src="./notes/coin-lio/degeneration direction2.gif" alt="drawing" width="360"/>
</p>



## fast_gicp
- **fast_gicp**: [code解析详情：fastgicp](https://github.com/YZH-bot/SLAM_NOTED/tree/master/fast_gicp) [working on]

## LIO-SAM
- **LIO-SAM**：[working on]

## direct_inertial_lidar_odemetry
- **direct_inertial_lidar_odemetry**：[todo]

## VINS-mono
- **VINS-mono**：[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/VINS-Mono-Learning)$~~~$[📖[Add details upon the Repository VINS-Mono-Learning, thanks to ManiiXu]](https://github.com/ManiiXu/VINS-Mono-Learning)

## FAST_LIO_LOCALIZATION
- **FAST_LIO_LOCALIZATION**：[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/FAST_LIO_LOCALIZATION)$~~~$[📖[Original Repository]](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION)
- **总结分析**：
  - 这个库的思路比较简单，就简单说一下，其实关键就是三个 `.py` 节点，详细看一下注释，目前我觉得这种做法不太稳定，对于 `fast-lio2` 本身没有改进，只是低频估计初始的 `map-to-odom` 的变换，将这个变换作用到 fast-lio2 的结果上。
  
## FAST_LIO_Localization_QN
- **FAST_LIO_Localization_QN**：[⌨[code详细注释]](https://github.com/YZH-bot/SLAM_NOTED/tree/master/FAST_LIO_Localization_QN)$~~~$[📖[Original Repository]](https://github.com/engcang/FAST-LIO-Localization-QN)
- **总结分析**：
  - 这个库的思路和 `FAST_LIO_LOCALIZATION` 不完全一样，维护了历史关键帧，通过搜寻当前关键帧最近关联帧（意味着没有回环检测，就是欧式距离法），然后进行修正，不过都对 `fast-lio2` 本身没有改进，也只是低频估计和更新初始的 `map-to-odom` 的变换（没有实现 `rviz` 设置起点功能），将这个变换作用到 `fast-lio2` 的结果上。

## open_vins
- **open_vins**：[working on]$~~~$[📖[Original Repository]](https://github.com/rpng/open_vins)