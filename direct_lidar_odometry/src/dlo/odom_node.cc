/************************************************************
 *
 * Copyright (c) 2022, University of California, Los Angeles
 *
 * Authors: Kenny J. Chen, Brett T. Lopez
 * Contact: kennyjchen@ucla.edu, btlopez@ucla.edu
 *
 ***********************************************************/

#include "dlo/odom.h"

void controlC(int sig) {

  dlo::OdomNode::abort();

}

int main(int argc, char** argv) {

  ros::init(argc, argv, "dlo_odom_node");
  ros::NodeHandle nh("~");

  // linux捕抓Ctrl+C，并执行controlC函数
  signal(SIGTERM, controlC);
  sleep(0.5);

  // 本文件看完之后进入OdomNode阅读具体实现
  dlo::OdomNode node(nh);
  // 开辟多个线程，保证数据流的畅通。如果不指定线程数或者线程数设置为0，它将在每个cpu内核开辟一个线程。
  ros::AsyncSpinner spinner(0);
  spinner.start();
  node.start();
  ros::waitForShutdown();

  return 0;

}
