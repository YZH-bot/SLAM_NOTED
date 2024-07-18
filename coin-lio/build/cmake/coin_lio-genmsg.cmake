# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "coin_lio: 1 messages, 0 services")

set(MSG_I_FLAGS "-Icoin_lio:/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(coin_lio_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" NAME_WE)
add_custom_target(_coin_lio_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "coin_lio" "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(coin_lio
  "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/coin_lio
)

### Generating Services

### Generating Module File
_generate_module_cpp(coin_lio
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/coin_lio
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(coin_lio_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(coin_lio_generate_messages coin_lio_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" NAME_WE)
add_dependencies(coin_lio_generate_messages_cpp _coin_lio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(coin_lio_gencpp)
add_dependencies(coin_lio_gencpp coin_lio_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS coin_lio_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(coin_lio
  "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/coin_lio
)

### Generating Services

### Generating Module File
_generate_module_eus(coin_lio
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/coin_lio
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(coin_lio_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(coin_lio_generate_messages coin_lio_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" NAME_WE)
add_dependencies(coin_lio_generate_messages_eus _coin_lio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(coin_lio_geneus)
add_dependencies(coin_lio_geneus coin_lio_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS coin_lio_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(coin_lio
  "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/coin_lio
)

### Generating Services

### Generating Module File
_generate_module_lisp(coin_lio
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/coin_lio
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(coin_lio_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(coin_lio_generate_messages coin_lio_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" NAME_WE)
add_dependencies(coin_lio_generate_messages_lisp _coin_lio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(coin_lio_genlisp)
add_dependencies(coin_lio_genlisp coin_lio_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS coin_lio_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(coin_lio
  "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/coin_lio
)

### Generating Services

### Generating Module File
_generate_module_nodejs(coin_lio
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/coin_lio
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(coin_lio_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(coin_lio_generate_messages coin_lio_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" NAME_WE)
add_dependencies(coin_lio_generate_messages_nodejs _coin_lio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(coin_lio_gennodejs)
add_dependencies(coin_lio_gennodejs coin_lio_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS coin_lio_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(coin_lio
  "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/coin_lio
)

### Generating Services

### Generating Module File
_generate_module_py(coin_lio
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/coin_lio
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(coin_lio_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(coin_lio_generate_messages coin_lio_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/media/yzh/YZH2/Projects/SLAM_NOTED/coin-lio/msg/Pose6D.msg" NAME_WE)
add_dependencies(coin_lio_generate_messages_py _coin_lio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(coin_lio_genpy)
add_dependencies(coin_lio_genpy coin_lio_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS coin_lio_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/coin_lio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/coin_lio
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(coin_lio_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/coin_lio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/coin_lio
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(coin_lio_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/coin_lio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/coin_lio
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(coin_lio_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/coin_lio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/coin_lio
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(coin_lio_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/coin_lio)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/coin_lio\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/coin_lio
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(coin_lio_generate_messages_py geometry_msgs_generate_messages_py)
endif()
