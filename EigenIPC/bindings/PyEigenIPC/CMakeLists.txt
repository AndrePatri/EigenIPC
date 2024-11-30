cmake_minimum_required(VERSION 3.5)

add_compile_options(-std=c++17)

option(WITH_ROS_BRIDGE "Compile and install custom ros bridge extensions" TRUE)

if(${WITH_ROS_BRIDGE})

    message(STATUS "Ros bridge extension will be built and installed.")
    add_definitions(-DWITH_ROS_BRIDGE)

endif()

# Setting default visibility to hidden
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
# Helps in reducing the possibility of symbol clashes and also minimizes
# the chance of accidentally exporting symbols that aren't intended to be
# part of your library's public API

set(PyBindName "PyEigenIPC")
set(PyLayerName "PySharsor")

# set(BINDINGS_BINARY_DIR ${CMAKE_BINARY_DIR}/bindings/${LIBRARY_NAME}/${PyBindName})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} include/ src/)

find_package(pybind11 REQUIRED)

find_package(Eigen3 3.3 REQUIRED)

message(STATUS "Compiling python bindings for ${LIBRARY_NAME}. Codename: ${PyBindName}")

# to get PYTHON_SITE
execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_python_lib(prefix='', plat_specific=True))"
      OUTPUT_VARIABLE PYTHON_SITE
      OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library("${PyBindName}Libs" STATIC src/PyStringTensor.cpp
                                       src/PyServer.cpp
                                       src/PyClient.cpp
                                       src/PyProducer.cpp
                                       src/PyConsumer.cpp
                                       )
target_link_libraries("${PyBindName}Libs" PUBLIC EigenIPC PRIVATE pybind11::module)
# set_target_properties("${PyBindName}Libs" PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${BINDINGS_BINARY_DIR}
#     RUNTIME_OUTPUT_DIRECTORY ${BINDINGS_BINARY_DIR})

pybind11_add_module(${PyBindName} PyEigenIPC.cpp)
target_link_libraries(${PyBindName} PRIVATE "${PyBindName}Libs" pybind11::module)
# set_target_properties(${PyBindName} PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${BINDINGS_BINARY_DIR}
#     RUNTIME_OUTPUT_DIRECTORY ${BINDINGS_BINARY_DIR})

# install python bindings
install(TARGETS ${PyBindName}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE}/${LIBRARY_NAME})

install(FILES __init__.py DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE}/${LIBRARY_NAME})

# install python wrappers
install(
    DIRECTORY wrappers/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE}/${LIBRARY_NAME}/${PyLayerName}/wrappers
)

# if(${WITH_ROS_BRIDGE})

#     # compile custom messages
#     find_package(catkin QUIET COMPONENTS
#         std_msgs
#     )

#     if(catkin_FOUND)

#         message(STATUS "Catkin found. Compiling custom messages for ROS1.")

#         catkin_package(
#             CATKIN_DEPENDS std_msgs
#         )

#         file(GLOB_RECURSE ROS1_MESSAGES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/extensions/ros_bridge/msgs/ros1/ *.msg)
#         foreach(MSG_FILE ${ROS1_MESSAGES})
#             add_message_files(
#                 FILES
#                 ${CMAKE_CURRENT_SOURCE_DIR}/extensions/ros_bridge/msgs/ros1/${MSG_FILE}
#             )
#         endforeach()

#         generate_messages(
#             DEPENDENCIES
#             std_msgs
#         )

#         install(
#             DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/extensions/ros_bridge/msgs/ros1/
#             DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/msg
#             FILES_MATCHING PATTERN "*.msg"
#         )

#         message(STATUS "Custom messages for ROS1 compiled and installed.")

#     else()

#         message(WARNING "Catkin not found. ROS1 custom messages will not be compiled.")
        
#     endif()

#     # Check if ament_cmake is available for ROS2
#     find_package(ament_cmake QUIET)

#     if(ament_cmake_FOUND)

#         message(STATUS "Ament CMake found. Compiling custom messages for ROS2.")

#         file(GLOB_RECURSE ROS2_MESSAGES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/extensions/ros_bridge/msgs/ros2/ *.msg)

#         foreach(MSG_FILE ${ROS2_MESSAGES})
#             rosidl_generate_interfaces(${PyLayerName}
#                 "${CMAKE_CURRENT_SOURCE_DIR}/extensions/ros_bridge/msgs/ros2/${MSG_FILE}"
#             )
#         endforeach()

#         install(
#             DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/extensions/ros_bridge/msgs/ros2/
#             DESTINATION share/${PyLayerName}/msg
#             FILES_MATCHING PATTERN "*.msg"
#         )

#         message(STATUS "Custom messages for ROS2 compiled and installed.")

#     else()

#         message(WARNING "Ament CMake not found. ROS2 custom messages will not be compiled.")

#     endif()

# endif()

if(${WITH_ROS_BRIDGE})

    # install python extensions
    install(
        DIRECTORY extensions/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE}/${LIBRARY_NAME}/${PyLayerName}/extensions
    )

endif()

# add init file for package
install(FILES __init__.py DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE}/${LIBRARY_NAME}/${PyLayerName})
