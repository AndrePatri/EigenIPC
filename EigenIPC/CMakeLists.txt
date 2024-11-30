cmake_minimum_required(VERSION 3.10)

set(LIBRARY_NAME EigenIPC)
add_definitions("-DLIBRARY_NAME=${LIBRARY_NAME}")

project(${LIBRARY_NAME} VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# default to Release
if(NOT CMAKE_BUILD_TYPE)

  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")

    add_compile_definitions(IS_RELEASE=1)

endif()

# if in conda env, install it the environment
if (DEFINED ENV{CONDA_PREFIX})

    set(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX} CACHE PATH "Install prefix" FORCE)

endif()
# set(CMAKE_INSTALL_PREFIX "${PREFIX}" CACHE PATH "Installation prefix" FORCE)

find_package(pybind11 QUIET)

if(pybind11_FOUND)

    option(WITH_PYTHON "Compile and install python bindings" FALSE)

    if(${WITH_PYTHON})

        message(STATUS "Python bindings for ${LIBRARY_NAME} will be built and installed.")
        add_definitions(-DWITH_PYTHON) # can be used by preprocessor macros

    endif()

endif()

option(WITH_TESTS "Compile tests" FALSE)
if(${WITH_TESTS})

    message(STATUS "Tests for ${LIBRARY_NAME} will be built.")

endif()

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Eigen3 3.3 REQUIRED)

# static library for max compatibility and self-containment
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${LIBRARY_NAME} STATIC
    src/Server.cpp
    src/Client.cpp
    src/StringTensor.cpp
    src/MemUtils.hpp
    src/SharedMemConfig.hpp
    src/CondVar.cpp
    src/Producer.cpp
    src/Consumer.cpp
    include/${LIBRARY_NAME}/Journal.hpp
    include/${LIBRARY_NAME}/Helpers.hpp
    include/${LIBRARY_NAME}/ReturnCodes.hpp
    include/${LIBRARY_NAME}/MemDefs.hpp
)

# public headers
target_include_directories(${LIBRARY_NAME}
    PUBLIC
    $<INSTALL_INTERFACE:include>  # For installed headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> # private headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for local build
)

target_link_libraries(${LIBRARY_NAME} PUBLIC
    Eigen3::Eigen
    rt  # Link against the real-time library
    pthread
)

if(${WITH_TESTS})

    # Enable testing
    enable_testing()

    add_subdirectory(tests/)

endif()

# Compile python bindings
if(${WITH_PYTHON})

    add_subdirectory(bindings/PyEigenIPC/)

endif()

install(TARGETS  ${LIBRARY_NAME}
    EXPORT   ${LIBRARY_NAME}
    LIBRARY  DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT shlib
    ARCHIVE  DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT lib
    RUNTIME  DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT bin)

include(GNUInstallDirs)

# Mark cpp headers files for installation
install(DIRECTORY include/${LIBRARY_NAME}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
    FILES_MATCHING PATTERN "*.h*"
    PATTERN ".svn" EXCLUDE)

# export library so it can be used from external packages
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExportEigenIPC.cmake)