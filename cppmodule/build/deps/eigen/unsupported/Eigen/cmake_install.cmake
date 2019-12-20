# Install script for directory: /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/AdolcForward"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/AlignedVector3"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/ArpackSupport"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/AutoDiff"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/BVH"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/EulerAngles"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/FFT"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/IterativeSolvers"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/KroneckerProduct"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/MatrixFunctions"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/MoreVectorization"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/MPRealSupport"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/NonLinearOptimization"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/NumericalDiff"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/OpenGLSupport"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/Polynomials"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/Skyline"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/SparseExtra"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/SpecialFunctions"
    "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

