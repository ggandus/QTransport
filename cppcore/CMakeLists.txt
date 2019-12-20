cmake_minimum_required(VERSION 2.8.12)
project(example_core)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
# include_directories(include)

file(GLOB SOURCES "src/*.cpp")
# file(GLOB INCLUDES "include/*.h")

add_library(cppcore ${SOURCES}) # ${INCLUDES})

target_include_directories(cppcore PUBLIC include)
target_include_directories(cppcore PUBLIC ${EIGEN3_INCLUDE_DIRS})
