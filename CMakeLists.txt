cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dn_roof)

set(CMAKE_BUILD_TYPE Release)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("D:\\research\\DN_Roof\\include\\")

add_executable(dn_roof Utils.h Utils.cpp dn_roof.h dn_roof.cpp)
target_link_libraries(dn_roof "${TORCH_LIBRARIES}")
target_link_libraries(dn_roof "${OpenCV_LIBS}")
set_property(TARGET dn_roof PROPERTY CXX_STANDARD 14)