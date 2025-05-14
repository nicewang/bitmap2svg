cmake_minimum_required(VERSION 3.18)
project(bitmap2svg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc) 
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

find_package(pybind11 REQUIRED)
message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}") 
message(STATUS "Pybind11 library: ${pybind11_LIBRARIES}") 

pybind11_add_module(bitmap2svg_core 
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp 
)

target_include_directories(bitmap2svg_core PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    # ${pybind11_INCLUDE_DIRS} # pybind11_add_module usually handles this
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmap2svg/cpp" 
)
target_link_libraries(bitmap2svg_core PRIVATE
    ${OpenCV_LIBS}
    pybind11::module 
)

install(TARGETS bitmap2svg_core DESTINATION bitmap2svg)
