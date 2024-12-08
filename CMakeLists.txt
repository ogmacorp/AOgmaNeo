# ----------------------------------------------------------------------------
#  AOgmaNeo
#  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of AOgmaNeo is licensed to you under the terms described
#  in the AOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(AOgmaNeo)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_definitions(USE_OMP) # Use OpenMP
add_compile_definitions(USE_STD_MATH) # Use math funcs from standard library

include_directories("${PROJECT_SOURCE_DIR}/source")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/aogmaneo/helpers.cpp"
    "${SOURCE_PATH}/aogmaneo/layer.cpp"
    "${SOURCE_PATH}/aogmaneo/hierarchy.cpp"
    "${SOURCE_PATH}/aogmaneo/image_encoder.cpp"
)

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This also links -fopenmp

add_library(AOgmaNeo ${SOURCES})

set_target_properties(AOgmaNeo PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(AOgmaNeo ${OpenMP_CXX_LIBRARIES})

install(TARGETS AOgmaNeo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h*")
