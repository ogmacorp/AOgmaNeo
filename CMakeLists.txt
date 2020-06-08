# ----------------------------------------------------------------------------
#  AOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
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

include_directories("${PROJECT_SOURCE_DIR}/source")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/aogmaneo/Actor.cpp"
    "${SOURCE_PATH}/aogmaneo/Helpers.cpp"
    "${SOURCE_PATH}/aogmaneo/Hierarchy.cpp"
    "${SOURCE_PATH}/aogmaneo/Predictor.cpp"
    "${SOURCE_PATH}/aogmaneo/SparseCoder.cpp"
    "${SOURCE_PATH}/aogmaneo/SparseMatrix.cpp"
)

set(HEADERS
    "${SOURCE_PATH}/aogmaneo/Actor.h"
    "${SOURCE_PATH}/aogmaneo/Array.h"
	"${SOURCE_PATH}/aogmaneo/Helpers.h"
    "${SOURCE_PATH}/aogmaneo/Hierarchy.h"
    "${SOURCE_PATH}/aogmaneo/Predictor.h"
    "${SOURCE_PATH}/aogmaneo/Ptr.h"
    "${SOURCE_PATH}/aogmaneo/SparseCoder.h"
	"${SOURCE_PATH}/aogmaneo/SparseMatrix.h"
)

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This links -fopenmp

add_library(AOgmaNeo ${SOURCES} ${HEADERS})

target_link_libraries(AOgmaNeo ${OpenMP_CXX_LIBRARIES})

install(TARGETS AOgmaNeo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h*")