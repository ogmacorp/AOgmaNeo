# --------------------------------------------------------------------------
# AOgmaNeo
# Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of AOgmaNeo is licensed to you under the terms described
# in the AOGMANEO_LICENSE.md file included in this distribution.
# --------------------------------------------------------------------------

# Locate AOgmaNeo library
#
# This module defines
# AOGMANEO_LIBRARY, the name of the library to link against
# AOGMANEO_FOUND, if false, do not try to link to AOgmaNeo
# AOGMANEO_INCLUDE_DIR, where to find AOgmaNeo headers

if(AOGMANEO_INCLUDE_DIR)
    # Already in cache, be silent
    set(AOGMANEO_FIND_QUIETLY TRUE)
endif(AOGMANEO_INCLUDE_DIR)

find_path(AOGMANEO_INCLUDE_DIR aogmaneo/Hierarchy.h)

set(AOGMANEO_NAMES aogmaneo AOgmaNeo AOGMANEO)

find_library(AOGMANEO_LIBRARY NAMES ${AOGMANEO_NAMES})

# Per-recommendation
set(AOGMANEO_INCLUDE_DIRS "${AOGMANEO_INCLUDE_DIR}")
set(AOGMANEO_LIBRARIES "${AOGMANEO_LIBRARY}")

# handle the QUIETLY and REQUIRED arguments and set AOGMANEO_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AOgmaNeo DEFAULT_MSG AOGMANEO_LIBRARY AOGMANEO_INCLUDE_DIR)
