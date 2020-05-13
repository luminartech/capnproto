cmake_minimum_required(VERSION 3.10)

# Leaving this commented out block in in case we decide to come back to it
#include(cmake/toolchains/x86_64/windows/msvc/FindWindowsSDK.cmake)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake/")
#set (WINDOWSSDK_DIRS "C:/Program Files (x86)/Windows Kits/10")
#find_package(WindowsSDK)
#message("WINDOWSSDK_PREFFERED_DIR: ${WINDOWSSDK_PREFERRED_DIR}")
#message("Did we find an sdk: ${WINDOWSSDK_FOUND}")
#set(out "out")
#get_windowssdk_library_dirs(${WINDOWSSDK_DIRS} ${out})
#message(FATAL_ERROR "out is : ${out}")

set (LUMPDK_TARGET_TRIPLET_ARCH x86_64)
set (LUMPDK_TARGET_TRIPLET_PLAT windows10)
set (LUMPDK_TARGET_TRIPLET_COMPILER msvc)

set (LUMPDK_TARGET_TRIPLET ${LUMPDK_TARGET_TRIPLET_ARCH}_${LUMPDK_TARGET_TRIPLET_PLAT}_${LUMPDK_TARGET_TRIPLET_COMPILER})

set (BUILD_ARCH x86_64)
set (BUILD_OS "Windows10")

set (DEFAULT_LUMPDK_VCPKG_TOOLCHAIN_FILE "c:/src/vcpkg/scripts/buildsystems/vcpkg.cmake")

if (LUMPDK_VCPKG_TOOLCHAIN_FILE)
  include("${LUMPDK_VCPKG_TOOLCHAIN_FILE}")
  
elseif (EXISTS "${DEFAULT_LUMPDK_VCPKG_TOOLCHAIN_FILE}")
  include("c:/src/vcpkg/scripts/buildsystems/vcpkg.cmake")
  
else()
  message(FATAL_ERROR "vcpkg toolchain file is unset, failing build!")
  
endif()
