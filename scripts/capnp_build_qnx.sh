#!/bin/bash

set -eo pipefail

if [[ -z "${QNX_SOURCE_SCRIPT}" ]]; then
    echo "Set the environment variable QNX_SOURCE_SCRIPT to specify the path to qnxsdp-env.sh, it's probably where you installed the software tools: /blah/qnx700/qnxsdp-env.sh"
    exit 1
fi

if [[ -z "${LDK_QNX_INSTALL_FOLDER}" ]]; then
    echo "Set the environment variable LDK_QNX_INSTALL_FOLDER to specify where dependencies should be installed to."
    exit 1
fi

if [[ $# -ne 2 ]]; then
    echo "You must specify a build type (Release, Debug) and a toolchain file!"
    exit 1
fi

. "${QNX_SOURCE_SCRIPT}"

rm -rf build
mkdir build
cd build

cmake -G "Unix Makefiles" ..                                                             \
 -DCMAKE_PREFIX_PATH:PATH=${LDK_QNX_INSTALL_FOLDER}                                      \
 -DCMAKE_INSTALL_PREFIX:PATH=${LDK_QNX_INSTALL_FOLDER}                                   \
 -DCMAKE_BUILD_TYPE="$1"                                                                 \
 -DCMAKE_TOOLCHAIN_FILE="$2"                                                             \
 -DBUILD_TESTING=OFF # off until I figure out what's up with capnp_tool and some of the message generation that happens with testing on

cmake --build . --target install -- -j "${LUMPDK_NPROC:-$(($(nproc) + 2))}"
