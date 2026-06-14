#!/bin/bash
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

LIB_EXTENSION="so"

if [[ "$OSTYPE" == "darwin"* ]] ; then
    LIB_EXTENSION="dylib"
fi

# Take into account # of cores and available RAM for deciding on compilation parallelism.
# TODO: Try importing psutil and if failed fall back to 1 thread
PARALLELISM=${PLFB_BUILD_JOBS:-$(python3 -c 'import psutil; import multiprocessing as mp; print(int(max(1,min((psutil.virtual_memory().available/1000000000-1)/0.5, mp.cpu_count()))))')}

# Clean in-source CMake artifacts from previous failed builds.
rm -rf third_party/gfootball_engine/CMakeFiles
rm -f third_party/gfootball_engine/CMakeCache.txt
rm -f third_party/gfootball_engine/Makefile
rm -f third_party/gfootball_engine/cmake_install.cmake
rm -f third_party/gfootball_engine/libgame.$LIB_EXTENSION
rm -f third_party/gfootball_engine/_gameplayfootball.so

# CMake 4 removed compatibility with projects that declare a <3.5 minimum.
# Keep the vendored engine buildable without requiring users to pin CMake.
CMAKE_POLICY_ARGS=("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")

CMAKE_PYTHON_ARGS=()
CMAKE_CONDA_ARGS=()
PYTHON_PREFIX=$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)
BUILD_PREFIX=${CONDA_PREFIX:-$PYTHON_PREFIX}
if [[ -n "${BUILD_PREFIX:-}" && -d "${BUILD_PREFIX}" ]]; then
    PYTHON_EXECUTABLE=$(command -v python)
    PYTHON_INCLUDE_DIR=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
    PYTHON_LIBRARY=$(python - <<'PY'
import glob
import os
import sys
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR") or os.path.join(sys.prefix, "lib")
version = f"{sys.version_info.major}.{sys.version_info.minor}"
candidates = [
    os.path.join(libdir, f"libpython{version}.so"),
    os.path.join(libdir, f"libpython{version}.dylib"),
    os.path.join(libdir, sysconfig.get_config_var("LDLIBRARY") or ""),
]
candidates.extend(sorted(glob.glob(os.path.join(libdir, f"libpython{version}.*"))))
for candidate in candidates:
    if candidate and os.path.exists(candidate):
        print(candidate)
        break
else:
    print(os.path.join(libdir, sysconfig.get_config_var("LDLIBRARY") or f"libpython{version}.so"))
PY
)
    CMAKE_PYTHON_ARGS=(
        "-DPython_ROOT_DIR=${BUILD_PREFIX}"
        "-DPython_EXECUTABLE=${PYTHON_EXECUTABLE}"
        "-DPython_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}"
        "-DPython_LIBRARY=${PYTHON_LIBRARY}"
        "-DPython_LIBRARY_RELEASE=${PYTHON_LIBRARY}"
        "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}"
        "-DPython3_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}"
        "-DPython3_LIBRARY=${PYTHON_LIBRARY}"
        "-DPython3_LIBRARY_RELEASE=${PYTHON_LIBRARY}"
        "-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
        "-DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}"
        "-DPYTHON_LIBRARY=${PYTHON_LIBRARY}"
    )

    # Non-interactive conda-prefix builds do not always expose package roots to CMake.
    # Prefer conda-provided OpenGL/SDL/Boost libraries when they are present.
    CMAKE_CONDA_ARGS=("-DCMAKE_PREFIX_PATH=${BUILD_PREFIX}")
    if [[ -f "${BUILD_PREFIX}/lib/libOpenGL.so" ]]; then
        CMAKE_CONDA_ARGS+=("-DOPENGL_opengl_LIBRARY=${BUILD_PREFIX}/lib/libOpenGL.so")
    fi
    if [[ -f "${BUILD_PREFIX}/lib/libGLX.so" ]]; then
        CMAKE_CONDA_ARGS+=("-DOPENGL_glx_LIBRARY=${BUILD_PREFIX}/lib/libGLX.so")
    fi
    if [[ -d "${BUILD_PREFIX}/include/GL" ]]; then
        CMAKE_CONDA_ARGS+=("-DOPENGL_INCLUDE_DIR=${BUILD_PREFIX}/include")
    fi
fi

pushd third_party/gfootball_engine && cmake "${CMAKE_POLICY_ARGS[@]}" "${CMAKE_PYTHON_ARGS[@]}" "${CMAKE_CONDA_ARGS[@]}" . && make -j $PARALLELISM && popd
# Keep the Python extension self-contained after editable installs; a symlink breaks
# if build cleanup later removes libgame.so while leaving _gameplayfootball.so expected.
pushd third_party/gfootball_engine && cp -f libgame.$LIB_EXTENSION _gameplayfootball.so && popd
