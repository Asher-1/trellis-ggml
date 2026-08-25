# cumesh_glue — CMake build for JeffreyXiang/CuMesh (MIT) with a torch-free C API.

# ── Find PyTorch ────────────────────────────────────────────────────
# Suppress missing-MKL / kineto warnings from conda torch.
set(MKL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "" FORCE)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/__mkl_stub")
set(MKL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/__mkl_stub" CACHE PATH "" FORCE)
if(CUMESH_TORCH_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${CUMESH_TORCH_DIR}")
endif()
find_package(Torch REQUIRED)
get_target_property(_torch_inc torch INTERFACE_INCLUDE_DIRECTORIES)
if(_torch_inc)
    set(_clean)
    foreach(_d IN LISTS _torch_inc)
        if(EXISTS "${_d}")
            list(APPEND _clean "${_d}")
        endif()
    endforeach()
    set_target_properties(torch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_clean}")
endif()

# ── Source files from CuMesh submodule ──────────────────────────────
# We compile all .cu sources needed for chart clustering and mesh
# cleanup.  remesh/ and simplify.cu are excluded (use our own sloppy
# decimation for now).
set(CUMESH_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/atlas.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/clean_up.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/connectivity.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/cumesh.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/geometry.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/hash/hash.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/io.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/src/shared.cu
)

# cubvh headers are needed by cumesh.h (for cub_temp_storage field)
set(CUBVH_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/../CuMesh/third_party/cubvh/include
)

# ── Library ─────────────────────────────────────────────────────────
add_library(cumesh_glue SHARED
    ${CUMESH_SRC}
    cumesh_glue.cu
)

target_include_directories(cumesh_glue PRIVATE
    ${CUBVH_INCLUDE}
    ${CMAKE_SOURCE_DIR}   # for third_party/CuMesh/src/... paths
)

# Filter out non-existent include dirs (conda torch may reference
# MKL paths that don't exist on this system).
set(_filtered "${TORCH_INCLUDE_DIRS}")
foreach(_dir ${TORCH_INCLUDE_DIRS})
    if(NOT EXISTS "${_dir}")
        list(REMOVE_ITEM _filtered "${_dir}")
    endif()
endforeach()
target_include_directories(cumesh_glue PRIVATE ${_filtered})

# torch/extension.h -> torch/python.h -> Python.h.
# Resolve Python include from the conda environment that provides torch.
set(_conda_root "")
if(CUMESH_TORCH_DIR)
    # Walk up from CUMESH_TORCH_DIR to find the conda env root
    set(_d "${CUMESH_TORCH_DIR}")
    foreach(i RANGE 6)
        get_filename_component(_d "${_d}" DIRECTORY)
        file(GLOB _py_incs LIST_DIRECTORIES true "${_d}/include/python*")
        if(_py_incs)
            list(GET _py_incs 0 _py_inc)
            if(EXISTS "${_py_inc}/Python.h")
                target_include_directories(cumesh_glue PRIVATE "${_py_inc}")
                set(_conda_root "${_d}")
                break()
            endif()
        endif()
    endforeach()
endif()

# Link libtorch directly (avoid -lFALSE from conda's broken TORCH_LIBRARIES).
# CUMESH_TORCH_DIR = .../torch/share/cmake/Torch
# torch root = ../../.. from there (= .../torch)
set(_torch_pkg "${CUMESH_TORCH_DIR}/../../..")
get_filename_component(_torch_pkg "${_torch_pkg}" ABSOLUTE)
set(_torch_lib_dir "${_torch_pkg}/lib")
if(NOT EXISTS "${_torch_lib_dir}/libtorch.so")
    message(FATAL_ERROR "libtorch.so not found at ${_torch_lib_dir}. Set CUMESH_TORCH_DIR correctly.")
endif()
message(STATUS "CuMesh torch lib dir: ${_torch_lib_dir}")

target_link_directories(cumesh_glue PRIVATE "${_torch_lib_dir}")
target_link_libraries(cumesh_glue
    PRIVATE
    "${_torch_lib_dir}/libtorch.so"
    "${_torch_lib_dir}/libtorch_cpu.so"
    "${_torch_lib_dir}/libtorch_cuda.so"
    "${_torch_lib_dir}/libc10.so"
    "${_torch_lib_dir}/libc10_cuda.so"
    # torch/extension.h drags in the pybind11 layer: without these two the
    # shared lib keeps ~100 undefined Py* symbols and every downstream link
    # fails ("undefined reference to PyErr_Fetch").
    "${_torch_lib_dir}/libtorch_python.so"
    -lnvrtc
)
if(_conda_root)
    file(GLOB _py_libs "${_conda_root}/lib/libpython3*.so")
    if(_py_libs)
        list(GET _py_libs 0 _py_lib)
        target_link_libraries(cumesh_glue PRIVATE "${_py_lib}")
    endif()
endif()

# CUDA architecture for RTX 4090
set_target_properties(cumesh_glue PROPERTIES
    CUDA_ARCHITECTURES "89"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Silence nvcc warnings and set C++17 for ATen compatibility
target_compile_options(cumesh_glue PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

# ── Smoke test ─────────────────────────────────────────────────────
add_executable(test_cumesh test_cumesh.cu)
target_link_libraries(test_cumesh PRIVATE cumesh_glue)
target_include_directories(test_cumesh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_cumesh PROPERTIES CUDA_ARCHITECTURES "89")

# ── Export for parent project ───────────────────────────────────────
# The parent CMakeLists.txt should do:
#   add_subdirectory(third_party/cumesh_glue)
#   target_link_libraries(trellis2 PRIVATE cumesh_glue)
# and cumesh_glue.h is the public API header.