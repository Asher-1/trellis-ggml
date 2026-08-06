cmake_minimum_required(VERSION 3.14)
project(trellis2.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────────────────────────────────
# External ggml option
#
# When OFF (default): vendor ggml as a subdirectory — the upstream behavior.
# When ON : skip the bundled ggml and resolve `find_package(ggml CONFIG)`
#           against an installed ggml. Lets a downstream project link multiple
#           ggml-based libraries against a single shared ggml so they don't
#           ship duplicate copies into the same process.
# ─────────────────────────────────────────────────────────────────────────
option(TRELLIS2_USE_EXTERNAL_GGML "Consume external ggml via find_package() instead of the bundled submodule" OFF)

# libFuzzer harnesses (clang only). The sanitizer flags apply globally — set
# BEFORE any target so the whole tree (ggml, stb decode, preprocessing,
# loaders) is instrumented. Harness targets live in fuzz/ (added at the end).
option(TRELLIS2_FUZZ "Build libFuzzer harnesses (requires clang)" OFF)
set(TRELLIS2_FUZZ_SANITIZERS "address,undefined" CACHE STRING
    "Sanitizers for the fuzz build")
if(TRELLIS2_FUZZ)
    add_compile_options(-fsanitize=${TRELLIS2_FUZZ_SANITIZERS} -fsanitize=fuzzer-no-link -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=${TRELLIS2_FUZZ_SANITIZERS})
endif()

# ggml options — enable Metal on Apple (bundled-ggml path only)
option(TRELLIS2_METAL "Enable Metal backend" ON)
if(APPLE AND TRELLIS2_METAL AND NOT TRELLIS2_USE_EXTERNAL_GGML)
    set(GGML_METAL              ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

# ---------------------------------------------------------------------
# Submodule materialization + patches
#
# The bundled ggml and RMBG-2.0-GGML submodules carry a few required fixes
# (CUDA Q8_0->Q8_0 copy support, RMBG CMake/stb handling). Instead of
# patching the submodule working trees in place, `git archive` materializes
# the pinned commit into the build tree and the patches are applied there.
# Submodules stay pristine; the patches in patches/ are the source of truth.
# The ggml-rmbg-ops.patch is synced automatically from the RMBG submodule's
# own third_party/ggml-rmbg.patch, so an upstream update needs no manual copy.
# ---------------------------------------------------------------------
function(_trellis_materialize_submodule out_var subdir)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
    if(NOT EXISTS "${_src}/.git")
        message(FATAL_ERROR "${subdir} submodule missing. Run:\n"
            "  git submodule update --init --recursive ${subdir}")
    endif()
    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${_src}" rev-parse HEAD
        OUTPUT_VARIABLE _commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rev_res)
    if(NOT _rev_res EQUAL 0)
        message(FATAL_ERROR "Failed to read ${subdir} submodule commit")
    endif()
    # Fingerprint = commit + every patch's SHA256; any change re-materializes.
    set(_stamp_parts "${_commit}")
    set(_patches)
    foreach(_p IN LISTS ARGN)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_p}")
            message(FATAL_ERROR "Required patch missing: ${_p}")
        endif()
        file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_p}" _ph)
        list(APPEND _patches "${CMAKE_CURRENT_SOURCE_DIR}/${_p}")
        string(APPEND _stamp_parts "\n${_ph}")
    endforeach()
    string(SHA256 "${_stamp_parts}" _stamp_hash)
    string(SUBSTRING "${_stamp_hash}" 0 12 _stamp_short)
    string(SUBSTRING "${_commit}" 0 12 _commit_short)
    string(REPLACE "/" "-" _mapped_name "${subdir}")
    set(_mapped "${CMAKE_BINARY_DIR}/_deps/${_mapped_name}-${_commit_short}-${_stamp_short}")
    set(_stamp "${_mapped}/.trellis-patch-stamp")
    set(_current "")
    if(EXISTS "${_stamp}")
        file(READ "${_stamp}" _current)
    endif()
    if(NOT _current STREQUAL _stamp_parts)
        file(REMOVE_RECURSE "${_mapped}")
        file(MAKE_DIRECTORY "${_mapped}")
        set(_archive "${CMAKE_BINARY_DIR}/_deps/${_mapped_name}-${_commit_short}.tar")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_src}" archive --format=tar --output "${_archive}" HEAD
            RESULT_VARIABLE _arch_res)
        if(_arch_res EQUAL 0)
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E tar xf "${_archive}"
                WORKING_DIRECTORY "${_mapped}"
                RESULT_VARIABLE _extract_res)
        endif()
        file(REMOVE "${_archive}")
        if(NOT _arch_res EQUAL 0)
            message(FATAL_ERROR "Failed to materialize ${subdir} commit ${_commit}")
        endif()
        if(NOT _extract_res EQUAL 0)
            message(FATAL_ERROR "Failed to extract ${subdir} archive")
        endif()
        foreach(_p IN LISTS _patches)
            # The build tree sits inside the parent repository: prevent git from
            # treating it as a path prefix and skipping patch entries outside it.
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                        "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
                        "${GIT_EXECUTABLE}" apply "${_p}"
                WORKING_DIRECTORY "${_mapped}"
                RESULT_VARIABLE _apply_res
                OUTPUT_VARIABLE _apply_out ERROR_VARIABLE _apply_out)
            if(NOT _apply_res EQUAL 0)
                message(FATAL_ERROR
                    "Failed to apply ${_p} to ${subdir} ${_commit_short}.\n"
                    "The submodule was updated and this integration patch is stale.\n"
                    "  ${_apply_out}")
            endif()
        endforeach()
        file(WRITE "${_stamp}" "${_stamp_parts}")
        message(STATUS "Materialized ${subdir} ${_commit_short} (${_stamp_short})")
    endif()
    set(${out_var} "${_mapped}" PARENT_SCOPE)
    set(${out_var}_COMMIT "${_commit}" PARENT_SCOPE)
endfunction()

if(TRELLIS2_USE_EXTERNAL_GGML)
    find_package(ggml CONFIG REQUIRED)
    set(TRELLIS2_GGML_TARGET ggml::ggml)
else()
    # Single shared ggml, materialized in the build tree with both the Q8_0
    # CUDA copy fix (needed by the trellis2 pipeline) and the RMBG custom-ops
    # patch (deform-im2col, swin, ...). The RMBG submodule reuses this same
    # ggml target when built here.
    # patches/ggml-rmbg-ops.patch is the canonical copy of the RMBG submodule's
    # third_party/ggml-rmbg.patch; when the submodule updates it, sync it here
    # automatically so this build follows upstream without manual steps.
    set(_rmbg_upstream_patch "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RMBG-2.0-GGML/third_party/ggml-rmbg.patch")
    set(_rmbg_local_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-rmbg-ops.patch")
    if(EXISTS "${_rmbg_upstream_patch}" AND EXISTS "${_rmbg_local_patch}")
        file(SHA256 "${_rmbg_upstream_patch}" _up_sha)
        file(SHA256 "${_rmbg_local_patch}" _loc_sha)
        if(NOT _up_sha STREQUAL _loc_sha)
            configure_file("${_rmbg_upstream_patch}" "${_rmbg_local_patch}" COPYONLY)
            message(STATUS "Synced patches/ggml-rmbg-ops.patch from RMBG submodule")
        endif()
    endif()
    _trellis_materialize_submodule(TRELLIS_GGML_MAPPED third_party/ggml
        patches/ggml-cuda-cpy-q8_0.patch
        patches/ggml-rmbg-ops.patch)
    # The materialized tree has no .git: pass the real commit so ggml's
    # configure skips its git rev-parse (which would otherwise find the
    # parent repository) and reports the submodule commit correctly.
    set(GGML_BUILD_COMMIT "${TRELLIS_GGML_MAPPED_COMMIT}")
    add_subdirectory("${TRELLIS_GGML_MAPPED}")
    set(TRELLIS2_GGML_TARGET ggml)
endif()

add_library(meshoptimizer STATIC
    third_party/meshoptimizer/allocator.cpp
    third_party/meshoptimizer/indexgenerator.cpp
    third_party/meshoptimizer/simplifier.cpp)
set_target_properties(meshoptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(meshoptimizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/meshoptimizer)

# Centralised stb_image / stb_image_write implementations. Compiled exactly once
# as an OBJECT library and linked PUBLIC to trellis2 so every consumer
# (examples, shared lib, tests) gets the symbols without multiple-definition
# conflicts with librmbg.a.
add_library(stb_impl OBJECT stb_impl.cpp)
target_include_directories(stb_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
set_target_properties(stb_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(trellis2
    trellis2.cpp trellis2.h
    trellis2_capi.cpp trellis2_capi.h
    mesh_export.cpp mesh_export.h
    print_remesh.cpp print_remesh.h
    third_party/xatlas/xatlas.cpp)
target_link_libraries(trellis2 PUBLIC stb_impl)
target_include_directories(trellis2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)
# trellis2_capi.cpp reuses the example isosurface extractor for the demo path;
# mesh_export.cpp vendors xatlas for the optional chart-based UV unwrap.
target_include_directories(trellis2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xatlas)

find_package(Threads REQUIRED)  # xatlas uses std::thread

# Optional libwebp: lets the C++ image decode path understand WebP inputs
# (stb_image supports JPEG/PNG/TGA/BMP/GIF/HDR/PIC/PNM but not WebP).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(WebP QUIET libwebp)
endif()
if(NOT WebP_FOUND)
    find_library(WebP_LIBRARIES NAMES webp)
    find_path(WebP_INCLUDE_DIRS NAMES webp/decode.h)
    if(WebP_LIBRARIES AND WebP_INCLUDE_DIRS)
        set(WebP_FOUND TRUE)
    endif()
endif()
if(WebP_FOUND)
    message(STATUS "WebP decode: enabled (${WebP_LIBRARIES})")
else()
    message(STATUS "WebP decode: disabled (libwebp not found)")
endif()

if(TRELLIS2_USE_EXTERNAL_GGML AND DEFINED GGML_INCLUDE_DIR)
    target_include_directories(trellis2 PUBLIC ${GGML_INCLUDE_DIR})
endif()

target_link_libraries(trellis2 PUBLIC ${TRELLIS2_GGML_TARGET}
    PRIVATE Threads::Threads meshoptimizer)
target_compile_features(trellis2 PUBLIC cxx_std_14)

if(WebP_FOUND)
    target_include_directories(trellis2 PRIVATE ${WebP_INCLUDE_DIRS})
    target_link_libraries(trellis2 PRIVATE ${WebP_LIBRARIES})
    target_compile_definitions(trellis2 PRIVATE TRELLIS2_HAVE_WEBP=1)
endif()

# o-voxel flexible dual grid (QEF mesh -> sparse grid), Eigen-only.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense PATHS /usr/include/eigen3 /usr/local/include/eigen3)
    if(EIGEN3_INCLUDE_DIR)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
        set(Eigen3_FOUND TRUE)
    endif()
endif()
if(Eigen3_FOUND)
    add_library(o_voxel_fdg STATIC third_party/o-voxel-fdg/flexible_dual_grid.cpp)
    set_property(TARGET o_voxel_fdg PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(o_voxel_fdg PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/o-voxel-fdg)
    target_link_libraries(o_voxel_fdg PUBLIC Eigen3::Eigen)
    target_compile_features(o_voxel_fdg PUBLIC cxx_std_14)
    target_sources(trellis2 PRIVATE examples/mesh_to_dual_grid.cpp)
    target_link_libraries(trellis2 PRIVATE o_voxel_fdg)
    target_compile_definitions(trellis2 PRIVATE TRELLIS2_HAVE_MESH_TO_GRID=1)
    message(STATUS "mesh->dual-grid QEF: enabled (Eigen)")
else()
    message(STATUS "mesh->dual-grid QEF: disabled (Eigen3 not found)")
endif()

# CGAL Alpha Wrap turns arbitrary TRELLIS triangle soups into watertight,
# intersection-free 2-manifolds for printing. Auto-detect it so ordinary builds
# retain no CGAL dependency; the C API reports whether the backend is present.
option(TRELLIS2_CGAL "Enable CGAL Alpha Wrap print remeshing when available" ON)
option(TRELLIS2_FETCH_PRINT_REMESH_DEPS
    "Fetch the project-pinned CGAL and Boost headers for print remeshing" OFF)
set(TRELLIS2_PRINT_REMESH_DEPS_DIR "${CMAKE_BINARY_DIR}/_deps/print-remesh" CACHE PATH
    "Directory containing project-pinned CGAL and Boost headers")
if(TRELLIS2_CGAL)
    if(TRELLIS2_FETCH_PRINT_REMESH_DEPS)
        find_program(TRELLIS2_BASH_EXECUTABLE bash)
        if(NOT TRELLIS2_BASH_EXECUTABLE)
            message(FATAL_ERROR "bash is required to fetch print-remesh dependencies")
        endif()
        execute_process(
            COMMAND "${TRELLIS2_BASH_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fetch_print_remesh_deps.sh"
                    "${TRELLIS2_PRINT_REMESH_DEPS_DIR}"
            RESULT_VARIABLE TRELLIS2_PRINT_REMESH_DEPS_RESULT)
        if(NOT TRELLIS2_PRINT_REMESH_DEPS_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to fetch pinned print-remesh dependencies")
        endif()
        set(CGAL_DIR "${TRELLIS2_PRINT_REMESH_DEPS_DIR}/cgal" CACHE PATH
            "CGAL package directory" FORCE)
        set(Boost_INCLUDE_DIR "${TRELLIS2_PRINT_REMESH_DEPS_DIR}/boost" CACHE PATH
            "Boost header directory" FORCE)
        set(Boost_DIR "${TRELLIS2_PRINT_REMESH_DEPS_DIR}/boost-cmake" CACHE PATH
            "Header-only Boost CMake package directory" FORCE)
    endif()
    find_package(CGAL 5.5 QUIET)
    if(CGAL_FOUND)
        target_link_libraries(trellis2 PRIVATE CGAL::CGAL)
        target_compile_definitions(trellis2 PRIVATE TRELLIS2_USE_CGAL=1)
        target_compile_features(trellis2 PRIVATE cxx_std_17)
        set(TRELLIS2_HAVE_CGAL ON)
        message(STATUS "CGAL Alpha Wrap print remeshing: enabled")
    elseif(TRELLIS2_FETCH_PRINT_REMESH_DEPS)
        message(FATAL_ERROR "The project-pinned CGAL print-remesh dependencies were fetched but not found")
    else()
        message(STATUS "CGAL Alpha Wrap print remeshing: unavailable (CGAL >= 5.5 not found)")
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────
# RMBG-2.0 background removal (optional)
#
# When the submodule is present and TRELLIS2_RMBG is ON, the rmbg library is
# linked into trellis2 and the C/C++ API exposes AI-based background removal
# via BiRefNet/RMBG-2.0 GGUF models.
# ─────────────────────────────────────────────────────────────────────────
option(TRELLIS2_RMBG "Enable RMBG-2.0 AI background removal" ON)
if(TRELLIS2_RMBG AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RMBG-2.0-GGML/CMakeLists.txt")
    # Materialize the RMBG submodule in the build tree (its working tree stays
    # pristine) and apply the integration patch there: skip the nested ggml
    # and use this repo's shared ggml + stb implementations.
    _trellis_materialize_submodule(TRELLIS_RMBG_MAPPED third_party/RMBG-2.0-GGML
        patches/rmbg-cmake-preprocess.patch)
    set(RMBG_BUILD_CLI        OFF CACHE BOOL "" FORCE)
    set(RMBG_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
    set(TRELLIS2_STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stb")
    add_subdirectory("${TRELLIS_RMBG_MAPPED}" EXCLUDE_FROM_ALL)
    # RMBG reuses this repo's single ggml target (patched with Q8 + RMBG ops
    # above) and needs stb_image / stb_image_write implementations, which
    # trellis2 provides once via stb_impl.
    target_link_libraries(rmbg PRIVATE stb_impl)
    target_link_libraries(trellis2 PRIVATE rmbg)
    target_compile_definitions(trellis2 PRIVATE TRELLIS2_HAVE_RMBG=1)
    target_include_directories(trellis2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/RMBG-2.0-GGML/include)
    set(TRELLIS2_HAVE_RMBG ON)
    message(STATUS "RMBG-2.0 background removal: enabled")
else()
    message(STATUS "RMBG-2.0 background removal: disabled")
endif()

set_property(TARGET trellis2 PROPERTY POSITION_INDEPENDENT_CODE ON)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(trellis2 PUBLIC  TRELLIS2_SHARED)
    target_compile_definitions(trellis2 PRIVATE TRELLIS2_BUILD)
endif()

option(TRELLIS2_BUILD_EXAMPLES "Build example executables" ON)
if(TRELLIS2_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

option(TRELLIS2_BUILD_TESTS "Build test executables" OFF)
if(TRELLIS2_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TRELLIS2_FUZZ)
    add_subdirectory(fuzz)
endif()
