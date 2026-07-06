# Numerical validation of the SS-flow DiT forward pass against a PyTorch
# reference (generate the reference with tests/ref_ss_flow.py).
add_executable(test_ss_flow_forward test_ss_flow_forward.cpp)
target_link_libraries(test_ss_flow_forward PRIVATE trellis2)
target_compile_features(test_ss_flow_forward PRIVATE cxx_std_14)

# Full flow-Euler sampling loop vs the PyTorch reference (tests/ref_ss_sample.py).
add_executable(test_ss_sample test_ss_sample.cpp)
target_link_libraries(test_ss_sample PRIVATE trellis2)
target_compile_features(test_ss_sample PRIVATE cxx_std_14)

# SS decoder forward vs the PyTorch reference (tests/ref_ss_dec.py).
add_executable(test_ss_dec test_ss_dec.cpp)
target_link_libraries(test_ss_dec PRIVATE trellis2)
target_compile_features(test_ss_dec PRIVATE cxx_std_14)

# PIL-compatible preprocessing byte-exactness vs the Python reference.
add_executable(test_preprocess test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE trellis2)
target_include_directories(test_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../stb)
target_compile_features(test_preprocess PRIVATE cxx_std_14)

# DINOv3 encoder per-layer parity vs the PyTorch reference
# (scripts/dump_dino_reference.py -> dumps/reference_dino.gguf).
add_executable(test_dino test_dino.cpp)
target_link_libraries(test_dino PRIVATE trellis2)
target_compile_features(test_dino PRIVATE cxx_std_14)

# Shape-SLAT flow + FDG decoder parity vs the PyTorch reference
# (scripts/dump_slat_reference.py -> dumps/reference_slat.gguf).
add_executable(test_slat test_slat.cpp)
target_link_libraries(test_slat PRIVATE trellis2)
target_compile_features(test_slat PRIVATE cxx_std_14)

# Self-contained isosurface extractor (examples/marching_cubes.h) on analytic
# fields — watertight-manifold + Euler-characteristic invariants. No model.
add_executable(test_marching_cubes test_marching_cubes.cpp)
target_include_directories(test_marching_cubes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
target_compile_features(test_marching_cubes PRIVATE cxx_std_14)

# ── ctest registration ────────────────────────────────────────────────────────
# Asset paths default to the in-repo locations produced by
# scripts/download_models.sh + converters + scripts/refgen.sh, overridable via
# TRELLIS2_GGUF_DIR / TRELLIS2_DUMPS. Tests exit 77 when assets are absent, so
# a fresh checkout still gets a green (skipped) suite.
set(T2_GGUFS "$ENV{TRELLIS2_GGUF_DIR}")
if(NOT T2_GGUFS)
    set(T2_GGUFS "${CMAKE_CURRENT_SOURCE_DIR}/../ggufs")
endif()
set(T2_DUMPS "$ENV{TRELLIS2_DUMPS}")
if(NOT T2_DUMPS)
    set(T2_DUMPS "${CMAKE_CURRENT_SOURCE_DIR}/../dumps")
endif()
set(T2_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME marching_cubes COMMAND test_marching_cubes)

add_test(NAME preprocess COMMAND test_preprocess
         "${T2_DUMPS}/fixture_rgba.png" "${T2_DUMPS}/fixture_512.png")
set_tests_properties(preprocess PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME dino COMMAND test_dino
         "${T2_GGUFS}/dino_f32.gguf" "${T2_DUMPS}/reference_dino.gguf")
set_tests_properties(dino PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME ss_flow_forward COMMAND test_ss_flow_forward
         "${T2_GGUFS}/ss_flow_f32.gguf" "${T2_TESTS}/ss_flow_ref.bin")
set_tests_properties(ss_flow_forward PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME ss_sample COMMAND test_ss_sample
         "${T2_GGUFS}/ss_flow_f32.gguf" "${T2_TESTS}/ss_sample_ref.bin")
set_tests_properties(ss_sample PROPERTIES SKIP_RETURN_CODE 77 LABELS "model;slow")

add_test(NAME ss_dec COMMAND test_ss_dec
         "${T2_GGUFS}/ss_dec_f32.gguf" "${T2_TESTS}/ss_dec_ref.bin")
set_tests_properties(ss_dec PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

add_test(NAME slat COMMAND test_slat
         "${T2_GGUFS}/slat_flow_f32.gguf" "${T2_GGUFS}/shape_dec_f32.gguf"
         "${T2_DUMPS}/reference_slat.gguf")
set_tests_properties(slat PROPERTIES SKIP_RETURN_CODE 77 LABELS "model;slow"
                     ENVIRONMENT "TRELLIS2_DINODATA=${T2_DUMPS}/fixture.dinodata")
