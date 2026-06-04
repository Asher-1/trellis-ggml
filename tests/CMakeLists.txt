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

# Self-contained isosurface extractor (examples/marching_cubes.h) on analytic
# fields — watertight-manifold + Euler-characteristic invariants. No model.
add_executable(test_marching_cubes test_marching_cubes.cpp)
target_include_directories(test_marching_cubes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples)
target_compile_features(test_marching_cubes PRIVATE cxx_std_14)
