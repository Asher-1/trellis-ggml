# Example: run a single SS-flow DiT forward on supplied x/t/cond and dump
# the velocity output for cross-backend numerics comparison.
add_executable(t2_forward_probe t2_forward_probe.cpp)
target_link_libraries(t2_forward_probe PRIVATE trellis2)
target_compile_features(t2_forward_probe PRIVATE cxx_std_14)

# Example: dump coarse-pipeline stage intermediates for cross-backend
# numerical comparison (preprocess/DINO/SS-flow/SS-dec).
add_executable(t2_stage_probe t2_stage_probe.cpp)
target_link_libraries(t2_stage_probe PRIVATE trellis2)
target_compile_features(t2_stage_probe PRIVATE cxx_std_14)

add_executable(shape_dec_probe shape_dec_probe.cpp)
target_link_libraries(shape_dec_probe PRIVATE trellis2)
target_compile_features(shape_dec_probe PRIVATE cxx_std_14)

# Example: inspect a .dinodata DINOv3 conditioning file.
add_executable(dino_info dino_info.cpp)
target_link_libraries(dino_info PRIVATE trellis2)
target_compile_features(dino_info PRIVATE cxx_std_14)

# Example: inspect a converted SS-flow DiT GGUF (hparams + tensor inventory).
add_executable(ss_flow_info ss_flow_info.cpp)
target_link_libraries(ss_flow_info PRIVATE trellis2)
target_compile_features(ss_flow_info PRIVATE cxx_std_14)

# Example: stage-1 sampling, .dinodata cond -> sparse-structure latent z_s.
add_executable(ss_sample ss_sample.cpp)
target_link_libraries(ss_sample PRIVATE trellis2)
target_compile_features(ss_sample PRIVATE cxx_std_14)

# Example: stage-1 decoding, z_s latent -> occupancy logit grid (64^3).
add_executable(ss_decode ss_decode.cpp)
target_link_libraries(ss_decode PRIVATE trellis2)
target_compile_features(ss_decode PRIVATE cxx_std_14)

# Example: decode + extract the occupancy isosurface as an OBJ (marching cubes).
add_executable(ss_mesh ss_mesh.cpp)
target_link_libraries(ss_mesh PRIVATE trellis2)
target_compile_features(ss_mesh PRIVATE cxx_std_14)

# Example: image -> preprocessed 512x512 -> DINOv3 conditioning (.dinodata).
add_executable(dino_encode dino_encode.cpp)
target_link_libraries(dino_encode PRIVATE trellis2)
target_compile_features(dino_encode PRIVATE cxx_std_14)
if(WebP_FOUND)
    target_include_directories(dino_encode PRIVATE ${WebP_INCLUDE_DIRS})
    target_link_libraries(dino_encode PRIVATE ${WebP_LIBRARIES})
    target_compile_definitions(dino_encode PRIVATE TRELLIS2_HAVE_WEBP=1)
endif()

# Example: export a demo mesh (T2MESH01/02/03) into a portable GLB (no GPU).
add_executable(mesh2glb mesh2glb.cpp)
target_link_libraries(mesh2glb PRIVATE trellis2)
target_compile_features(mesh2glb PRIVATE cxx_std_14)

# Example: full image -> 3D mesh pipeline (same t2_generate path as the demo server).
add_executable(t2_generate t2_generate.cpp)
target_link_libraries(t2_generate PRIVATE trellis2)
target_compile_features(t2_generate PRIVATE cxx_std_14)

# Example: standalone PBR texturing (mesh + T2GRID sidecar + image).
add_executable(t2_texture t2_texture.cpp)
target_link_libraries(t2_texture PRIVATE trellis2)
target_compile_features(t2_texture PRIVATE cxx_std_14)
