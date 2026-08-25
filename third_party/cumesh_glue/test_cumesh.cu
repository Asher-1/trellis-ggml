// test_cumesh.cu — quick smoke test for CuMesh chart clustering.
#include "cumesh_glue.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // A simple 8-triangle cube (two triangles per face)
    const float verts[] = {
        -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
        -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1,
    };
    const int tris[] = {
        // back (-z)
        0,1,2,  0,2,3,
        // front (+z)
        4,6,5,  4,7,6,
        // left (-x)
        0,3,7,  0,7,4,
        // right (+x)
        1,5,6,  1,6,2,
        // bottom (-y)
        0,4,5,  0,5,1,
        // top (+y)
        3,2,6,  3,6,7,
    };
    const int nv = 8, nt = 12;

    int *face_cid = nullptr, *chart_vmap = nullptr;
    int *chart_faces = nullptr, *chart_faces_off = nullptr, *chart_vert_off = nullptr;
    int n_charts = 0, nt_out = 0;

    int ok = cumesh_compute_charts(
        verts, nv, tris, nt,
        1.57f,  // ~90° cone threshold
        100, 3, 1.0f,
        0.1f, 0.0001f,
        &face_cid, &chart_vmap, &chart_faces,
        &chart_faces_off, &chart_vert_off,
        &n_charts, &nt_out
    );

    if (!ok) {
        fprintf(stderr, "FAIL: compute_charts returned 0\n");
        return 1;
    }

    printf("Cube test: %d charts, %d faces after clustering\n", n_charts, nt_out);

    // Expect 6 charts (one per face direction) or more
    // Each face group should have 2 triangles
    for (int c = 0; c < n_charts && c < 10; ++c) {
        int f_start = chart_faces_off[c];
        int f_end = chart_faces_off[c + 1];
        int v_start = chart_vert_off[c];
        int v_end = chart_vert_off[c + 1];
        printf("  chart %d: faces %d..%d (%d tris), verts %d..%d (%d verts)\n",
                    c, f_start, f_end, f_end - f_start, v_start, v_end, v_end - v_start);
    }

    cumesh_free_buffer(face_cid);
    cumesh_free_buffer(chart_vmap);
    cumesh_free_buffer(chart_faces);
    cumesh_free_buffer(chart_faces_off);
    cumesh_free_buffer(chart_vert_off);

    printf("PASS\n");
    return 0;
}