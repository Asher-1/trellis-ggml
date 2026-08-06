# Submodules & patches

How this repo vendors and patches its git submodules, and the materialization
approach that keeps submodule working trees pristine.

## TL;DR

- There is **one** shared `ggml` (`third_party/ggml`), pinned to
  `90951f99` (v0.18.1).
- All required fixes are applied **automatically at CMake configure time** via
  `patches/` — no manual step, no submodule working tree is ever modified.
- `third_party/RMBG-2.0-GGML` is pinned to `99980fe` and **reuses the shared
  ggml** instead of pulling its own nested copy.
- Never add a second ggml target. If a submodule bundles its own ggml, patch it
  to reuse the host's ggml target (see the RMBG adapt patch).

## Patch list

| Patch | Applies to | Purpose |
|-------|-----------|---------|
| `patches/ggml-cuda-cpy-q8_0.patch` | `third_party/ggml` | Add `ggml_cpy` for Q8_0 → Q8_0 on CUDA (needed by the trellis2 pipeline; upstream CUDA backend lacks it). |
| `patches/ggml-rmbg-ops.patch` | `third_party/ggml` | RMBG custom operators (deform-im2col, swin qkv/windows, ...) — **auto-synced** from RMBG upstream's `third_party/ggml-rmbg.patch` at configure time. |
| `patches/rmbg-cmake-preprocess.patch` | `third_party/RMBG-2.0-GGML` | Make the RMBG submodule reuse the host's shared `ggml` target (instead of `add_subdirectory`-ing its own, which would collide) and take stb headers from the host's centralized `stb/`. |

## How patches are applied (materialization, not in-place)

Patches are **not** applied to the submodule working tree. Instead
`CMakeLists.txt` defines `_trellis_materialize_submodule()` which does this:

1. `git archive HEAD` of the submodule → tar into `${CMAKE_BINARY_DIR}/_deps/`.
2. Extract the tar into a build-tree directory named by a **fingerprint**:
   `{submodule-name}-{commit-short}-{patch-SHA256-short}`.
3. `git apply` every patch onto the **extracted copy** (the submodule working
   tree is never touched).
4. Cache the fingerprint via a `.trellis-patch-stamp` so that only a changed
   commit or changed patch triggers re-materialization.

```sh
git submodule update --init --recursive   # ggml + RMBG-2.0-GGML
cmake -B build-cuda -DGGML_CUDA=ON ...    # materialization + patches → build tree
```

### Auto-sync of the RMBG ops patch

The RMBG submodule ships its own `third_party/ggml-rmbg.patch` (the source of
truth for `patches/ggml-rmbg-ops.patch`). At every `cmake -B`, the build system
compares their SHA256 hashes and copies the upstream file if they differ. This
means **bumping the RMBG submodule automatically updates the local patch** — no
manual copy step.

## Why there is exactly one ggml (the "two ggml" pitfall)

`RMBG-2.0-GGML` upstream switched to a **self-contained** design: it bundles its
own `third_party/ggml` submodule and `add_subdirectory`s it, plus it ships its
own RMBG-op patch. If the host simply bumped the RMBG gitlink and let RMBG build
that nested ggml, CMake would see **two targets named `ggml`** (the host's and
RMBG's) → target collision, and the build breaks.

The fix:

- The host owns the single `ggml` and applies **both** patches to it (Q8 copy +
  RMBG ops). Both patches are disjoint (Q8 only touches `src/ggml-cuda/cpy.cu`;
  RMBG touches `include/ggml.h`, `src/ggml-cuda/*`, `src/ggml-vulkan/*`).
- `patches/rmbg-cmake-preprocess.patch` rewrites the RMBG CMakeLists so it
  branches on `if(NOT TARGET ggml)`: when built inside trellis-ggml the `ggml`
  target already exists, so RMBG reuses it and points its stb include at the
  host's `stb/`; only when built standalone (`if(NOT TARGET ggml)`) does it pull
  its own nested ggml.

## Operational rules (read before touching submodules)

1. **Bump the RMBG gitlink deliberately.** `git submodule update --remote` or a
   plain `update` will move RMBG to whatever commit its upstream `dev` points at.
   Because upstream is a moving target, a fresh clone may not match this doc.
   Always **pin the exact commit** and record it:

   ```sh
   git -C third_party/RMBG-2.0-GGML fetch origin
   git -C third_party/RMBG-2.0-GGML checkout <new-commit>
   git add third_party/RMBG-2.0-GGML
   ```

2. **Re-verify the adapt patch still applies.** If the new RMBG commit changed
   its `CMakeLists.txt` or `src/preprocess.cpp`, `patches/rmbg-cmake-preprocess.patch`
   may stop applying. Regenerate it from a clean check:

   ```sh
   git -C third_party/RMBG-2.0-GGML apply --check patches/rmbg-cmake-preprocess.patch \
     || echo "patch is stale, regenerate"
   ```

3. **Never introduce a second ggml target.** If a dependency bundles its own
   ggml, patch it to reuse the host's `ggml` target (mirror the RMBG adapt
   patch) rather than accepting a nested copy.

4. **Keep the two ggml patches non-overlapping.** If a future ggml bump causes
   `ggml-rmbg-ops.patch` and `ggml-cuda-cpy-q8_0.patch` to touch the same file,
   reconcile them into one combined patch.

5. **The `ggml-rmbg-ops.patch` is auto-synced.** Do not edit
   `patches/ggml-rmbg-ops.patch` directly — modify the RMBG submodule's
   `third_party/ggml-rmbg.patch` instead, or if the change is trellis-local,
   edit it and note in the commit message that it will be overwritten by the
   next auto-sync (the CMake output logs "Synced patches/ggml-rmbg-ops.patch
   from RMBG submodule" when it happens).