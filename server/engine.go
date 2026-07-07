package main

// engine.go — in-process FFI to libtrellis2.so via purego (no cgo), following
// the depth-anything.cpp server pattern. The C ABI is trellis2_capi.h; the
// t2_abi_version binding guards against header/library drift.

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const abiVersion = 2

// Progress stages (enum t2_stage).
const (
	stagePreprocess = 0
	stageDino       = 1
	stageSSFlow     = 2
	stageSSDec      = 3
	stageSLATFlow   = 4
	stageShapeDec   = 5
	stageMesh       = 6
	stageUpsample   = 7
	stageSLATFlowHR = 8
	stageShapeDecHR = 9
)

var stageNames = map[int]string{
	stagePreprocess: "preprocess",
	stageDino:       "encoding image (DINOv3)",
	stageSSFlow:     "sampling sparse structure",
	stageSSDec:      "decoding occupancy",
	stageSLATFlow:   "sampling shape SLAT",
	stageShapeDec:   "decoding shape",
	stageMesh:       "extracting mesh",
	stageUpsample:   "upsampling scaffold",
	stageSLATFlowHR: "sampling shape SLAT (1024)",
	stageShapeDecHR: "decoding shape (1024)",
}

// Pipeline types (enum t2_pipeline_type) and capability bits (enum t2_caps).
const (
	pipeAuto   = 0
	pipeCoarse = 1
	pipe512    = 2
	pipe1024   = 3

	capCoarse = 1
	cap512    = 2
	cap1024   = 4
)

type engine struct {
	// inference is not thread-safe: one generation at a time.
	mu sync.Mutex

	pipeline uintptr
	backend  string
	caps     int // bitmask of t2_caps

	abiVersion      func() int32
	pipelineLoad    func(dino, flow, dec, slat, slatHR, shapeDec string, flags int32, err unsafe.Pointer, errLen int32) uintptr
	pipelineFree    func(p uintptr)
	pipelineBackend func(p uintptr) string
	pipelineCaps    func(p uintptr) int32
	generate        func(p uintptr, img unsafe.Pointer, imgLen int32, pipelineType int32, seed uint64,
		steps int32, guidance float32, cb uintptr, user unsafe.Pointer,
		err unsafe.Pointer, errLen int32) uintptr
	meshNVerts   func(r uintptr) int32
	meshNTris    func(r uintptr) int32
	meshVerts    func(r uintptr) uintptr
	meshNormals  func(r uintptr) uintptr
	meshTris     func(r uintptr) uintptr
	meshFree     func(r uintptr)
}

// progressSink receives per-stage/step updates for the currently running
// generation. Exactly one generation runs at a time (engine.mu), so a single
// global callback + current sink is safe.
var (
	progressMu   sync.Mutex
	progressSink func(stage, step, total int)
)

var progressCallback uintptr // created once; purego callbacks are permanent

// slatGGUF/shapeDecGGUF may be "" for the coarse path; slatHRGGUF may be "" to
// disable the 1024 cascade (512 fine only).
func newEngine(libPath, dinoGGUF, flowGGUF, decGGUF, slatGGUF, slatHRGGUF, shapeDecGGUF string) (*engine, error) {
	lib, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", libPath, err)
	}

	e := &engine{}
	purego.RegisterLibFunc(&e.abiVersion, lib, "t2_abi_version")
	if got := e.abiVersion(); got != abiVersion {
		return nil, fmt.Errorf("ABI mismatch: library reports %d, server built for %d", got, abiVersion)
	}
	purego.RegisterLibFunc(&e.pipelineLoad, lib, "t2_pipeline_load")
	purego.RegisterLibFunc(&e.pipelineFree, lib, "t2_pipeline_free")
	purego.RegisterLibFunc(&e.pipelineBackend, lib, "t2_pipeline_backend")
	purego.RegisterLibFunc(&e.pipelineCaps, lib, "t2_pipeline_caps")
	purego.RegisterLibFunc(&e.generate, lib, "t2_generate")
	purego.RegisterLibFunc(&e.meshNVerts, lib, "t2_mesh_n_verts")
	purego.RegisterLibFunc(&e.meshNTris, lib, "t2_mesh_n_tris")
	purego.RegisterLibFunc(&e.meshVerts, lib, "t2_mesh_verts")
	purego.RegisterLibFunc(&e.meshNormals, lib, "t2_mesh_normals")
	purego.RegisterLibFunc(&e.meshTris, lib, "t2_mesh_tris")
	purego.RegisterLibFunc(&e.meshFree, lib, "t2_mesh_free")

	if progressCallback == 0 {
		progressCallback = purego.NewCallback(func(user unsafe.Pointer, stage, step, total int32) uintptr {
			progressMu.Lock()
			sink := progressSink
			progressMu.Unlock()
			if sink != nil {
				sink(int(stage), int(step), int(total))
			}
			return 0
		})
	}

	errBuf := make([]byte, 512)
	p := e.pipelineLoad(dinoGGUF, flowGGUF, decGGUF, slatGGUF, slatHRGGUF, shapeDecGGUF,
		0 /*flags*/, unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if p == 0 {
		return nil, fmt.Errorf("pipeline load: %s", cstr(errBuf))
	}
	e.pipeline = p
	e.backend = e.pipelineBackend(p)
	e.caps = int(e.pipelineCaps(p))
	return e, nil
}

type meshData struct {
	NVerts  int
	NTris   int
	Verts   []float32 // 3 * NVerts
	Normals []float32 // 3 * NVerts
	Tris    []int32   // 3 * NTris
}

// Generate runs the full image->mesh pipeline. onProgress may be nil.
func (e *engine) Generate(image []byte, pipelineType int, seed uint64, steps int, guidance float32,
	onProgress func(stage, step, total int)) (*meshData, error) {

	e.mu.Lock()
	defer e.mu.Unlock()

	progressMu.Lock()
	progressSink = onProgress
	progressMu.Unlock()
	defer func() {
		progressMu.Lock()
		progressSink = nil
		progressMu.Unlock()
	}()

	cb := uintptr(0)
	if onProgress != nil {
		cb = progressCallback
	}

	errBuf := make([]byte, 512)
	r := e.generate(e.pipeline, unsafe.Pointer(&image[0]), int32(len(image)), int32(pipelineType),
		seed, int32(steps), guidance, cb, nil,
		unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if r == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.meshFree(r)

	nv := int(e.meshNVerts(r))
	nt := int(e.meshNTris(r))
	if nv == 0 || nt == 0 {
		return nil, fmt.Errorf("empty mesh")
	}

	m := &meshData{NVerts: nv, NTris: nt}
	m.Verts = copyFloats(e.meshVerts(r), 3*nv)
	m.Normals = copyFloats(e.meshNormals(r), 3*nv)
	m.Tris = copyInts(e.meshTris(r), 3*nt)
	return m, nil
}

func copyFloats(p uintptr, n int) []float32 {
	src := unsafe.Slice((*float32)(unsafe.Pointer(p)), n)
	dst := make([]float32, n)
	copy(dst, src)
	return dst
}

func copyInts(p uintptr, n int) []int32 {
	src := unsafe.Slice((*int32)(unsafe.Pointer(p)), n)
	dst := make([]int32, n)
	copy(dst, src)
	return dst
}

func cstr(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
