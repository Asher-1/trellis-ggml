// trellis2 demo server: upload an image, get a 3D mesh back.
//
//	cmake -B build-shared -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-shared -j
//	cd server && CGO_ENABLED=0 go build -o trellis2-server .
//	./trellis2-server -lib ../build-shared/libtrellis2.so -ggufs ../ggufs
//
// API:
//	GET  /                 self-contained WebGL viewer (embedded web/index.html)
//	GET  /api/info         {backend, defaults}
//	POST /api/generate     multipart image [+ seed, steps, guidance, preview] -> {job}
//	GET  /api/job/{id}     {state, stage, step, total, previewSeq, error}
//	GET  /api/mesh/{id}    binary mesh: "T2MESH01" u32 nv u32 nt f32[3nv] verts
//	                       f32[3nv] normals u32[3nt] tris (little-endian)
//	GET  /api/preview/{id} latest live 3D preview: "T2VOX01" u32 res u32 nvox
//	                       u16[3nvox] voxel coords (little-endian)
package main

import (
	"embed"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

//go:embed web
var webFS embed.FS

const maxUpload = 32 << 20 // 32 MiB

type job struct {
	mu    sync.Mutex
	ID    string `json:"id"`
	State string `json:"state"` // queued | running | done | error
	Stage string `json:"stage,omitempty"`
	Step  int    `json:"step"`
	Total int    `json:"total"`
	Error string `json:"error,omitempty"`

	// Live intermediate 3D preview: PreviewSeq bumps whenever a fresh voxel blob
	// lands (the browser watches it to know when to fetch /api/preview).
	PreviewSeq   int    `json:"previewSeq"`
	PreviewStage string `json:"previewStage,omitempty"`
	PreviewStep  int    `json:"previewStep"`
	PreviewTotal int    `json:"previewTotal"`

	image       []byte
	pipeline    int
	seed        uint64
	steps       int
	guidance    float32
	wantPreview bool
	preview     []byte // latest preview blob (T2VOX01); served by /api/preview
	mesh        *meshData
	glb         []byte // cached last GLB bake
	glbKey      string // "tex-tris" the cached GLB was baked with
}

type server struct {
	eng  *engine
	mu   sync.Mutex
	jobs map[string]*job
	q    chan *job
}

func (s *server) worker() {
	for j := range s.q {
		j.mu.Lock()
		j.State = "running"
		j.mu.Unlock()

		var onPreview func(stage, step, total int, blob []byte)
		if j.wantPreview {
			onPreview = func(stage, step, total int, blob []byte) {
				j.mu.Lock()
				j.preview = blob
				j.PreviewSeq++
				j.PreviewStage = stageNames[stage]
				j.PreviewStep = step
				j.PreviewTotal = total
				j.mu.Unlock()
			}
		}

		mesh, err := s.eng.Generate(j.image, j.pipeline, j.seed, j.steps, j.guidance,
			func(stage, step, total int) {
				j.mu.Lock()
				j.Stage = stageNames[stage]
				j.Step = step
				j.Total = total
				j.mu.Unlock()
			},
			onPreview)

		j.mu.Lock()
		j.image = nil
		if err != nil {
			j.State = "error"
			j.Error = err.Error()
		} else {
			j.State = "done"
			j.mesh = mesh
		}
		j.mu.Unlock()
	}
}

func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)
	if err := r.ParseMultipartForm(maxUpload); err != nil {
		http.Error(w, "bad multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "missing image field", http.StatusBadRequest)
		return
	}
	defer file.Close()
	img, err := io.ReadAll(file)
	if err != nil || len(img) == 0 {
		http.Error(w, "empty image", http.StatusBadRequest)
		return
	}

	// quality: "coarse" | "512" | "1024" (default auto → best available)
	pt := pipeAuto
	switch r.FormValue("quality") {
	case "coarse":
		pt = pipeCoarse
	case "512":
		pt = pipe512
	case "1024":
		pt = pipe1024
	}

	j := &job{
		ID:          fmt.Sprintf("%08x", rand.Uint32()),
		State:       "queued",
		image:       img,
		pipeline:    pt,
		seed:        formUint(r, "seed", rand.Uint64()%1_000_000),
		steps:       int(formUint(r, "steps", 12)),
		guidance:    formFloat(r, "guidance", 7.5),
		wantPreview: r.FormValue("preview") != "0", // live 3D previews unless opted out
	}
	if j.steps < 1 || j.steps > 50 {
		j.steps = 12
	}
	if j.guidance < 0 || j.guidance > 20 {
		j.guidance = 7.5
	}

	s.mu.Lock()
	s.jobs[j.ID] = j
	s.mu.Unlock()

	select {
	case s.q <- j:
	default:
		s.mu.Lock()
		delete(s.jobs, j.ID)
		s.mu.Unlock()
		http.Error(w, "queue full, try again later", http.StatusServiceUnavailable)
		return
	}

	writeJSON(w, map[string]string{"job": j.ID})
}

func (s *server) getJob(r *http.Request, prefix string) *job {
	id := r.URL.Path[len(prefix):]
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

func (s *server) handleJob(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/job/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	writeJSON(w, j)
}

func (s *server) handleMesh(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/mesh/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	j.mu.Lock()
	mesh := j.mesh
	j.mu.Unlock()
	if mesh == nil {
		http.Error(w, "mesh not ready", http.StatusConflict)
		return
	}

	// Wire format. T2MESH01: verts, normals, tris. T2MESH02 (textured): also a
	// per-vertex PBR block (5 floats: base_color rgb, metallic, roughness) after
	// the normals, before the tris.
	//   magic[8] u32 nv u32 nt  f32[3nv] verts  f32[3nv] normals
	//   [T2MESH02: f32[5nv] pbr]  i32[3nt] tris
	w.Header().Set("Content-Type", "application/octet-stream")
	textured := len(mesh.PBR) == 5*mesh.NVerts
	if textured {
		w.Write([]byte("T2MESH02"))
	} else {
		w.Write([]byte("T2MESH01"))
	}
	binary.Write(w, binary.LittleEndian, uint32(mesh.NVerts))
	binary.Write(w, binary.LittleEndian, uint32(mesh.NTris))
	binary.Write(w, binary.LittleEndian, mesh.Verts)
	binary.Write(w, binary.LittleEndian, mesh.Normals)
	if textured {
		binary.Write(w, binary.LittleEndian, mesh.PBR)
	}
	binary.Write(w, binary.LittleEndian, mesh.Tris)
}

// handlePreview streams the latest live intermediate-preview blob (a T2VOX01
// voxel set) for a running job. The browser polls /api/job for previewSeq and
// fetches this whenever it advances, swapping the voxels into the viewer.
func (s *server) handlePreview(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/preview/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	j.mu.Lock()
	blob := j.preview
	j.mu.Unlock()
	if blob == nil {
		http.Error(w, "no preview yet", http.StatusConflict)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(blob)
}

// handleGLB bakes the mesh into a UV-atlas-textured GLB on demand and streams
// it as a download. Result is cached per (tex,tris) on the job so re-requests
// are free. Baking is CPU-bound and can take a few seconds for the dense mesh.
func (s *server) handleGLB(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/glb/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	j.mu.Lock()
	mesh := j.mesh
	j.mu.Unlock()
	if mesh == nil {
		http.Error(w, "mesh not ready", http.StatusConflict)
		return
	}

	tex := int(formUint(r, "tex", 2048))
	tris := int(formUint(r, "tris", 150000))
	if tex < 256 || tex > 4096 {
		tex = 2048
	}
	if tris < 1000 || tris > 4000000 {
		tris = 150000
	}
	key := fmt.Sprintf("%d-%d", tex, tris)

	j.mu.Lock()
	glb, cached := j.glb, j.glbKey == key && j.glb != nil
	j.mu.Unlock()

	if !cached {
		var err error
		glb, err = s.eng.BakeGLB(mesh, tex, tris)
		if err != nil {
			http.Error(w, "bake glb: "+err.Error(), http.StatusInternalServerError)
			return
		}
		j.mu.Lock()
		j.glb, j.glbKey = glb, key
		j.mu.Unlock()
	}

	w.Header().Set("Content-Type", "model/gltf-binary")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"trellis2-%s.glb\"", j.ID))
	w.Write(glb)
}

func (s *server) handleInfo(w http.ResponseWriter, r *http.Request) {
	qualities := []string{"coarse"}
	if s.eng.caps&cap512 != 0 {
		qualities = append(qualities, "512")
	}
	if s.eng.caps&cap1024 != 0 {
		qualities = append(qualities, "1024")
	}
	best := "coarse"
	if s.eng.caps&cap1024 != 0 {
		best = "1024"
	} else if s.eng.caps&cap512 != 0 {
		best = "512"
	}
	writeJSON(w, map[string]interface{}{
		"backend":   s.eng.backend,
		"qualities": qualities,
		"best":      best,
		"textured":  s.eng.textured,
		"defaults": map[string]interface{}{
			"steps":    12,
			"guidance": 7.5,
		},
	})
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func formUint(r *http.Request, key string, def uint64) uint64 {
	if v := r.FormValue(key); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

func formFloat(r *http.Request, key string, def float32) float32 {
	if v := r.FormValue(key); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			return float32(f)
		}
	}
	return def
}

func main() {
	libPath := flag.String("lib", "../build-shared/libtrellis2.so", "path to libtrellis2.so")
	ggufDir := flag.String("ggufs", "../ggufs", "directory with the model ggufs")
	dino := flag.String("dino", "", "dino gguf (default <ggufs>/dino_f16.gguf)")
	flow := flag.String("flow", "", "ss_flow gguf (default <ggufs>/ss_flow_f16.gguf)")
	dec := flag.String("dec", "", "ss_dec gguf (default <ggufs>/ss_dec_f16.gguf)")
	slat := flag.String("slat", "", "512 shape-slat flow gguf (default <ggufs>/slat_flow_f16.gguf)")
	slatHR := flag.String("slat-hr", "", "1024 shape-slat flow gguf (default <ggufs>/slat_flow_1024_f16.gguf)")
	shapeDec := flag.String("shape-dec", "", "shape decoder gguf (default <ggufs>/shape_dec_f16.gguf)")
	shapeEnc := flag.String("shape-enc", "", "shape encoder gguf (default <ggufs>/shape_enc_f16.gguf)")
	texDec := flag.String("tex-dec", "", "texture decoder gguf (default <ggufs>/tex_dec_f16.gguf)")
	texSlat := flag.String("tex-slat", "", "512 texture-slat flow gguf (default <ggufs>/tex_slat_flow_512_f16.gguf)")
	texSlatHR := flag.String("tex-slat-hr", "", "1024 texture-slat flow gguf (default <ggufs>/tex_slat_flow_1024_f16.gguf)")
	coarse := flag.Bool("coarse", false, "coarse marching-cubes path only (skip shape-SLAT models)")
	no1024 := flag.Bool("no-1024", false, "disable the 1024 cascade (512 fine max)")
	noTexture := flag.Bool("no-texture", false, "disable PBR texturing (geometry only)")
	addr := flag.String("addr", ":8742", "listen address")
	flag.Parse()

	pick := func(explicit, name string) string {
		if explicit != "" {
			return explicit
		}
		return filepath.Join(*ggufDir, name)
	}
	// The fine (dual-grid) path needs the two shape-SLAT models; the 1024 cascade
	// additionally needs the 1024 model. Missing files degrade gracefully.
	slatPath, shapePath, slatHRPath := "", "", ""
	shapeEncPath, texDecPath, texSlatPath, texSlatHRPath := "", "", "", ""
	if !*coarse {
		slatPath = pick(*slat, "slat_flow_f16.gguf")
		shapePath = pick(*shapeDec, "shape_dec_f16.gguf")
		if !fileExists(slatPath) || !fileExists(shapePath) {
			log.Printf("shape-SLAT models not found, using coarse path")
			slatPath, shapePath = "", ""
		} else {
			if !*no1024 {
				slatHRPath = pick(*slatHR, "slat_flow_1024_f16.gguf")
				if !fileExists(slatHRPath) {
					log.Printf("1024 model not found, 512 fine max")
					slatHRPath = ""
				}
			}
			// PBR texturing needs the shape encoder, tex decoder, and 512 tex flow.
			if !*noTexture {
				shapeEncPath = pick(*shapeEnc, "shape_enc_f16.gguf")
				texDecPath = pick(*texDec, "tex_dec_f16.gguf")
				texSlatPath = pick(*texSlat, "tex_slat_flow_512_f16.gguf")
				if fileExists(shapeEncPath) && fileExists(texDecPath) && fileExists(texSlatPath) {
					texSlatHRPath = pick(*texSlatHR, "tex_slat_flow_1024_f16.gguf")
					if !fileExists(texSlatHRPath) {
						texSlatHRPath = ""
					}
				} else {
					log.Printf("texture models not found, geometry only")
					shapeEncPath, texDecPath, texSlatPath = "", "", ""
				}
			}
		}
	}

	eng, err := newEngine(*libPath, pick(*dino, "dino_f16.gguf"),
		pick(*flow, "ss_flow_f16.gguf"), pick(*dec, "ss_dec_f16.gguf"),
		slatPath, slatHRPath, shapePath,
		shapeEncPath, texDecPath, texSlatPath, texSlatHRPath)
	if err != nil {
		log.Fatal(err)
	}
	mode := "coarse (marching cubes)"
	if eng.caps&cap1024 != 0 {
		mode = "1024 cascade (+ 512 fine, coarse)"
	} else if eng.caps&cap512 != 0 {
		mode = "512 fine (+ coarse)"
	}
	log.Printf("models loaded, backend: %s, qualities: %s", eng.backend, mode)

	s := &server{eng: eng, jobs: map[string]*job{}, q: make(chan *job, 8)}
	go s.worker()

	web, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(web)))
	mux.HandleFunc("/api/info", s.handleInfo)
	mux.HandleFunc("/api/generate", s.handleGenerate)
	mux.HandleFunc("/api/job/", s.handleJob)
	mux.HandleFunc("/api/mesh/", s.handleMesh)
	mux.HandleFunc("/api/preview/", s.handlePreview)
	mux.HandleFunc("/api/glb/", s.handleGLB)

	log.Printf("listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
