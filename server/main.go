// trellis2 demo server: upload an image, get a 3D mesh back.
//
//	cmake -B build-shared -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-shared -j
//	cd server && CGO_ENABLED=0 go build -o trellis2-server .
//	./trellis2-server -lib ../build-shared/libtrellis2.so -ggufs ../ggufs
//
// API:
//	GET  /                 self-contained WebGL viewer (embedded web/index.html)
//	GET  /api/info         {backend, defaults}
//	POST /api/generate     multipart image [+ seed, steps, guidance] -> {job}
//	GET  /api/job/{id}     {state, stage, step, total, error, queue}
//	GET  /api/mesh/{id}    binary mesh: "T2MESH01" u32 nv u32 nt f32[3nv] verts
//	                       f32[3nv] normals u32[3nt] tris (little-endian)
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

	image    []byte
	seed     uint64
	steps    int
	guidance float32
	mesh     *meshData
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

		mesh, err := s.eng.Generate(j.image, j.seed, j.steps, j.guidance,
			func(stage, step, total int) {
				j.mu.Lock()
				j.Stage = stageNames[stage]
				j.Step = step
				j.Total = total
				j.mu.Unlock()
			})

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

	j := &job{
		ID:       fmt.Sprintf("%08x", rand.Uint32()),
		State:    "queued",
		image:    img,
		seed:     formUint(r, "seed", rand.Uint64()%1_000_000),
		steps:    int(formUint(r, "steps", 12)),
		guidance: formFloat(r, "guidance", 7.5),
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

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write([]byte("T2MESH01"))
	binary.Write(w, binary.LittleEndian, uint32(mesh.NVerts))
	binary.Write(w, binary.LittleEndian, uint32(mesh.NTris))
	binary.Write(w, binary.LittleEndian, mesh.Verts)
	binary.Write(w, binary.LittleEndian, mesh.Normals)
	binary.Write(w, binary.LittleEndian, mesh.Tris)
}

func (s *server) handleInfo(w http.ResponseWriter, r *http.Request) {
	mesh := "coarse"
	if s.eng.fine {
		mesh = "fine"
	}
	writeJSON(w, map[string]interface{}{
		"backend": s.eng.backend,
		"mesh":    mesh,
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
	slat := flag.String("slat", "", "shape-slat flow gguf (default <ggufs>/slat_flow_f16.gguf)")
	shapeDec := flag.String("shape-dec", "", "shape decoder gguf (default <ggufs>/shape_dec_f16.gguf)")
	coarse := flag.Bool("coarse", false, "coarse marching-cubes path only (skip shape-SLAT models)")
	addr := flag.String("addr", ":8742", "listen address")
	flag.Parse()

	pick := func(explicit, name string) string {
		if explicit != "" {
			return explicit
		}
		return filepath.Join(*ggufDir, name)
	}
	// The fine (dual-grid) path needs the two shape-SLAT models; if either is
	// missing on disk, fall back to the coarse path automatically.
	slatPath, shapePath := "", ""
	if !*coarse {
		slatPath = pick(*slat, "slat_flow_f16.gguf")
		shapePath = pick(*shapeDec, "shape_dec_f16.gguf")
		if !fileExists(slatPath) || !fileExists(shapePath) {
			log.Printf("shape-SLAT models not found, using coarse path")
			slatPath, shapePath = "", ""
		}
	}

	eng, err := newEngine(*libPath, pick(*dino, "dino_f16.gguf"),
		pick(*flow, "ss_flow_f16.gguf"), pick(*dec, "ss_dec_f16.gguf"),
		slatPath, shapePath)
	if err != nil {
		log.Fatal(err)
	}
	mode := "coarse (marching cubes)"
	if eng.fine {
		mode = "fine (dual-grid)"
	}
	log.Printf("models loaded, backend: %s, mesh: %s", eng.backend, mode)

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

	log.Printf("listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
