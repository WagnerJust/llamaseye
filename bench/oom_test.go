package bench

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectOOMBytes_Positives(t *testing.T) {
	patterns := []string{
		"CUDA out of memory",
		"failed to allocate 4096 bytes",
		"ggml_cuda_pool_alloc: not enough memory",
		"cudaMalloc failed",
		"ggml_backend_alloc failed",
		"Cannot allocate memory",
		"Killed",
		"Segmentation fault",
		"bus error",
		"terminate called after throwing",
		"GGML_ASSERT: x != NULL",
		"failed to load model",
		"CUDA error: no kernel image is available",
	}
	for _, p := range patterns {
		if !DetectOOMBytes([]byte(p)) {
			t.Errorf("DetectOOMBytes(%q) = false, want true", p)
		}
	}
}

func TestDetectOOMBytes_Negatives(t *testing.T) {
	negatives := []string{
		"llama_model_load: loading model from ...",
		"sampling parameters:",
		"| pp 512 |",
		"avg_ts: 42.5",
		"",
	}
	for _, n := range negatives {
		if DetectOOMBytes([]byte(n)) {
			t.Errorf("DetectOOMBytes(%q) = true, want false", n)
		}
	}
}

func TestDetectOOM_File(t *testing.T) {
	dir := t.TempDir()

	oomFile := filepath.Join(dir, "oom.txt")
	if err := os.WriteFile(oomFile, []byte("CUDA error: out of memory\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if !DetectOOM(oomFile) {
		t.Error("DetectOOM on OOM file = false, want true")
	}

	okFile := filepath.Join(dir, "ok.txt")
	if err := os.WriteFile(okFile, []byte(`{"avg_ts":42.5,"n_gen":128}`+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if DetectOOM(okFile) {
		t.Error("DetectOOM on ok file = true, want false")
	}

	if DetectOOM(filepath.Join(dir, "nonexistent.txt")) {
		t.Error("DetectOOM on missing file = true, want false")
	}
}
