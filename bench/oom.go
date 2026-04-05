package bench

import (
	"os"
	"regexp"
)

// oomPattern matches known OOM and fatal error messages from llama.cpp / CUDA / Metal.
var oomPattern = regexp.MustCompile(`(?i)(out of memory|failed to allocate|ggml_cuda_pool_alloc|CUDA error|cudaMalloc failed|ggml_backend_alloc|Cannot allocate memory|Killed|Segmentation fault|bus error|terminate called|GGML_ASSERT|failed to load model)`)

// DetectOOM returns true if the file at path contains an OOM or fatal error string.
// Returns false if the file does not exist.
func DetectOOM(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return oomPattern.Match(data)
}

// DetectOOMBytes returns true if the byte slice contains an OOM / fatal error string.
func DetectOOMBytes(data []byte) bool {
	return oomPattern.Match(data)
}
