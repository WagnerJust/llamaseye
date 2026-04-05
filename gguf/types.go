// Package gguf implements a pure-Go GGUF metadata reader and NGL/ctx predictor.
package gguf

// Metadata holds the KV metadata fields extracted from a GGUF file header.
type Metadata struct {
	Architecture string
	FileGiB      float64
	NumLayers    int
	HeadCount    int
	KVHeadsMax   int // max head_count_kv across layers (handles per-layer arrays)
	KeyLen       int // head dimension for KV cache
	ValLen       int

	// Hybrid attention (Gemma 4 style)
	HasHybrid     bool
	NSWALayers    int
	NGlobalLayers int
	SWAKVHeads    int
	GlobalKVHeads int
	SWAHeadDim    int
	GlobalHeadDim int
	SlidingWin    int
}
