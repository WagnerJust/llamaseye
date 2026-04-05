package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeGGUF writes a minimal synthetic GGUF file with given KV pairs.
// Supports string, int (uint32), int32, uint32, float32, bool, float64 values.
func writeGGUF(t *testing.T, kvPairs []kvPair) string {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), "test.gguf")
	f, err := os.Create(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	writeU32 := func(v uint32) {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	writeU64 := func(v uint64) {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	writeStr := func(s string) {
		writeU64(uint64(len(s)))
		if _, err := f.WriteString(s); err != nil {
			t.Fatal(err)
		}
	}

	// Magic + version
	writeU32(ggufMagic)
	writeU32(3)

	// nTensors=0, nKV=len(kvPairs)
	writeU64(0)
	writeU64(uint64(len(kvPairs)))

	// KV pairs
	for _, p := range kvPairs {
		writeStr(p.key)
		switch val := p.val.(type) {
		case string:
			writeU32(typeSTRING)
			writeStr(val)
		case int:
			writeU32(typeUINT32)
			writeU32(uint32(val))
		case uint32:
			writeU32(typeUINT32)
			writeU32(val)
		case int32:
			writeU32(typeINT32)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case float32:
			writeU32(typeFLOAT32)
			if err := binary.Write(f, binary.LittleEndian, math.Float32bits(val)); err != nil {
				t.Fatal(err)
			}
		case bool:
			writeU32(typeBOOL)
			b := uint8(0)
			if val {
				b = 1
			}
			if err := binary.Write(f, binary.LittleEndian, b); err != nil {
				t.Fatal(err)
			}
		case float64:
			writeU32(typeFLOAT64)
			if err := binary.Write(f, binary.LittleEndian, math.Float64bits(val)); err != nil {
				t.Fatal(err)
			}
		case uint64:
			writeU32(typeUINT64)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case int64:
			writeU32(typeINT64)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case uint8:
			writeU32(typeUINT8)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case int8:
			writeU32(typeINT8)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case uint16:
			writeU32(typeUINT16)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case int16:
			writeU32(typeINT16)
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatal(err)
			}
		case []int: // int array
			writeU32(typeARRAY)
			writeU32(typeUINT32)
			writeU64(uint64(len(val)))
			for _, v := range val {
				writeU32(uint32(v))
			}
		case []bool: // bool array (for sliding_window_pattern)
			writeU32(typeARRAY)
			writeU32(typeBOOL)
			writeU64(uint64(len(val)))
			for _, v := range val {
				b := uint8(0)
				if v {
					b = 1
				}
				if err := binary.Write(f, binary.LittleEndian, b); err != nil {
					t.Fatal(err)
				}
			}
		default:
			t.Fatalf("unsupported KV type for %q: %T", p.key, val)
		}
	}
	return tmp
}

type kvPair struct{ key string; val any }

func kv(key string, val any) kvPair { return kvPair{key, val} }

func TestParse_Basic(t *testing.T) {
	path := writeGGUF(t, []kvPair{
		kv("general.architecture", "llama"),
		kv("llama.block_count", int(32)),
		kv("llama.attention.head_count", int(32)),
		kv("llama.attention.head_count_kv", int(8)),
		kv("llama.embedding_length", int(4096)),
	})

	meta, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if meta.Architecture != "llama" {
		t.Errorf("Architecture = %q, want llama", meta.Architecture)
	}
	if meta.NumLayers != 32 {
		t.Errorf("NumLayers = %d, want 32", meta.NumLayers)
	}
	if meta.KVHeadsMax != 8 {
		t.Errorf("KVHeadsMax = %d, want 8", meta.KVHeadsMax)
	}
	// KeyLen derived: 4096/32 = 128
	if meta.KeyLen != 128 {
		t.Errorf("KeyLen = %d, want 128", meta.KeyLen)
	}
	if meta.HasHybrid {
		t.Error("HasHybrid = true, want false")
	}
}

func TestParse_AllScalarTypes(t *testing.T) {
	// Test that we can parse all scalar types without errors
	path := writeGGUF(t, []kvPair{
		kv("general.architecture", "test"),
		kv("test.block_count", uint32(16)),
		kv("test.attention.head_count", int32(16)),
		kv("test.float32_val", float32(1.5)),
		kv("test.float64_val", float64(2.5)),
		kv("test.bool_val", true),
		kv("test.uint64_val", uint64(1000)),
		kv("test.int64_val", int64(2000)),
		kv("test.uint8_val", uint8(5)),
		kv("test.int8_val", int8(6)),
		kv("test.uint16_val", uint16(100)),
		kv("test.int16_val", int16(200)),
	})
	meta, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if meta.Architecture != "test" {
		t.Errorf("Architecture = %q", meta.Architecture)
	}
}

func TestParse_HybridAttention(t *testing.T) {
	// Gemma4-style hybrid attention: sliding_window_pattern array of bools
	// 2 global + 2 SWA layers
	path := writeGGUF(t, []kvPair{
		kv("general.architecture", "gemma4"),
		kv("gemma4.block_count", int(4)),
		kv("gemma4.attention.head_count", int(8)),
		kv("gemma4.attention.head_count_kv", int(4)),
		kv("gemma4.embedding_length", int(2048)),
		kv("gemma4.attention.sliding_window", int(4096)),
		kv("gemma4.attention.sliding_window_pattern", []bool{true, false, true, false}),
	})
	meta, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !meta.HasHybrid {
		t.Error("HasHybrid = false, want true for Gemma4 model")
	}
	if meta.NSWALayers != 2 {
		t.Errorf("NSWALayers = %d, want 2", meta.NSWALayers)
	}
	if meta.NGlobalLayers != 2 {
		t.Errorf("NGlobalLayers = %d, want 2", meta.NGlobalLayers)
	}
}

func TestParse_HybridWithPerLayerKVHeads(t *testing.T) {
	// Per-layer head_count_kv array
	path := writeGGUF(t, []kvPair{
		kv("general.architecture", "gemma4"),
		kv("gemma4.block_count", int(4)),
		kv("gemma4.attention.head_count", int(8)),
		kv("gemma4.attention.head_count_kv", []int{4, 16, 4, 16}), // alternating
		kv("gemma4.embedding_length", int(2048)),
		kv("gemma4.attention.sliding_window", int(4096)),
		kv("gemma4.attention.sliding_window_pattern", []bool{true, false, true, false}),
	})
	meta, err := Parse(path)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !meta.HasHybrid {
		t.Error("HasHybrid = false")
	}
	// SWA layers (indices 0,2) have head_kv=4
	if meta.SWAKVHeads != 4 {
		t.Errorf("SWAKVHeads = %d, want 4", meta.SWAKVHeads)
	}
	// Global layers (indices 1,3) have head_kv=16
	if meta.GlobalKVHeads != 16 {
		t.Errorf("GlobalKVHeads = %d, want 16", meta.GlobalKVHeads)
	}
}

func TestParse_NotGGUF(t *testing.T) {
	tmp := filepath.Join(t.TempDir(), "notgguf.bin")
	if err := os.WriteFile(tmp, []byte("this is not a gguf file"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := Parse(tmp)
	if err == nil {
		t.Error("expected error for non-GGUF file, got nil")
	}
}

func TestParse_MissingFile(t *testing.T) {
	_, err := Parse("/nonexistent/path/model.gguf")
	if err == nil {
		t.Error("expected error for missing file, got nil")
	}
}

func TestPredict_Basic(t *testing.T) {
	meta := &Metadata{
		Architecture:  "llama",
		FileGiB:       8.0,
		NumLayers:     32,
		KVHeadsMax:    8,
		KeyLen:        128,
		ValLen:        128,
		NGlobalLayers: 32,
		GlobalKVHeads: 8,
		GlobalHeadDim: 128,
	}

	result := Predict(meta, 24, 64)
	if result.MaxNGLPred <= 0 {
		t.Errorf("MaxNGLPred = %d, expected > 0", result.MaxNGLPred)
	}
	if result.MaxNGLPred > 32 {
		t.Errorf("MaxNGLPred = %d, cannot exceed NumLayers=32", result.MaxNGLPred)
	}
	if result.BestCtxVRAM <= 0 {
		t.Errorf("BestCtxVRAM = %d, expected > 0", result.BestCtxVRAM)
	}
}

func TestPredict_ZeroVRAM(t *testing.T) {
	meta := &Metadata{
		Architecture:  "llama",
		FileGiB:       8.0,
		NumLayers:     32,
		KVHeadsMax:    8,
		KeyLen:        128,
		NGlobalLayers: 32,
		GlobalKVHeads: 8,
		GlobalHeadDim: 128,
	}
	result := Predict(meta, 0, 32)
	if result.MaxNGLPred != 0 {
		t.Errorf("MaxNGLPred = %d, want 0 for no VRAM", result.MaxNGLPred)
	}
}

func TestPredict_ZeroLayers(t *testing.T) {
	meta := &Metadata{FileGiB: 1.0, NumLayers: 0}
	result := Predict(meta, 24, 64)
	if result.MaxNGLPred != 0 {
		t.Errorf("MaxNGLPred = %d, want 0 for zero layers", result.MaxNGLPred)
	}
}

func TestPredict_HybridModel(t *testing.T) {
	meta := &Metadata{
		Architecture:  "gemma4",
		FileGiB:       10.0,
		NumLayers:     26,
		KVHeadsMax:    16,
		KeyLen:        256,
		HasHybrid:     true,
		NSWALayers:    13,
		NGlobalLayers: 13,
		SWAKVHeads:    4,
		GlobalKVHeads: 16,
		SWAHeadDim:    256,
		GlobalHeadDim: 256,
		SlidingWin:    4096,
	}
	result := Predict(meta, 24, 64)
	// Should produce valid (non-negative) results
	if result.MaxNGLPred < 0 {
		t.Errorf("MaxNGLPred = %d, should be >= 0", result.MaxNGLPred)
	}
}

func TestPredict_StartCtxBelowPredicted(t *testing.T) {
	meta := &Metadata{
		Architecture:  "llama",
		FileGiB:       4.0,
		NumLayers:     16,
		KVHeadsMax:    4,
		KeyLen:        64,
		NGlobalLayers: 16,
		GlobalKVHeads: 4,
		GlobalHeadDim: 64,
	}
	result := Predict(meta, 48, 128) // plenty of VRAM
	// StartCtx should be one step below BestCtxVRAM
	if result.StartCtx >= result.BestCtxVRAM && result.BestCtxVRAM > 128 {
		t.Errorf("StartCtx=%d should be < BestCtxVRAM=%d", result.StartCtx, result.BestCtxVRAM)
	}
}
