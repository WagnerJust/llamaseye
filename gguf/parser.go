package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	ggufMagic   = 0x46554747 // "GGUF"
	typeUINT8   = 0
	typeINT8    = 1
	typeUINT16  = 2
	typeINT16   = 3
	typeUINT32  = 4
	typeINT32   = 5
	typeFLOAT32 = 6
	typeBOOL    = 7
	typeSTRING  = 8
	typeARRAY   = 9
	typeUINT64  = 10
	typeINT64   = 11
	typeFLOAT64 = 12
)

// Parse reads the GGUF header from path and extracts model metadata.
func Parse(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	fileGiB := float64(fi.Size()) / (1 << 30)

	// Magic
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file (magic=%08x)", magic)
	}

	// Version
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, err
	}

	// nTensors, nKV
	var nTensors, nKV uint64
	if err := binary.Read(f, binary.LittleEndian, &nTensors); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &nKV); err != nil {
		return nil, err
	}

	// Read all KV pairs
	kv := make(map[string]any)
	for i := uint64(0); i < nKV; i++ {
		key, val, err := readKV(f)
		if err != nil {
			return nil, fmt.Errorf("KV[%d]: %w", i, err)
		}
		kv[key] = val
		// Also index by suffix after last dot
		for j := len(key) - 1; j >= 0; j-- {
			if key[j] == '.' {
				suffix := key[j+1:]
				if _, exists := kv[suffix]; !exists {
					kv[suffix] = val
				}
				break
			}
		}
	}

	arch := kvStr(kv, "general.architecture", "unknown")
	meta := &Metadata{
		Architecture: arch,
		FileGiB:      fileGiB,
		NumLayers:    kvInt(kv, "block_count", arch+".block_count"),
		HeadCount:    kvInt(kv, "head_count", arch+".attention.head_count"),
		SlidingWin:   kvInt(kv, "sliding_window", arch+".attention.sliding_window"),
	}

	// KV heads — may be an array for hybrid attention
	hckvRaw := kvLookup(kv, "head_count_kv", arch+".attention.head_count_kv")
	switch v := hckvRaw.(type) {
	case int64:
		meta.KVHeadsMax = int(v)
	case uint32:
		meta.KVHeadsMax = int(v)
	case uint64:
		meta.KVHeadsMax = int(v)
	case []any:
		for _, elem := range v {
			n := anyToInt(elem)
			if n > meta.KVHeadsMax {
				meta.KVHeadsMax = n
			}
		}
	case int:
		meta.KVHeadsMax = v
	}

	// Key/value head dimensions
	// MLA: prefer compressed dim if present
	keyLen := kvInt(kv, "key_length_mla", arch+".attention.key_length_mla")
	if keyLen == 0 {
		keyLen = kvInt(kv, "key_length", arch+".attention.key_length")
	}
	if keyLen == 0 && meta.HeadCount > 0 {
		embedding := kvInt(kv, "embedding_length", arch+".embedding_length")
		keyLen = embedding / meta.HeadCount
	}
	meta.KeyLen = keyLen

	valLen := kvInt(kv, "value_length_mla", arch+".attention.value_length_mla")
	if valLen == 0 {
		valLen = kvInt(kv, "value_length", arch+".attention.value_length")
	}
	if valLen == 0 {
		valLen = keyLen
	}
	meta.ValLen = valLen

	// Hybrid attention: sliding_window_pattern is array of bools
	winPatternRaw := kvLookup(kv, "sliding_window_pattern", arch+".attention.sliding_window_pattern")
	if arr, ok := winPatternRaw.([]any); ok && len(arr) > 0 {
		meta.HasHybrid = true
		for _, v := range arr {
			switch b := v.(type) {
			case bool:
				if b {
					meta.NSWALayers++
				} else {
					meta.NGlobalLayers++
				}
			}
		}
		meta.GlobalHeadDim = keyLen
		meta.SWAHeadDim = keyLen

		// Per-layer head_count_kv
		if hckvArr, ok := hckvRaw.([]any); ok && len(hckvArr) == len(arr) {
			var swaMax, globalMax int
			for i, v := range arr {
				if i >= len(hckvArr) {
					break
				}
				n := anyToInt(hckvArr[i])
				isSWA, _ := v.(bool)
				if isSWA {
					if n > swaMax {
						swaMax = n
					}
				} else {
					if n > globalMax {
						globalMax = n
					}
				}
			}
			meta.SWAKVHeads = swaMax
			meta.GlobalKVHeads = globalMax
		} else {
			meta.SWAKVHeads = meta.KVHeadsMax
			meta.GlobalKVHeads = meta.KVHeadsMax
		}
	} else {
		meta.NGlobalLayers = meta.NumLayers
		meta.GlobalKVHeads = meta.KVHeadsMax
		meta.SWAKVHeads = meta.KVHeadsMax
		meta.GlobalHeadDim = keyLen
		meta.SWAHeadDim = keyLen
	}

	return meta, nil
}

// readKV reads one key-value pair from the file.
func readKV(r io.Reader) (string, any, error) {
	kLen, err := readU64(r)
	if err != nil {
		return "", nil, err
	}
	keyBytes := make([]byte, kLen)
	if _, err := io.ReadFull(r, keyBytes); err != nil {
		return "", nil, err
	}
	key := string(keyBytes)

	var vtype uint32
	if err := binary.Read(r, binary.LittleEndian, &vtype); err != nil {
		return "", nil, err
	}

	val, err := readValue(r, vtype)
	return key, val, err
}

func readValue(r io.Reader, vtype uint32) (any, error) {
	switch vtype {
	case typeUINT8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeINT8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeUINT16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeINT16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeUINT32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeINT32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeFLOAT32:
		var v uint32
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return math.Float32frombits(v), nil
	case typeBOOL:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case typeSTRING:
		sLen, err := readU64(r)
		if err != nil {
			return nil, err
		}
		buf := make([]byte, sLen)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		return string(buf), nil
	case typeARRAY:
		var atype uint32
		if err := binary.Read(r, binary.LittleEndian, &atype); err != nil {
			return nil, err
		}
		aLen, err := readU64(r)
		if err != nil {
			return nil, err
		}
		elems := make([]any, aLen)
		for i := uint64(0); i < aLen; i++ {
			v, err := readValue(r, atype)
			if err != nil {
				return nil, fmt.Errorf("array[%d]: %w", i, err)
			}
			elems[i] = v
		}
		return elems, nil
	case typeUINT64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeINT64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case typeFLOAT64:
		var v uint64
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return math.Float64frombits(v), nil
	default:
		return nil, fmt.Errorf("unknown value type %d", vtype)
	}
}

func readU64(r io.Reader) (uint64, error) {
	var v uint64
	return v, binary.Read(r, binary.LittleEndian, &v)
}

// kvLookup returns the value for either of two keys.
func kvLookup(kv map[string]any, keys ...string) any {
	for _, k := range keys {
		if v, ok := kv[k]; ok {
			return v
		}
	}
	return nil
}

// kvStr returns a string from kv, checking multiple key candidates.
func kvStr(kv map[string]any, keys ...string) string {
	v := kvLookup(kv, keys...)
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// kvInt returns an int from kv, checking multiple key candidates.
func kvInt(kv map[string]any, keys ...string) int {
	return anyToInt(kvLookup(kv, keys...))
}

// anyToInt converts common numeric types to int.
func anyToInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int8:
		return int(n)
	case int16:
		return int(n)
	case int32:
		return int(n)
	case int64:
		return int(n)
	case uint8:
		return int(n)
	case uint16:
		return int(n)
	case uint32:
		return int(n)
	case uint64:
		return int(n)
	case float32:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}
