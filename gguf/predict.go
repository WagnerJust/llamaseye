package gguf

// PredictResult holds derived NGL and context ceiling predictions.
type PredictResult struct {
	MaxNGLPred    int
	BestCtxVRAM   int
	BestCtxRAM    int
	StartNGL      int // one step below max (Phase 1 smart start)
	StartCtx      int // one step below predicted ceiling
}

// CTKBytesPerElem maps ctk type strings to bytes per element.
var CTKBytesPerElem = map[string]float64{
	"f16":    2.0,
	"q8_0":   1.0,
	"q4_0":   0.5,
	"turbo4": 0.5,
	"turbo3": 0.375,
	"turbo2": 0.25,
}

// CTXList is the standard sequence of context sizes.
var CTXList = []int{128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}

// Predict uses model metadata and hardware info to derive optimal start points.
func Predict(meta *Metadata, vramGiB, ramGiB int) PredictResult {
	layers := meta.NumLayers
	if layers == 0 {
		return PredictResult{}
	}
	fileGiB := meta.FileGiB
	vram := float64(vramGiB)
	ram := float64(ramGiB)

	// NGL prediction
	embedOverhead := fileGiB / float64(layers)
	perLayer := (fileGiB - embedOverhead) / float64(layers)
	cap := vram * 0.90
	ngl := int((cap - embedOverhead) / perLayer)
	if ngl < 0 {
		ngl = 0
	}
	if ngl > layers {
		ngl = layers
	}

	// VRAM headroom after weights
	vramWeights := embedOverhead + perLayer*float64(ngl)
	vramHead := vram*0.90 - vramWeights
	if vramHead < 0 {
		vramHead = 0
	}

	// RAM headroom (85% usable, 65% of file assumed hot)
	ramHead := ram*0.85 - fileGiB*0.65
	if ramHead < 0 {
		ramHead = 0
	}

	// Context ceiling prediction per ctk type
	var bestCtxVRAM, bestCtxRAM int
	for _, ctx := range CTXList {
		for _, bpe := range CTKBytesPerElem {
			var kvGPU, kvTotal float64
			if meta.HasHybrid && meta.SlidingWin > 0 {
				swaCtx := float64(ctx)
				if float64(meta.SlidingWin) < swaCtx {
					swaCtx = float64(meta.SlidingWin)
				}
				kvTotal = (float64(meta.NGlobalLayers)*float64(meta.GlobalKVHeads)*float64(meta.GlobalHeadDim)*float64(ctx)*2*bpe +
					float64(meta.NSWALayers)*float64(meta.SWAKVHeads)*float64(meta.SWAHeadDim)*swaCtx*2*bpe) / (1 << 30)
				gpuFrac := 0.0
				if ngl > 0 {
					gpuFrac = float64(ngl) / float64(meta.NGlobalLayers+meta.NSWALayers)
				}
				kvGPU = kvTotal * gpuFrac
			} else {
				kvTotal = float64(layers) * float64(meta.KVHeadsMax) * float64(meta.KeyLen) * float64(ctx) * 2 * bpe / (1 << 30)
				gpuFrac := 0.0
				if ngl > 0 {
					gpuFrac = float64(ngl) / float64(layers)
				}
				kvGPU = kvTotal * gpuFrac
			}
			if kvGPU <= vramHead && ctx > bestCtxVRAM {
				bestCtxVRAM = ctx
			}
			if kvTotal <= ramHead && ctx > bestCtxRAM {
				bestCtxRAM = ctx
			}
		}
	}

	// Pick overall best ctx
	bestCtx := bestCtxVRAM
	if bestCtxRAM > bestCtxVRAM {
		bestCtx = bestCtxRAM
	}
	if bestCtx == 0 {
		bestCtx = 512
	}

	// Start ctx: one step below predicted ceiling
	startCtx := bestCtx
	for i, v := range CTXList {
		if v == bestCtx && i > 0 {
			startCtx = CTXList[i-1]
			break
		}
	}

	// Start NGL: 2 steps below max
	startNGL := ngl - 2*4 // default step is 4
	if startNGL < 0 {
		startNGL = 0
	}

	return PredictResult{
		MaxNGLPred:  ngl,
		BestCtxVRAM: bestCtxVRAM,
		BestCtxRAM:  bestCtxRAM,
		StartNGL:    startNGL,
		StartCtx:    startCtx,
	}
}
