package arboreal_test

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	arboreal "github.com/stillmatic/arboreal"
)

// Cache-scaling benchmarks: SoA vs AoS at different ensemble sizes.
//
// Actual mortgage model trees average 23 nodes (not 127), so the real
// per-tree footprints are:
//   - SoA: ~391 bytes (full), ~299 bytes (hot path only)
//   - AoS: ~552 bytes (24 bytes × 23 nodes)
//
// Cache boundaries (48 KB L1, 2 MB L2):
//   - SoA fits L1: ≤125 trees    AoS fits L1: ≤89 trees
//   - SoA fits L2: ≤5363 trees   AoS fits L2: ≤3799 trees
//
// Key transitions to test:
//   - 5–50 trees:     both fit L1 comfortably
//   - 100 trees:      SoA fits L1, AoS barely spills (the existing benchmark!)
//   - 200–500 trees:  neither fits L1, both fit L2
//   - 1000–2000:      both fit L2 but with increasing pressure
//   - 4000–5000:      SoA fits L2, AoS approaches L2 limit
//   - 8000–10000:     both exceed L2, fall back to L3

var treeCounts = []int{1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 4000, 8000}

// BenchmarkCacheScalingSoA benchmarks SoA ensemble prediction at various tree counts.
func BenchmarkCacheScalingSoA(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	if err != nil {
		b.Fatal(err)
	}
	booster := res.Learner.GradientBooster.(*arboreal.GBTModelOptimized)

	maxCount := treeCounts[len(treeCounts)-1]
	soaTrees := makeSoATrees(booster.Trees, maxCount)

	features := sparseToNaNDense(vec, 44)

	for _, n := range treeCounts {
		b.Run(fmt.Sprintf("trees=%d", n), func(b *testing.B) {
			trees := soaTrees[:n]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var acc float32
				for j := range trees {
					acc += trees[j].PredictDense(features)
				}
				_ = acc
			}
		})
	}
}

// BenchmarkCacheScalingAoS benchmarks AoS ensemble prediction at various tree counts.
func BenchmarkCacheScalingAoS(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	if err != nil {
		b.Fatal(err)
	}

	maxCount := treeCounts[len(treeCounts)-1]
	aosTrees := makeAoSTrees(res.ModelAoS.Trees, maxCount)

	features := sparseToNaNDense(vec, 44)

	for _, n := range treeCounts {
		b.Run(fmt.Sprintf("trees=%d", n), func(b *testing.B) {
			trees := aosTrees[:n]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var acc float32
				for j := range trees {
					acc += trees[j].PredictDense(features)
				}
				_ = acc
			}
		})
	}
}

// makeSoATrees creates `want` SoA trees by deep-copying from src.
// Each tree gets its own memory allocations — no aliased backing arrays.
func makeSoATrees(src []arboreal.TreeOptimized, want int) []arboreal.TreeOptimized {
	out := make([]arboreal.TreeOptimized, want)
	for i := 0; i < want; i++ {
		s := &src[i%len(src)]
		n := len(s.LeftChild)
		t := arboreal.TreeOptimized{
			LeftChild:      make([]int32, n),
			RightChild:     make([]int32, n),
			SplitIndex:     make([]int32, n),
			SplitCondition: make([]float32, n),
			DefaultLeft:    make([]bool, n),
			HasCategorical: s.HasCategorical,
		}
		copy(t.LeftChild, s.LeftChild)
		copy(t.RightChild, s.RightChild)
		copy(t.SplitIndex, s.SplitIndex)
		copy(t.SplitCondition, s.SplitCondition)
		copy(t.DefaultLeft, s.DefaultLeft)
		if s.HasCategorical {
			t.SplitType = make([]uint8, n)
			t.Category = make([]int32, n)
			copy(t.SplitType, s.SplitType)
			copy(t.Category, s.Category)
		}
		out[i] = t
	}
	return out
}

// makeAoSTrees creates `want` AoS trees by deep-copying from src.
func makeAoSTrees(src []arboreal.TreeAoS, want int) []arboreal.TreeAoS {
	out := make([]arboreal.TreeAoS, want)
	for i := 0; i < want; i++ {
		s := &src[i%len(src)]
		t := arboreal.TreeAoS{
			Nodes:          make([]arboreal.NodeAoS, len(s.Nodes)),
			HasCategorical: s.HasCategorical,
		}
		copy(t.Nodes, s.Nodes)
		out[i] = t
	}
	return out
}

// TestTreeMemoryFootprint prints actual memory footprints for analysis.
func TestTreeMemoryFootprint(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	if err != nil {
		t.Fatal(err)
	}
	booster := res.Learner.GradientBooster.(*arboreal.GBTModelOptimized)

	var totalNodes int
	var minNodes, maxNodes int
	minNodes = math.MaxInt
	for i := range booster.Trees {
		n := len(booster.Trees[i].LeftChild)
		totalNodes += n
		if n < minNodes {
			minNodes = n
		}
		if n > maxNodes {
			maxNodes = n
		}
	}
	nTrees := len(booster.Trees)
	avgNodes := totalNodes / nTrees

	t.Logf("Model: %d trees, %d total nodes", nTrees, totalNodes)
	t.Logf("Nodes per tree: min=%d, max=%d, avg=%d", minNodes, maxNodes, avgNodes)

	soaHotPerTree := avgNodes*4*3 + avgNodes
	soaFullPerTree := avgNodes*4*4 + avgNodes
	aosPerTree := avgNodes * int(unsafe.Sizeof(arboreal.NodeAoS{}))

	t.Logf("")
	t.Logf("=== Per-tree memory footprint (avg %d nodes) ===", avgNodes)
	t.Logf("SoA hot (3 arrays + bool):  %d bytes (%.1f KB)", soaHotPerTree, float64(soaHotPerTree)/1024)
	t.Logf("SoA full (4 arrays + bool): %d bytes (%.1f KB)", soaFullPerTree, float64(soaFullPerTree)/1024)
	t.Logf("AoS (24 bytes/node):        %d bytes (%.1f KB)", aosPerTree, float64(aosPerTree)/1024)
	t.Logf("NodeAoS struct size:        %d bytes", unsafe.Sizeof(arboreal.NodeAoS{}))

	t.Logf("")
	t.Logf("=== Ensemble memory footprint ===")
	t.Logf("%-8s  %-12s  %-12s  %-12s  %-10s", "Trees", "SoA hot", "SoA full", "AoS", "Ratio")
	for _, n := range treeCounts {
		soaH := n * soaHotPerTree
		soaF := n * soaFullPerTree
		aos := n * aosPerTree
		t.Logf("%-8d  %-12s  %-12s  %-12s  %.2fx",
			n,
			fmtBytes(soaH), fmtBytes(soaF), fmtBytes(aos),
			float64(aos)/float64(soaF))
	}

	t.Logf("")
	t.Logf("=== Cache thresholds (typical Xeon) ===")
	t.Logf("L1 data cache:  48 KB")
	t.Logf("L2 cache:       2 MB (per core)")
	t.Logf("L3 cache:       ~2-4 MB per core (shared)")
	t.Logf("")
	t.Logf("SoA full fits L1 up to:  %d trees", 48*1024/soaFullPerTree)
	t.Logf("AoS fits L1 up to:       %d trees", 48*1024/aosPerTree)
	t.Logf("SoA full fits L2 up to:  %d trees", 2*1024*1024/soaFullPerTree)
	t.Logf("AoS fits L2 up to:       %d trees", 2*1024*1024/aosPerTree)
}

func fmtBytes(b int) string {
	switch {
	case b >= 1024*1024:
		return fmt.Sprintf("%.1f MB", float64(b)/(1024*1024))
	case b >= 1024:
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
