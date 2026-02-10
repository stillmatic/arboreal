package arboreal

// Compact AoS (array-of-structs) tree representation.
//
// This preserves the original cache locality design: all fields for a single
// node are adjacent in memory, so visiting t.Nodes[idx] loads everything into
// one or two cache lines. Combined with int32 fields (24 bytes/node vs 80 bytes
// in the original), ~2.5 nodes fit per 64-byte L1 cache line.
//
// Original optimizations preserved:
//   - AoS layout for per-node cache locality
//   - IsLeaf precomputed at load time
//   - rightChild = leftChild + 1 for numerical trees (skips RightChild field)
//   - Separate predictNumerical/predictCategorical (function outlining)
//   - Single node dereference per loop iteration

// NodeAoS is a compact 24-byte node struct. Fields are ordered largest-first
// to minimize padding. Hot fields (SplitCondition, LeftChild, SplitIndex,
// DefaultLeft, IsLeaf) are all within the first 16 bytes.
type NodeAoS struct {
	SplitCondition float32 // split threshold or leaf value
	LeftChild      int32   // left child index (-1 for leaf)
	SplitIndex     int32   // feature index for this split
	RightChild     int32   // right child index (categorical only)
	Category       int32   // categorical split value
	DefaultLeft    bool    // missing value → go left
	IsLeaf         bool    // precomputed: leftChild == -1 && rightChild == -1
	SplitType      uint8   // 0=numerical, 1=categorical
	// 1 byte padding to 24 bytes (Go aligns to largest field = 4 bytes)
}

type TreeAoS struct {
	Nodes          []NodeAoS // contiguous value slice, not pointers
	HasCategorical bool
}

type GBTModelAoS struct {
	Trees []TreeAoS
}

func (m *GBTModelAoS) GetName() string {
	return "gbtree_aos"
}

func (m *GBTModelAoS) Predict(features SparseVector) ([]float32, error) {
	result := make([]float32, len(m.Trees))
	for idx := range m.Trees {
		result[idx] = m.Trees[idx].Predict(features)
	}
	return result, nil
}

// Predict dispatches to numerical or categorical path.
func (t *TreeAoS) Predict(features SparseVector) float32 {
	if t.HasCategorical {
		return t.predictMixedSparse(features)
	}
	return t.predictNumericalSparse(features)
}

// PredictDense dispatches using dense []float32. Missing = NaN.
func (t *TreeAoS) PredictDense(features []float32) float32 {
	if t.HasCategorical {
		return t.predictMixedDense(features)
	}
	return t.predictNumericalDense(features)
}

// predictNumericalDense is the hot path: AoS layout, dense input, rightChild = leftChild+1.
func (t *TreeAoS) predictNumericalDense(features []float32) float32 {
	idx := 0
	nFeatures := int32(len(features))
	for {
		node := t.Nodes[idx] // single load, 24 bytes, fits cache line
		if node.IsLeaf {
			return node.SplitCondition
		}

		leftChild := node.LeftChild
		splitCol := node.SplitIndex

		if splitCol >= nFeatures {
			if node.DefaultLeft {
				idx = int(leftChild)
			} else {
				idx = int(leftChild) + 1
			}
			continue
		}

		fval := features[splitCol]

		if fval != fval { // NaN = missing
			if node.DefaultLeft {
				idx = int(leftChild)
			} else {
				idx = int(leftChild) + 1
			}
			continue
		}

		if fval < node.SplitCondition {
			idx = int(leftChild)
		} else {
			idx = int(leftChild) + 1
		}
	}
}

// predictNumericalSparse: AoS layout, sparse map input.
func (t *TreeAoS) predictNumericalSparse(features SparseVector) float32 {
	idx := 0
	for {
		node := t.Nodes[idx]
		if node.IsLeaf {
			return node.SplitCondition
		}

		leftChild := node.LeftChild
		fval, ok := features[int(node.SplitIndex)]

		if !ok {
			if node.DefaultLeft {
				idx = int(leftChild)
			} else {
				idx = int(leftChild) + 1
			}
			continue
		}

		if fval < node.SplitCondition {
			idx = int(leftChild)
		} else {
			idx = int(leftChild) + 1
		}
	}
}

// predictMixedDense handles trees with categorical splits, dense input.
func (t *TreeAoS) predictMixedDense(features []float32) float32 {
	idx := 0
	nFeatures := int32(len(features))
	for {
		node := t.Nodes[idx]
		if node.IsLeaf {
			return node.SplitCondition
		}

		leftChild := node.LeftChild
		splitCol := node.SplitIndex

		if splitCol >= nFeatures {
			if node.DefaultLeft {
				idx = int(leftChild)
			} else {
				idx = int(node.RightChild)
			}
			continue
		}

		fval := features[splitCol]

		if fval != fval {
			if node.DefaultLeft {
				idx = int(leftChild)
			} else {
				idx = int(node.RightChild)
			}
			continue
		}

		if node.SplitType == 1 {
			if int32(fval) == node.Category {
				idx = int(node.RightChild)
			} else {
				idx = int(leftChild)
			}
		} else {
			if fval < node.SplitCondition {
				idx = int(leftChild)
			} else {
				idx = int(node.RightChild)
			}
		}
	}
}

// predictMixedSparse handles trees with categorical splits, sparse map input.
func (t *TreeAoS) predictMixedSparse(features SparseVector) float32 {
	idx := 0
	for {
		node := t.Nodes[idx]
		if node.IsLeaf {
			return node.SplitCondition
		}

		leftChild := node.LeftChild
		fval, ok := features[int(node.SplitIndex)]

		if !ok {
			if node.DefaultLeft {
				idx = int(leftChild)
			} else {
				idx = int(node.RightChild)
			}
			continue
		}

		if node.SplitType == 1 {
			if int32(fval) == node.Category {
				idx = int(node.RightChild)
			} else {
				idx = int(leftChild)
			}
		} else {
			if fval < node.SplitCondition {
				idx = int(leftChild)
			} else {
				idx = int(node.RightChild)
			}
		}
	}
}

// OptimizedGBTModelAoS converts a parsed model to the compact AoS representation.
func OptimizedGBTModelAoS(in *model) *GBTModelAoS {
	out := &GBTModelAoS{
		Trees: make([]TreeAoS, len(in.Trees)),
	}
	for idx, tree := range in.Trees {
		out.Trees[idx] = OptimizedTreeAoS(tree)
	}
	return out
}

// OptimizedTreeAoS converts a single parsed tree to compact AoS layout.
func OptimizedTreeAoS(in *tree) TreeAoS {
	n := len(in.LeftChildren)
	nodes := make([]NodeAoS, n)
	hasCategorical := false

	for i := 0; i < n; i++ {
		nodes[i] = NodeAoS{
			SplitCondition: in.SplitConditions[i],
			LeftChild:      int32(in.LeftChildren[i]),
			SplitIndex:     int32(in.SplitIndices[i]),
			RightChild:     int32(in.RightChildren[i]),
			DefaultLeft:    in.DefaultLeft[i] == 1,
			IsLeaf:         in.LeftChildren[i] == -1 && in.RightChildren[i] == -1,
		}
		if len(in.SplitType) > i {
			nodes[i].SplitType = uint8(in.SplitType[i])
			if in.SplitType[i] == 1 {
				hasCategorical = true
			}
		}
		if len(in.Categories) > i {
			nodes[i].Category = int32(in.Categories[i])
		}
	}

	return TreeAoS{
		Nodes:          nodes,
		HasCategorical: hasCategorical,
	}
}
