package arboreal

import math "github.com/chewxy/math32"

// TreeOptimized uses a struct-of-arrays layout for cache-friendly tree traversal.
// Each field is a contiguous slice indexed by node ID. Numerical-only trees use
// the fast path (predictNumerical); trees with any categorical splits use predictMixed.
type TreeOptimized struct {
	LeftChild      []int32
	RightChild     []int32
	SplitIndex     []int32
	SplitCondition []float32
	DefaultLeft    []bool
	SplitType      []uint8 // 0=numerical, 1=categorical
	Category       []int32
	HasCategorical bool
}

// GBTModelOptimized holds the ensemble of optimized trees.
type GBTModelOptimized struct {
	Trees []TreeOptimized
}

func (m *GBTModelOptimized) GetName() string {
	return "gbtree_optimized"
}

func (m *GBTModelOptimized) Predict(features SparseVector) ([]float32, error) {
	result := make([]float32, len(m.Trees))
	for idx := range m.Trees {
		result[idx] = m.Trees[idx].Predict(features)
	}
	return result, nil
}

// PredictDense predicts using a dense feature vector. Missing values are
// represented as NaN. This is the fast path — no map lookups.
func (m *GBTModelOptimized) PredictDense(features []float32) []float32 {
	result := make([]float32, len(m.Trees))
	for idx := range m.Trees {
		result[idx] = m.Trees[idx].PredictDense(features)
	}
	return result
}

// Predict dispatches to the appropriate traversal method using SparseVector input.
func (t *TreeOptimized) Predict(features SparseVector) float32 {
	if t.HasCategorical {
		return t.predictMixedSparse(features)
	}
	return t.predictNumericalSparse(features)
}

// PredictDense dispatches using a dense []float32 slice. Missing = NaN.
func (t *TreeOptimized) PredictDense(features []float32) float32 {
	if t.HasCategorical {
		return t.predictMixedDense(features)
	}
	return t.predictNumericalDense(features)
}

// predictNumericalDense is the hot path for numerical-only trees with dense input.
func (t *TreeOptimized) predictNumericalDense(features []float32) float32 {
	idx := 0
	nFeatures := int32(len(features))
	for {
		leftChild := t.LeftChild[idx]
		if leftChild < 0 { // leaf
			return t.SplitCondition[idx]
		}

		splitCol := t.SplitIndex[idx]

		if splitCol >= nFeatures {
			// out of bounds = missing
			if t.DefaultLeft[idx] {
				idx = int(leftChild)
			} else {
				idx = int(leftChild) + 1
			}
			continue
		}

		fval := features[splitCol]

		// NaN = missing
		if fval != fval {
			if t.DefaultLeft[idx] {
				idx = int(leftChild)
			} else {
				idx = int(leftChild) + 1
			}
			continue
		}

		// xgboost uses <
		if fval < t.SplitCondition[idx] {
			idx = int(leftChild)
		} else {
			idx = int(leftChild) + 1
		}
	}
}

// predictNumericalSparse handles numerical-only trees with sparse map input.
func (t *TreeOptimized) predictNumericalSparse(features SparseVector) float32 {
	idx := 0
	for {
		leftChild := t.LeftChild[idx]
		if leftChild < 0 { // leaf
			return t.SplitCondition[idx]
		}

		splitCol := t.SplitIndex[idx]
		fval, ok := features[int(splitCol)]

		if !ok {
			if t.DefaultLeft[idx] {
				idx = int(leftChild)
			} else {
				idx = int(leftChild) + 1
			}
			continue
		}

		if fval < t.SplitCondition[idx] {
			idx = int(leftChild)
		} else {
			idx = int(leftChild) + 1
		}
	}
}

// predictMixedDense handles trees with categorical splits and dense input.
func (t *TreeOptimized) predictMixedDense(features []float32) float32 {
	idx := 0
	nFeatures := int32(len(features))
	for {
		leftChild := t.LeftChild[idx]
		if leftChild < 0 { // leaf
			return t.SplitCondition[idx]
		}

		splitCol := t.SplitIndex[idx]

		if splitCol >= nFeatures {
			if t.DefaultLeft[idx] {
				idx = int(leftChild)
			} else {
				idx = int(t.RightChild[idx])
			}
			continue
		}

		fval := features[splitCol]

		if fval != fval { // NaN = missing
			if t.DefaultLeft[idx] {
				idx = int(leftChild)
			} else {
				idx = int(t.RightChild[idx])
			}
			continue
		}

		if t.SplitType[idx] == 1 {
			if int32(fval) == t.Category[idx] {
				idx = int(t.RightChild[idx])
			} else {
				idx = int(leftChild)
			}
		} else {
			if fval < t.SplitCondition[idx] {
				idx = int(leftChild)
			} else {
				idx = int(t.RightChild[idx])
			}
		}
	}
}

// predictMixedSparse handles trees with categorical splits and sparse map input.
func (t *TreeOptimized) predictMixedSparse(features SparseVector) float32 {
	idx := 0
	for {
		leftChild := t.LeftChild[idx]
		if leftChild < 0 { // leaf
			return t.SplitCondition[idx]
		}

		splitCol := t.SplitIndex[idx]
		fval, ok := features[int(splitCol)]

		if !ok {
			if t.DefaultLeft[idx] {
				idx = int(leftChild)
			} else {
				idx = int(t.RightChild[idx])
			}
			continue
		}

		if t.SplitType[idx] == 1 {
			if int32(fval) == t.Category[idx] {
				idx = int(t.RightChild[idx])
			} else {
				idx = int(leftChild)
			}
		} else {
			if fval < t.SplitCondition[idx] {
				idx = int(leftChild)
			} else {
				idx = int(t.RightChild[idx])
			}
		}
	}
}

func sigmoidSingle(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}
