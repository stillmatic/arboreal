package arboreal

import (
	"fmt"
	"math"

	"github.com/pkg/errors"
	"golang.org/x/exp/constraints"
)

type SparseVector map[int]float64

func max[T constraints.Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func sigmoid(x []float64) []float64 {
	for i, v := range x {
		x[i] = sigmoidSingle(v)
	}
	return x
}

func sigmoidSingle(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func softmax(ys []float64) []float64 {
	output := make([]float64, len(ys))
	var sum float64
	for i, y := range ys {
		exp := math.Exp(y)
		sum += exp
		output[i] = exp
	}
	if sum != 0.0 {
		for i := range output {
			output[i] /= sum
		}
	}
	return output
}

func (m *xgboostSchema) Predict(features *SparseVector) ([]float64, error) {
	internalResults, err := m.Learner.GradientBooster.Predict(features)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to predict with gradient booster")
	}
	switch m.Learner.GradientBooster.GetName() {
	case "gbtree", "gbtree_optimized":
		numClasses := max(m.Learner.LearnerModelParam.NumClass, 1)
		treesPerClass := len(internalResults) / numClasses
		perClassScore := make([]float64, numClasses)
		for i := 0; i < numClasses; i++ {
			for j := 0; j < treesPerClass; j++ {
				perClassScore[i] += internalResults[(i*treesPerClass)+j]
			}
		}
		// TODO: handle objective
		return sigmoid(perClassScore), nil
	case "gblinear":
		return internalResults, nil
	default:
		return nil, fmt.Errorf("unknown gradient booster: %s", m.Learner.GradientBooster.GetName())
	}
}
