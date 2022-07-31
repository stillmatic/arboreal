package arboreal

import (
	"math"

	"golang.org/x/exp/constraints"
)

func max[T constraints.Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func sigmoidSingle(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Softmax(ys []float64) []float64 {
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
