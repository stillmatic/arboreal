package arboreal

import (
	math "github.com/chewxy/math32"
	"golang.org/x/exp/constraints"
)

func max[T constraints.Ordered](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func sigmoidSingle(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Softmax(ys []float32) []float32 {
	output := make([]float32, len(ys))
	var sum float32
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
