package arboreal

import math "github.com/chewxy/math32"

// Softmax applies the softmax function to a slice of scores.
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
