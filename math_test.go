package arboreal_test

import (
	"math"
	"testing"
)

func sigmoid(x float64) float64 {
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

func softmaxAlt(vector []float64) []float64 {
	sum := 0.0
	r := make([]float64, len(vector))
	for i, v := range vector {
		exp := math.Exp(v)
		r[i] = exp
		sum += exp
	}
	if sum != 0.0 {
		inverseSum := 1.0 / sum
		for i := range r {
			r[i] *= inverseSum
		}
	}
	return r
}

func BenchmarkSoftmax(b *testing.B) {
	vector := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b.Run("softmax", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			softmax(vector)
		}
	})
	b.Run("softmaxAlt", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			softmaxAlt(vector)
		}
	})
}
