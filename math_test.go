package arboreal_test

import (
	"testing"

	math "github.com/chewxy/math32"
	"github.com/stillmatic/arboreal"
)

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func softmax(ys []float32) []float32 {
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

func softmaxAlt(vector []float32) []float32 {
	sum := float32(0.0)
	r := make([]float32, len(vector))
	for i, v := range vector {
		exp := math.Exp(v)
		r[i] = exp
		sum += exp
	}
	if sum != float32(0.0) {
		inverseSum := float32(1.0) / sum
		for i := range r {
			r[i] *= inverseSum
		}
	}
	return r
}

func BenchmarkSoftmax(b *testing.B) {
	vector := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b.Run("softmax", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			arboreal.Softmax(vector)
		}
	})
	b.Run("softmaxAlt", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			softmaxAlt(vector)
		}
	})
}
