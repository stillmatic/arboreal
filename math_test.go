package arboreal_test

import (
	"math"
	"testing"

	arboreal "github.com/stillmatic/arboreal"
)

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
			arboreal.Softmax(vector)
		}
	})
	b.Run("softmaxAlt", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			softmaxAlt(vector)
		}
	})
}
