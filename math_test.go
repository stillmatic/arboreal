package arboreal_test

import (
	"github.com/viterin/vek/vek32"
	"testing"

	math "github.com/chewxy/math32"
	"github.com/stillmatic/arboreal"
	"github.com/stretchr/testify/require"
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

func softmaxSimd(vector []float32) []float32 {
	r := make([]float32, len(vector))
	vek32.Exp_Into(r, vector)
	sum := vek32.Sum(r)
	if sum != float32(0.0) {
		inverseSum := float32(1.0) / sum
		vek32.MulNumber_Inplace(r, inverseSum)
	}
	return r
}

func softmaxSimdInplace(vector []float32) []float32 {
	vek32.Exp_Inplace(vector)
	sum := vek32.Sum(vector)
	if sum != float32(0.0) {
		inverseSum := float32(1.0) / sum
		vek32.MulNumber_Inplace(vector, inverseSum)
	}
	return vector
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

type sigmoidTable struct {
	expTable     []float32
	expTableSize int
	maxExp       float32
	cache        float32
}

func newSigmoidTable() *sigmoidTable {
	s := new(sigmoidTable)
	s.expTableSize = 1024
	s.maxExp = 6.0
	s.cache = float32(s.expTableSize) / s.maxExp / 2.0
	s.expTable = make([]float32, s.expTableSize)
	for i := 0; i < s.expTableSize; i++ {
		expval := math.Exp((float32(i)/float32(s.expTableSize)*2. - 1.) * s.maxExp)
		s.expTable[i] = expval / (expval + 1.)
	}
	return s
}

// sigmoid returns: f(x) = (x + max_exp) * (exp_table_size / max_exp / 2)
// If you set x to over |max_exp|, it raises index out of range error.
func (s *sigmoidTable) sigmoid(x float32) float32 {
	if x < -s.maxExp {
		return 0.0
	} else if x > s.maxExp {
		return 1.0
	}
	return s.expTable[int((x+s.maxExp)*s.cache)]
}

// inplace SIMD saves an alloc and a couple nanoseconds but it's not a big difference.
// BenchmarkSoftmax/softmax-10                   	10690101	       107.1 ns/op	      48 B/op	       1 allocs/op
// BenchmarkSoftmax/softmaxAlt-10                	11185256	       107.5 ns/op	      48 B/op	       1 allocs/op
// BenchmarkSoftmax/SIMD-10                      	10492280	       113.2 ns/op	      48 B/op	       1 allocs/op
// BenchmarkSoftmax/SIMD_Inplace-10              	11581250	       103.2 ns/op	       0 B/op	       0 allocs/op
func BenchmarkSoftmax(b *testing.B) {
	vector := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b.ResetTimer()
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
	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			softmaxSimd(vector)
		}
	})
	b.Run("SIMD_Inplace", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			softmaxSimdInplace(vector)
		}
	})
}

func BenchmarkSigmoid(b *testing.B) {
	vector := []float32{-7.0, -1.0, -0.5, 0.0, 0.5, 1.0, 7.0}
	b.Run("sigmoid", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = sigmoid(vector[i%len(vector)])
			// require.NotEqual(b, res, 0)
		}
	})
	b.Run("lookup table", func(b *testing.B) {
		st := newSigmoidTable()
		for i := 0; i < b.N; i++ {
			_ = st.sigmoid(vector[i%len(vector)])
		}
	})
}

func TestSigmoid(t *testing.T) {
	vector := []float32{-7.0, -1.0, -0.5, 0.0, 0.5, 1.0, 7.0}
	st := newSigmoidTable()
	for _, v := range vector {
		require.InDelta(t, st.sigmoid(v), sigmoid(v), 0.002)
	}
}
