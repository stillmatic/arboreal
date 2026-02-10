package arboreal_test

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"testing"

	arboreal "github.com/stillmatic/arboreal"
	"github.com/stretchr/testify/assert"
)

// mustNotError is a test helper that panics on error.
func mustNotError[T any](input T, err error) T {
	if err != nil {
		panic(err)
	}
	return input
}

// Sparse vector used by existing tests.
var vec = arboreal.SparseVector{
	0:  2016.0,
	1:  1.0,
	2:  480.0,
	3:  33.0,
	4:  270.0,
	5:  4791.0,
	6:  90300.0,
	7:  144.06,
	8:  1420.0,
	9:  1450.0,
	10: 0.0,
	11: 1.0,
	12: 0.0,
	13: 0.0,
	14: 0.0,
	15: 0.0,
	16: 1.0,
	17: 0.0,
	18: 0.0,
	19: 0.0,
	20: 0.0,
	21: 1.0,
	22: 0.0,
	23: 1.0,
	24: 0.0,
	25: 1.0,
	26: 0.0,
	27: 0.0,
	28: 0.0,
	29: 0.0,
	30: 0.0,
	31: 0.0,
	32: 0.0,
	33: 0.0,
	34: 0.0,
	35: 1.0,
	36: 0.0,
	37: 0.0,
	38: 0.0,
	39: 1.0,
	40: 0.0,
	41: 0.0,
	42: 1.0,
	43: 0.0,
}

// Dense equivalent for benchmarking the dense path.
var vecDense = sparseToNaNDense(vec, 44)

// nilVecDense is all-NaN (all features missing), equivalent to empty SparseVector.
var nilVecDense = makeNaNSlice(44)

func makeNaNSlice(n int) []float32 {
	s := make([]float32, n)
	nan := float32(math.NaN())
	for i := range s {
		s[i] = nan
	}
	return s
}

func sparseToNaNDense(sv arboreal.SparseVector, n int) []float32 {
	out := makeNaNSlice(n)
	for k, v := range sv {
		if k < n {
			out[k] = v
		}
	}
	return out
}

func TestXGBoostJson(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(t, err)

	finalRes, err := res.Predict(vec)
	assert.NoError(t, err)
	assert.NotEmpty(t, finalRes)
	t.Log("final score ", finalRes)
}

func TestToy(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/toy.json")
	assert.NoError(t, err)
	sv0 := arboreal.SparseVector{
		0:  25,
		1:  2,
		2:  226802,
		3:  1,
		4:  7,
		5:  4,
		6:  6,
		7:  3,
		8:  2,
		9:  1,
		10: 0,
		11: 0,
		12: 40,
		13: 38,
	}
	res0 := mustNotError(res.Predict(sv0))
	t.Log((res0))
	assert.InDelta(t, 0.4343974019963509, res0[0], 0.01)
	sv1 := arboreal.SparseVector{
		0:  38,
		1:  2,
		2:  89814,
		3:  11,
		4:  9,
		5:  2,
		6:  4,
		7:  0,
		8:  4,
		9:  1,
		10: 0,
		11: 0,
		12: 50,
		13: 38,
	}
	res1 := mustNotError(res.Predict(sv1))
	t.Log((res1))
	assert.InDelta(t, 0.4694540577007751, res1[0], 0.01)
}

func TestRegression(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
	assert.NoError(t, err)
	score, err := res.Predict(vec)
	assert.NoError(t, err)
	assert.InDelta(t, 8.417279, score[0], 0.01)
}

func TestSoftprob(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/toysoftmax.json")
	assert.NoError(t, err)
	smvec0 := arboreal.SparseVector{
		0:  25,
		1:  2,
		2:  226802,
		3:  1,
		4:  7,
		5:  6,
		6:  3,
		7:  2,
		8:  1,
		9:  0,
		10: 0,
		11: 40,
		12: 38,
		13: 0,
	}
	score, err := res.Predict(smvec0)
	assert.NoError(t, err)
	assert.InDelta(t, 0.57720053, score[0], 0.01)
	t.Log(score)
	smvec1 := arboreal.SparseVector{
		0:  38,
		1:  2,
		2:  89814,
		3:  11,
		4:  9,
		5:  4,
		6:  0,
		7:  4,
		8:  1,
		9:  0,
		10: 0,
		11: 50,
		12: 38,
		13: 0,
	}
	score, err = res.Predict(smvec1)
	assert.NoError(t, err)
	assert.InDelta(t, 0.40584144, score[0], 0.01)
	t.Log(score)
}

// TestPredictDense verifies that dense prediction matches sparse prediction.
func TestPredictDense(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(t, err)

	sparseResult := mustNotError(res.Predict(vec))
	denseResult := mustNotError(res.PredictDense(vecDense))

	assert.Equal(t, len(sparseResult), len(denseResult))
	for i := range sparseResult {
		assert.InDelta(t, sparseResult[i], denseResult[i], 0.0001,
			"sparse vs dense mismatch at index %d", i)
	}
}

func TestPredictDenseRegression(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
	assert.NoError(t, err)

	sparseResult := mustNotError(res.Predict(vec))
	denseResult := mustNotError(res.PredictDense(vecDense))

	for i := range sparseResult {
		assert.InDelta(t, sparseResult[i], denseResult[i], 0.0001)
	}
}

// TestPredictDenseAoS verifies AoS matches SoA dense prediction.
func TestPredictDenseAoS(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(t, err)

	soaResult := mustNotError(res.PredictDense(vecDense))
	aosResult := mustNotError(res.PredictDenseAoS(vecDense))

	assert.Equal(t, len(soaResult), len(aosResult))
	for i := range soaResult {
		assert.InDelta(t, soaResult[i], aosResult[i], 0.0001,
			"SoA vs AoS mismatch at index %d", i)
	}
}

func TestPredictDenseAoSRegression(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
	assert.NoError(t, err)

	soaResult := mustNotError(res.PredictDense(vecDense))
	aosResult := mustNotError(res.PredictDenseAoS(vecDense))

	for i := range soaResult {
		assert.InDelta(t, soaResult[i], aosResult[i], 0.0001)
	}
}

// --- Benchmarks ---

// BenchmarkXGBoost benchmarks the full sparse prediction pipeline.
func BenchmarkXGBoost(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	nilVec := make(arboreal.SparseVector, 44)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := res.Predict(vec)
		assert.NoError(b, err)
		_, err = res.Predict(nilVec)
		assert.NoError(b, err)
	}
}

func BenchmarkXGBoostRegression(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
	assert.NoError(b, err)

	nilVec := make(arboreal.SparseVector, 44)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := res.Predict(vec)
		assert.NoError(b, err)
		_, err = res.Predict(nilVec)
		_ = err
	}
}

// BenchmarkXGBoostDense benchmarks the dense prediction pipeline (no map lookups).
func BenchmarkXGBoostDense(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := res.PredictDense(vecDense)
		assert.NoError(b, err)
		_, err = res.PredictDense(nilVecDense)
		assert.NoError(b, err)
	}
}

// BenchmarkXGBoostTree benchmarks a single tree's sparse prediction.
func BenchmarkXGBoostTree(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	booster := res.Learner.GradientBooster.(*arboreal.GBTModelOptimized)
	t0 := &booster.Trees[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = t0.Predict(vec)
	}
}

// BenchmarkXGBoostTreeDense benchmarks a single tree with dense input.
func BenchmarkXGBoostTreeDense(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	booster := res.Learner.GradientBooster.(*arboreal.GBTModelOptimized)
	t0 := &booster.Trees[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = t0.PredictDense(vecDense)
	}
}

// BenchmarkXGBoostDenseAoS benchmarks the AoS dense prediction pipeline.
func BenchmarkXGBoostDenseAoS(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := res.PredictDenseAoS(vecDense)
		assert.NoError(b, err)
		_, err = res.PredictDenseAoS(nilVecDense)
		assert.NoError(b, err)
	}
}

// BenchmarkXGBoostTreeDenseAoS benchmarks a single AoS tree with dense input.
func BenchmarkXGBoostTreeDenseAoS(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	t0 := &res.ModelAoS.Trees[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = t0.PredictDense(vecDense)
	}
}

func BenchmarkLoadXGBoost(b *testing.B) {
	for i := 0; i < b.N; i++ {
		res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
		assert.NoError(b, err)
		_ = res
	}
}

func readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	// Skip first row (header)
	row1, _ := bufio.NewReader(f).ReadSlice('\n')
	_, _ = f.Seek(int64(len(row1)), io.SeekStart)

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

// BenchmarkXGBEndToEnd benchmarks sparse prediction over CSV data.
func BenchmarkXGBEndToEnd(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)
	inputs := readCsvFile("testdata/mortgage_data.csv")
	l := len(inputs)
	floatInputs := make([][]float32, l)
	for i, input := range inputs {
		floatInputs[i] = make([]float32, len(input))
		for j, v := range input {
			floatInputs[i][j] = float32(mustNotError(strconv.ParseFloat(v, 32)))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vec := arboreal.SparseVectorFromArray(floatInputs[i%l])
		res.Predict(vec)
	}
}

// BenchmarkXGBEndToEndDense benchmarks dense prediction over CSV data (no SparseVector).
func BenchmarkXGBEndToEndDense(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)
	inputs := readCsvFile("testdata/mortgage_data.csv")
	l := len(inputs)
	floatInputs := make([][]float32, l)
	for i, input := range inputs {
		floatInputs[i] = make([]float32, len(input))
		for j, v := range input {
			floatInputs[i][j] = float32(mustNotError(strconv.ParseFloat(v, 32)))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res.PredictDense(floatInputs[i%l])
	}
}

// BenchmarkXGBEndToEndDenseAoS benchmarks AoS dense prediction over CSV data.
func BenchmarkXGBEndToEndDenseAoS(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)
	inputs := readCsvFile("testdata/mortgage_data.csv")
	l := len(inputs)
	floatInputs := make([][]float32, l)
	for i, input := range inputs {
		floatInputs[i] = make([]float32, len(input))
		for j, v := range input {
			floatInputs[i][j] = float32(mustNotError(strconv.ParseFloat(v, 32)))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res.PredictDenseAoS(floatInputs[i%l])
	}
}
