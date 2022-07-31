package arboreal_test

import (
	"testing"

	arboreal "github.com/stillmatic/arboreal"
	"github.com/stretchr/testify/assert"
)

var vec = &arboreal.SparseVector{
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

func TestXGBoostJson(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(t, err)
	// assert.Equal(t, res.Learner.GradientBooster.GetName(), "gbtree")
	// gb := res.Learner.GradientBooster.(*arboreal.GBTree)
	// assert.Equal(t, len(gb.Model.Trees), 100)
	// assert.Equal(t, len(gb.Model.Trees), arboreal.MustNotError(strconv.Atoi(gb.Model.GbtreeModelParam.NumTrees)))

	// t0 := gb.Model.Trees[0]
	// treeRes, err := t0.Predict(vec, gb.BaseScore)
	// assert.NoError(t, err)
	// t.Log(treeRes)

	// ensembleRes, err := gb.Predict(vec)
	// assert.NoError(t, err)
	// t.Log("ensemble values ", ensembleRes)

	// first 5 are [0.9999933 , 0.30979705, 0.9999808 , 0.69328964, 0.6143614 , 0.09738026]
	finalRes, err := res.Predict(vec)
	assert.NoError(t, err)
	assert.NotEmpty(t, finalRes)
	t.Log("final score ", finalRes)

	// nilVec := make(arboreal.SparseVector, 44)
	// nullRes, err := res.Predict(&nilVec)
	// assert.NoError(t, err)
	// assert.NotEmpty(t, nullRes)
	// t.Log(nullRes)

}

func TestToy(t *testing.T) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/toy.json")
	assert.NoError(t, err)
	sv0 := &arboreal.SparseVector{
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
	res0 := arboreal.MustNotError(res.Predict(sv0))
	assert.InDelta(t, 0.4343974019963509, res0[0], 0.01)
	sv1 := &arboreal.SparseVector{
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
	res1 := arboreal.MustNotError(res.Predict(sv1))
	assert.InDelta(t, 0.4694540577007751, res1[0], 0.01)

}

func BenchmarkXGBoost(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	nilVec := make(arboreal.SparseVector, 44)
	for i := 0; i < b.N; i++ {
		_, err := res.Predict(vec)
		assert.NoError(b, err)
		_, err = res.Predict(&nilVec)
		assert.NoError(b, err)
	}
}

func BenchmarkXGBoostTree(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	t0 := res.Learner.GradientBooster.(*arboreal.GBTModelOptimized).Trees[0]

	for i := 0; i < b.N; i++ {
		_, err := t0.Predict(vec)
		assert.NoError(b, err)
	}
}

func BenchmarkXGBoostTreeConcurrent(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)

	t0 := res.Learner.GradientBooster.(*arboreal.GBTModelOptimized).Trees[0]

	for i := 0; i < b.N; i++ {
		// guard <- struct{}{}
		go func(sv arboreal.SparseVector) {
			_, err := t0.Predict(vec)
			assert.NoError(b, err)
			// <-guard
		}(*vec)
	}
}

func BenchmarkXGBoostConcurrent(b *testing.B) {
	res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(b, err)
	// maxGoroutines := 32
	// guard := make(chan struct{}, maxGoroutines)
	for i := 0; i < b.N; i++ {
		// guard <- struct{}{}
		go func(sv arboreal.SparseVector) {
			_, err := res.Predict(vec)
			assert.NoError(b, err)
			// <-guard
		}(*vec)

	}
}

func BenchmarkLoadXGBoost(b *testing.B) {
	for i := 0; i < b.N; i++ {
		res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
		assert.NoError(b, err)
		_ = res
	}
}
