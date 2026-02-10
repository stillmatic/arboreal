package arboreal_test

import (
	"github.com/stillmatic/arboreal"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestXGBoostSchema_Predict(t *testing.T) {
	schema, err := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
	assert.NoError(t, err)
	inpArr := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	inpVec := arboreal.SparseVectorFromArray(inpArr)
	res, err := schema.Predict(inpVec)
	assert.NoError(t, err)
	assert.InDelta(t, 6.245257, res[0], 0.000001)
}

func TestPredictDenseExample(t *testing.T) {
	schema, err := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
	assert.NoError(t, err)

	nilVec := make(arboreal.SparseVector, 44)
	_, err = schema.Predict(nilVec)
	assert.NoError(t, err)
}
