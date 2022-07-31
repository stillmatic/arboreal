package arboreal

import (
	"fmt"
)

type GradientBooster interface {
	GetName() string
	Predict(features *SparseVector) ([]float64, error)
}

type GBLinear struct {
	Name  string `json:"name"`
	Model struct {
		Weights []float64 `json:"weights"`
	} `json:"model"`
}

func (m *GBLinear) Predict(features *SparseVector) ([]float64, error) {
	var result []float64
	// for idx, tree := range m.Model.Trees {
	// 	res, err := tree.Predict(features)
	// 	if err != nil {
	// 		return nil, err
	// 	}
	// 	result[idx] = res
	// }
	return result, nil
}

func (m *GBLinear) GetName() string {
	return m.Name
}

func (m *GBTree) Predict(features *SparseVector) ([]float64, error) {
	result := make([]float64, len(m.Model.Trees))

	for idx, tree := range m.Model.Trees {
		res, err := tree.Predict(features)
		if err != nil {
			return nil, err
		}
		result[idx] = res
	}
	return result, nil
}

func (m *GBTree) GetName() string {
	return m.Name
}

func (t *tree) Predict(features *SparseVector) (float64, error) {
	idx := 0

	for {
		if t.Leaves[idx] {
			// splitConditions[idx] is return value for a leaf node
			return t.SplitConditions[idx], nil
		}

		leftChild := t.LeftChildren[idx]
		// this is a fairly insane optimization
		// but it's somehow a 10% speedup on M1 mac and safe with legal XGBoost
		// https://github.com/dmlc/xgboost/pull/6127
		rightChild := leftChild + 1

		splitCol := t.SplitIndices[idx]
		splitVal := t.SplitConditions[idx]
		fval, ok := (*features)[splitCol]

		// missing value behavior is determined by default left
		if !ok {
			if t.DefaultLeft[idx] == 1 {
				idx = leftChild
			} else {
				idx = rightChild
			}
			continue
		}
		switch t.SplitType[idx] {
		case 0:
			// xgboost uses <, lightgbm uses <=
			if fval < splitVal {
				idx = leftChild
			} else {
				idx = rightChild
			}
		// handle categorical case
		case 1:
			// todo: doublecheck this
			if int(fval) == t.Categories[idx] {
				idx = rightChild
			} else {
				idx = leftChild
			}
		default:
			return 0, fmt.Errorf("unknown split type %d", t.SplitType[idx])
		}
	}
}
