package arboreal

import (
	"fmt"

	"github.com/pkg/errors"
)

type GradientBooster interface {
	GetName() string
	Predict(features SparseVector) ([]float32, error)
}

type GBLinear struct {
	Name  string `json:"name"`
	Model struct {
		Weights []float32 `json:"weights"`
	} `json:"model"`
}

func (m *GBLinear) Predict(features SparseVector) ([]float32, error) {
	var result []float32
	return result, errors.New("not yet implemented")
}

func (m *GBLinear) GetName() string {
	return m.Name
}

func (m *GBTree) Predict(features SparseVector) ([]float32, error) {
	result := make([]float32, len(m.Model.Trees))

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

func (t *tree) Predict(features SparseVector) (float32, error) {
	return 0.0, nil
}

func (m *XGBoostSchema) Predict(features SparseVector) ([]float32, error) {
	internalResults, err := m.Learner.GradientBooster.Predict(features)
	if err != nil {
		return nil, errors.Wrap(err, "failed to predict with gradient booster")
	}
	switch m.Learner.GradientBooster.GetName() {
	case "gbtree", "gbtree_optimized":
		numClasses := max(m.Learner.LearnerModelParam.NumClass, 1)
		treesPerClass := len(internalResults) / numClasses
		perClassScore := make([]float32, numClasses)
		for i := 0; i < numClasses; i++ {
			for j := 0; j < treesPerClass; j++ {
				var idx int
				// there has GOT to be a better way to do this
				switch m.Learner.Objective.Name {
				case "multi:softprob", "multi:softmax":
					idx = i % numClasses
				default:
					idx = i*treesPerClass + j
				}
				perClassScore[i] += internalResults[idx]
			}
			switch m.Learner.Objective.Name {
			case "reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror":
				// weirdly only applied to regression, not to binary classification
				perClassScore[i] += m.Learner.LearnerModelParam.BaseScore
			case "reg:logistic", "binary:logistic":
				perClassScore[i] = sigmoidSingle(perClassScore[i])
			}
		}
		// final post process
		switch m.Learner.Objective.Name {
		case "multi:softmax", "multi:softprob":
			return Softmax(perClassScore), nil
		case "reg:logistic", "binary:logistic", "reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror":
			return perClassScore, nil
		default:
			return nil, fmt.Errorf("unknown objective: %s", m.Learner.Objective)
		}
	case "gblinear":
		return internalResults, nil
	default:
		return nil, fmt.Errorf("unknown gradient booster: %s", m.Learner.GradientBooster.GetName())
	}
}
