package arboreal

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/pkg/errors"
)

func (m *xgboostSchema) Predict(features *SparseVector) ([]float64, error) {
	internalResults, err := m.Learner.GradientBooster.Predict(features)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to predict with gradient booster")
	}
	switch m.Learner.GradientBooster.GetName() {
	case "gbtree", "gbtree_optimized":
		numClasses := max(m.Learner.LearnerModelParam.NumClass, 1)
		treesPerClass := len(internalResults) / numClasses
		perClassScore := make([]float64, numClasses)
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
		case "reg:logistic", "binary:logistic":
			return perClassScore, nil
		case "multi:softmax", "multi:softprob":
			fmt.Println("softmax", perClassScore)
			return Softmax(perClassScore), nil
		case "reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror":
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

func NewGBDTFromXGBoostJSON(filename string) (xgboostSchema, error) {
	var schema xgboostSchema
	jsonIO, err := ioutil.ReadFile(filename)
	if err != nil {
		return schema, errors.Wrapf(err, "failed to open %s", filename)
	}
	err = json.Unmarshal(jsonIO, &schema)
	if err != nil {
		return schema, errors.Wrapf(err, "couldn't unmarshal json")
	}
	return schema, nil
}
