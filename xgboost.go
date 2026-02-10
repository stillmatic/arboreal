// Package arboreal is a pure Go package for XGBoost model inference.
package arboreal

import "fmt"

// GradientBooster is the interface for tree ensemble and linear boosters.
type GradientBooster interface {
	GetName() string
	Predict(features SparseVector) ([]float32, error)
}

// GBLinear represents a linear booster model.
type GBLinear struct {
	Name  string `json:"name"`
	Model struct {
		Weights []float32 `json:"weights"`
	} `json:"model"`
}

func (m *GBLinear) Predict(features SparseVector) ([]float32, error) {
	return nil, fmt.Errorf("gblinear predict: not yet implemented")
}

func (m *GBLinear) GetName() string {
	return m.Name
}

// perScoreFn is applied to each class's accumulated score.
type perScoreFn func(float32) float32

// postProcessFn is applied to the entire score vector after per-score transforms.
type postProcessFn func([]float32) []float32

// resolveObjective converts an objective name into per-score and post-process
// functions, eliminating switch statements from the hot prediction path.
func resolveObjective(obj Objective, baseScore float32) (perScoreFn, postProcessFn, bool, error) {
	identity := func(x float32) float32 { return x }
	passthrough := func(xs []float32) []float32 { return xs }

	switch obj.Name {
	case "reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror":
		addBase := func(x float32) float32 { return x + baseScore }
		return addBase, passthrough, false, nil
	case "reg:logistic", "binary:logistic":
		return sigmoidSingle, passthrough, false, nil
	case "multi:softmax", "multi:softprob":
		return identity, Softmax, true, nil
	default:
		return nil, nil, false, fmt.Errorf("unknown objective: %s", obj.Name)
	}
}

// Predict runs inference through the full XGBoost pipeline.
func (m *XGBoostSchema) Predict(features SparseVector) ([]float32, error) {
	switch m.Learner.GradientBooster.GetName() {
	case "gbtree", "gbtree_optimized":
		internalResults, err := m.Learner.GradientBooster.Predict(features)
		if err != nil {
			return nil, fmt.Errorf("gradient booster predict: %w", err)
		}

		numClasses := m.Learner.LearnerModelParam.NumClass
		if numClasses < 1 {
			numClasses = 1
		}
		treesPerClass := len(internalResults) / numClasses
		perClassScore := make([]float32, numClasses)

		if m.multiclass {
			for i := 0; i < numClasses; i++ {
				for j := 0; j < treesPerClass; j++ {
					perClassScore[i] += internalResults[j*numClasses+i]
				}
				perClassScore[i] = m.perScore(perClassScore[i])
			}
		} else {
			for i := 0; i < numClasses; i++ {
				offset := i * treesPerClass
				for j := 0; j < treesPerClass; j++ {
					perClassScore[i] += internalResults[offset+j]
				}
				perClassScore[i] = m.perScore(perClassScore[i])
			}
		}

		return m.postProcess(perClassScore), nil
	case "gblinear":
		return m.Learner.GradientBooster.Predict(features)
	default:
		return nil, fmt.Errorf("unknown gradient booster: %s", m.Learner.GradientBooster.GetName())
	}
}

// PredictDense runs inference using a dense feature vector (no map overhead).
// Missing features should be set to NaN.
func (m *XGBoostSchema) PredictDense(features []float32) ([]float32, error) {
	booster, ok := m.Learner.GradientBooster.(*GBTModelOptimized)
	if !ok {
		return nil, fmt.Errorf("PredictDense requires gbtree model, got %s", m.Learner.GradientBooster.GetName())
	}

	internalResults := booster.PredictDense(features)

	numClasses := m.Learner.LearnerModelParam.NumClass
	if numClasses < 1 {
		numClasses = 1
	}
	treesPerClass := len(internalResults) / numClasses
	perClassScore := make([]float32, numClasses)

	if m.multiclass {
		for i := 0; i < numClasses; i++ {
			for j := 0; j < treesPerClass; j++ {
				perClassScore[i] += internalResults[j*numClasses+i]
			}
			perClassScore[i] = m.perScore(perClassScore[i])
		}
	} else {
		for i := 0; i < numClasses; i++ {
			offset := i * treesPerClass
			for j := 0; j < treesPerClass; j++ {
				perClassScore[i] += internalResults[offset+j]
			}
			perClassScore[i] = m.perScore(perClassScore[i])
		}
	}

	return m.postProcess(perClassScore), nil
}
