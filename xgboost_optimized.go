package arboreal

import math "github.com/chewxy/math32"

type OptimizedGBDTClassifier struct {
	Model      *GBTModelOptimized
	NumClasses int
}

func NewOptimizedGBDTClassifierFromSchema(model *xgboostSchema) OptimizedGBDTClassifier {
	origModel := model.Learner.GradientBooster.(*GBTModelOptimized)
	return OptimizedGBDTClassifier{
		Model:      origModel,
		NumClasses: model.Learner.LearnerModelParam.NumClass,
	}
}

func sigmoidOpt(x *[]float32) []float32 {
	for i, v := range *x {
		val := sigmoidSingleOpt(v)
		(*x)[i] = val
	}
	return *x
}

func sigmoidSingleOpt(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (m *OptimizedGBDTClassifier) Predict(features *SparseVector) ([]float32, error) {
	// internalResults, err := m.Model.Predict(features)
	// if err != nil {
	// 	return nil, errors.Wrapf(err, "failed to predict with gradient booster")
	// }

	numClasses := max(m.NumClasses, 1)
	treesPerClass := len(m.Model.Trees) / numClasses
	perClassScore := make([]float32, numClasses)
	for i := 0; i < numClasses; i++ {
		offset := (i * treesPerClass)
		for j := 0; j < treesPerClass; j++ {
			perClassScore[i] += m.Model.Trees[offset+j].Predict(features)
		}
		perClassScore[i] = sigmoidSingleOpt(perClassScore[i])
	}
	// TODO: handle objective
	return perClassScore, nil
}

type GBTModelOptimized struct {
	Trees []*TreeOptimized `json:"trees"`
}

type TreeOptimized struct {
	Nodes []*NodeOptimized
}

type NodeOptimized struct {
	CategoricalSize   int
	Category          int
	CategoriesNode    int
	CategoriesSegment int
	LeftChild         int
	RightChild        int
	SplitIndex        int
	SplitType         int
	SplitCondition    float32
	DefaultLeft       bool
	IsLeaf            bool
}

func (m *GBTModelOptimized) GetName() string {
	return "gbtree_optimized"
}

func (m *GBTModelOptimized) Predict(features *SparseVector) ([]float32, error) {
	result := make([]float32, len(m.Trees))

	for idx, tree := range m.Trees {
		res := tree.Predict(features)
		result[idx] = res
	}
	return result, nil
}

func (t *TreeOptimized) predictCategorical(features *SparseVector) float32 {
	idx := 0

	for {
		node := t.Nodes[idx]
		if node.IsLeaf {
			// splitConditions[idx] is return value for a leaf node
			return node.SplitCondition
		}

		leftChild := node.LeftChild
		// We don't need to do the insane optimization here, as
		// the optimized version already takes advantage of cache locality
		// rightChild := leftChild + 1
		rightChild := node.RightChild

		splitCol := node.SplitIndex
		// splitVal := node.SplitCondition
		fval, ok := (*features)[splitCol]

		// missing value behavior is determined by default left
		if !ok {
			if node.DefaultLeft {
				idx = leftChild
			} else {
				idx = rightChild
			}
			continue
		}
		// todo: doublecheck this
		if int(fval) == node.Category {
			idx = rightChild
		} else {
			idx = leftChild
		}
	}
}

func (t *TreeOptimized) predictNumerical(features *SparseVector) float32 {
	idx := 0
	for {
		node := t.Nodes[idx]
		if node.IsLeaf {
			// splitConditions[idx] is return value for a leaf node
			return node.SplitCondition
		}

		leftChild := node.LeftChild
		// We don't need to do the insane optimization here, as
		// the optimized version already takes advantage of cache locality
		rightChild := leftChild + 1
		// rightChild := node.RightChild

		splitCol := node.SplitIndex
		splitVal := node.SplitCondition
		fval, ok := (*features)[splitCol]

		// missing value behavior is determined by default left
		if !ok {
			if node.DefaultLeft {
				idx = leftChild
			} else {
				idx = rightChild
			}
			continue
		}
		// xgboost uses <, lightgbm uses <=
		if fval < splitVal {
			idx = leftChild
		} else {
			idx = rightChild
		}
	}
}

func (t *TreeOptimized) Predict(features *SparseVector) float32 {
	if t.Nodes[0].SplitType == 1 {
		return t.predictCategorical(features)
	} else {
		return t.predictNumerical(features)
	}
}
