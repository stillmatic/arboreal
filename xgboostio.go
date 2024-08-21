package arboreal

// IO for XGBoost JSON files
// see https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
// https://github.com/dmlc/xgboost/blob/24c237308097b693b744af2ad1f86f44be068523/demo/json-model/json_parser.py

import (
	"encoding/json"
	"strconv"

	// "github.com/goccy/go-json"

	"github.com/pkg/errors"
)

// UnmarshalJSON is a custom JSON unmarshal for learner
func (l *learner) UnmarshalJSON(b []byte) error {
	var tmp struct {
		FeatureNames      []featureName     `json:"feature_names,omitempty"`
		FeatureTypes      []featureType     `json:"feature_types,omitempty"`
		GradientBooster   json.RawMessage   `json:"gradient_booster"`
		LearnerModelParam learnerModelParam `json:"learner_model_param,omitempty"`
		Objective         json.RawMessage   `json:"objective"`
	}
	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}
	l.FeatureNames = tmp.FeatureNames
	l.FeatureTypes = tmp.FeatureTypes
	l.LearnerModelParam = tmp.LearnerModelParam
	var err error
	l.GradientBooster, err = parseGradientBooster(tmp.GradientBooster)
	if err != nil {
		return errors.Wrapf(err, "failed to parse gradient booster")
	}
	l.Objective, err = parseObjective(tmp.Objective)
	if err != nil {
		return errors.Wrapf(err, "failed to parse objective")
	}
	return nil
}

func parseGradientBooster(msg json.RawMessage) (GradientBooster, error) {
	var tmp struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(msg, &tmp); err != nil {
		return nil, err
	}
	switch tmp.Name {
	case "gbtree":
		var gbtree GBTree
		if err := json.Unmarshal(msg, &gbtree); err != nil {
			return nil, err
		}
		optimized := OptimizedGBTModel(&gbtree.Model)
		return optimized, nil
	case "gblinear":
		var gblinear GBLinear
		if err := json.Unmarshal(msg, &gblinear); err != nil {
			return nil, err
		}
		return &gblinear, nil
	}
	return nil, nil
}

// custom JSON unmarshal for LearnerModelParam
func (l *learnerModelParam) UnmarshalJSON(b []byte) error {
	var tmp struct {
		BaseScore  string `json:"base_score,omitempty"`
		NumClass   string `json:"num_class,omitempty"`
		NumFeature string `json:"num_feature,omitempty"`
	}
	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}
	l.BaseScore = float32(MustNotError(strconv.ParseFloat(tmp.BaseScore, 64)))
	l.NumClass = MustNotError(strconv.Atoi(tmp.NumClass))
	l.NumFeature = MustNotError(strconv.Atoi(tmp.NumFeature))
	return nil
}

func OptimizedGBTModel(in *model) *GBTModelOptimized {
	out := &GBTModelOptimized{
		Trees: make([]*TreeOptimized, len(in.Trees)),
	}
	for idx, tree := range in.Trees {
		out.Trees[idx] = OptimizedTree(tree)
	}
	return out
}

func OptimizedTree(in *tree) *TreeOptimized {
	out := &TreeOptimized{}
	nodes := make([]*NodeOptimized, len(in.LeftChildren))
	for i := range nodes {
		nodes[i] = &NodeOptimized{
			DefaultLeft:    in.DefaultLeft[i] == 1,
			LeftChild:      in.LeftChildren[i],
			RightChild:     in.RightChildren[i],
			SplitCondition: in.SplitConditions[i],
			SplitIndex:     in.SplitIndices[i],
			SplitType:      in.SplitType[i],
			// saves a lookup
			IsLeaf: (in.LeftChildren[i] == -1) && (in.RightChildren[i] == -1),
		}
		if len(in.CategoriesSizes) > 0 {
			nodes[i].CategoricalSize = in.CategoriesSizes[i]
			nodes[i].Category = in.Categories[i]
			nodes[i].CategoriesNode = in.CategoriesNodes[i]
			nodes[i].CategoriesSegment = in.CategoriesSegments[i]
		}
	}
	out.Nodes = nodes
	return out
}

type Objective struct {
	Name   string `json:"name"`
	Params map[string]interface{}
}

func parseObjective(msg json.RawMessage) (Objective, error) {
	var tmp struct {
		Name          string      `json:"name"`
		RegLossParams interface{} `json:"reg_loss_param"`
	}
	err := json.Unmarshal(msg, &tmp)
	if err != nil {
		return Objective{}, err
	}
	// TODO: this might be a map[string]string
	var params map[string]interface{}
	if tmp.RegLossParams != nil {
		params = tmp.RegLossParams.(map[string]interface{})
	}
	return Objective{
		Name:   tmp.Name,
		Params: params,
	}, nil
}
