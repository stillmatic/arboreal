package arboreal

// IO for XGBoost JSON files
// see https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
// https://github.com/dmlc/xgboost/blob/24c237308097b693b744af2ad1f86f44be068523/demo/json-model/json_parser.py

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// UnmarshalJSON is a custom JSON unmarshal for learner.
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
		return fmt.Errorf("failed to parse gradient booster: %w", err)
	}
	l.Objective, err = parseObjective(tmp.Objective)
	if err != nil {
		return fmt.Errorf("failed to parse objective: %w", err)
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
	return nil, fmt.Errorf("unknown gradient booster: %s", tmp.Name)
}

// UnmarshalJSON for learnerModelParam handles the string-to-number conversion
// that XGBoost's JSON format requires.
func (l *learnerModelParam) UnmarshalJSON(b []byte) error {
	var tmp struct {
		BaseScore  string `json:"base_score,omitempty"`
		NumClass   string `json:"num_class,omitempty"`
		NumFeature string `json:"num_feature,omitempty"`
	}
	if err := json.Unmarshal(b, &tmp); err != nil {
		return err
	}

	if tmp.BaseScore != "" {
		bs, err := strconv.ParseFloat(tmp.BaseScore, 64)
		if err != nil {
			return fmt.Errorf("invalid base_score %q: %w", tmp.BaseScore, err)
		}
		l.BaseScore = float32(bs)
	}
	if tmp.NumClass != "" {
		nc, err := strconv.Atoi(tmp.NumClass)
		if err != nil {
			return fmt.Errorf("invalid num_class %q: %w", tmp.NumClass, err)
		}
		l.NumClass = nc
	}
	if tmp.NumFeature != "" {
		nf, err := strconv.Atoi(tmp.NumFeature)
		if err != nil {
			return fmt.Errorf("invalid num_feature %q: %w", tmp.NumFeature, err)
		}
		l.NumFeature = nf
	}
	return nil
}

// OptimizedGBTModel converts a parsed model to the SoA optimized representation.
func OptimizedGBTModel(in *model) *GBTModelOptimized {
	out := &GBTModelOptimized{
		Trees: make([]TreeOptimized, len(in.Trees)),
	}
	for idx, tree := range in.Trees {
		out.Trees[idx] = OptimizedTree(tree)
	}
	return out
}

// OptimizedTree converts a single parsed tree to SoA layout.
func OptimizedTree(in *tree) TreeOptimized {
	n := len(in.LeftChildren)
	out := TreeOptimized{
		LeftChild:      make([]int32, n),
		RightChild:     make([]int32, n),
		SplitIndex:     make([]int32, n),
		SplitCondition: in.SplitConditions, // reuse the slice directly
		DefaultLeft:    make([]bool, n),
	}

	hasCategorical := false
	for i := 0; i < n; i++ {
		out.LeftChild[i] = int32(in.LeftChildren[i])
		out.RightChild[i] = int32(in.RightChildren[i])
		out.SplitIndex[i] = int32(in.SplitIndices[i])
		out.DefaultLeft[i] = in.DefaultLeft[i] == 1
		if len(in.SplitType) > i && in.SplitType[i] == 1 {
			hasCategorical = true
		}
	}

	out.HasCategorical = hasCategorical
	if hasCategorical {
		out.SplitType = make([]uint8, n)
		out.Category = make([]int32, n)
		for i := 0; i < n; i++ {
			if len(in.SplitType) > i {
				out.SplitType[i] = uint8(in.SplitType[i])
			}
			if len(in.Categories) > i {
				out.Category[i] = int32(in.Categories[i])
			}
		}
	}

	return out
}

// Objective holds the parsed objective function metadata.
type Objective struct {
	Name   string
	Params map[string]string
}

func parseObjective(msg json.RawMessage) (Objective, error) {
	var tmp struct {
		Name          string            `json:"name"`
		RegLossParam  map[string]string `json:"reg_loss_param"`
	}
	if err := json.Unmarshal(msg, &tmp); err != nil {
		return Objective{}, err
	}
	return Objective{
		Name:   tmp.Name,
		Params: tmp.RegLossParam,
	}, nil
}
