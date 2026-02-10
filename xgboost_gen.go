package arboreal

type featureName string

type featureType string

// GBTree is the deserialization target for gbtree gradient boosters.
// It is converted to GBTModelOptimized at load time.
type GBTree struct {
	Model model  `json:"model"`
	Name  string `json:"name"`
}

type gbtreeModelParam struct {
	NumTrees        string `json:"num_trees"`
	NumParallelTree string `json:"num_parallel_tree"`
}

type learner struct {
	FeatureNames      []featureName     `json:"feature_names,omitempty"`
	FeatureTypes      []featureType     `json:"feature_types,omitempty"`
	GradientBooster   GradientBooster   `json:"gradient_booster"`
	LearnerModelParam learnerModelParam `json:"learner_model_param,omitempty"`
	Objective         Objective         `json:"objective"`

	// rawModel is kept temporarily after unmarshal so NewGBDTFromXGBoostJSON
	// can build both SoA and AoS representations from the same parsed data.
	rawModel *model
}

type learnerModelParam struct {
	BaseScore  float32 `json:"base_score,omitempty"`
	NumClass   int     `json:"num_class,omitempty"`
	NumFeature int     `json:"num_feature,omitempty"`
	NumTarget  int     `json:"num_target,omitempty"`
}

type model struct {
	GbtreeModelParam gbtreeModelParam `json:"gbtree_model_param"`
	Trees            []*tree          `json:"trees"`
}

type tree struct {
	CategoriesSizes    []int     `json:"categories_sizes,omitempty"`
	Categories         []int     `json:"categories"`
	CategoriesNodes    []int     `json:"categories_nodes"`
	CategoriesSegments []int     `json:"categories_segments"`
	DefaultLeft        []int     `json:"default_left"`
	ID                 int       `json:"id,omitempty"`
	LeftChildren       []int     `json:"left_children"`
	RightChildren      []int     `json:"right_children"`
	SplitConditions    []float32 `json:"split_conditions"`
	SplitIndices       []int     `json:"split_indices"`
	SplitType          []int     `json:"split_type,omitempty"`
}

// XGBoostSchema is the top-level model representation. After loading,
// perScore and postProcess are resolved from the objective so the
// prediction hot path has no switch statements.
type XGBoostSchema struct {
	Learner *learner `json:"learner"`
	Version []int    `json:"version"`

	// resolved at load time by resolveObjective
	perScore    perScoreFn
	postProcess postProcessFn
	multiclass  bool

	// AoS model for benchmarking comparison (built alongside SoA at load time)
	ModelAoS *GBTModelAoS
}
