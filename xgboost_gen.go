package arboreal

// generated by "schematyper pkg/xgb/xgboost_schema.json -c"
// minor edits for readability/usability (i.e. removing interfaces)

type aftLossParam struct {
	AftLossDistribution      string `json:"aft_loss_distribution,omitempty"`
	AftLossDistributionScale string `json:"aft_loss_distribution_scale,omitempty"`
}

type categoricalSize int

type categoriesNode int

type categoriesSegment int

type category int

type featureName string

type featureType string

type GBTree struct {
	Model model  `json:"model"`
	Name  string `json:"name"`
}

type gbtreeModelParam struct {
	NumTrees string `json:"num_trees"`
	// NumParallelTree string `json:"num_parallel_tree,omitempty"`
	// SizeLeafVector  string `json:"size_leaf_vector"`
}

type lambdaRankParam struct {
	FixListWeight string `json:"fix_list_weight,omitempty"`
	NumPairsample string `json:"num_pairsample,omitempty"`
}

type learner struct {
	FeatureNames      []featureName     `json:"feature_names,omitempty"`
	FeatureTypes      []featureType     `json:"feature_types,omitempty"`
	GradientBooster   GradientBooster   `json:"gradient_booster"`
	LearnerModelParam learnerModelParam `json:"learner_model_param,omitempty"`
	Objective         interface{}       `json:"objective"`
}

type learnerModelParam struct {
	BaseScore  float64 `json:"base_score,omitempty"`
	NumClass   int     `json:"num_class,omitempty"`
	NumFeature int     `json:"num_feature,omitempty"`
}

type model struct {
	GbtreeModelParam gbtreeModelParam `json:"gbtree_model_param"`
	// TreeInfo         []treeInfoItem   `json:"tree_info"`
	Trees []*tree `json:"trees"`
}

type pseduoHuberParam struct {
	HuberSlope string `json:"huber_slope,omitempty"`
}

type regLossParam struct {
	ScalePosWeight string `json:"scale_pos_weight,omitempty"`
}

type softmaxMulticlassParam struct {
	NumClass string `json:"num_class,omitempty"`
}

type tree struct {
	// BaseWeights        []float64 `json:"base_weights"`
	CategoricalSizes   []int `json:"categorical_sizes,omitempty"`
	Categories         []int `json:"categories"`
	CategoriesNodes    []int `json:"categories_nodes"`
	CategoriesSegments []int `json:"categories_segments"`
	// DefaultLeft determines when feature is unknown, whether goes to left child
	DefaultLeft     []int     `json:"default_left"`
	ID              int       `json:"id,omitempty"`
	LeftChildren    []int     `json:"left_children"`
	RightChildren   []int     `json:"right_children"`
	SplitConditions []float64 `json:"split_conditions"`
	SplitIndices    []int     `json:"split_indices"`
	SplitType       []int     `json:"split_type,omitempty"`
	Leaves          []bool
	// Parents         []int            `json:"parents"`
	// TreeParam       treeTreeParam    `json:"tree_param"`
	// SumHessian  []float64 `json:"sum_hessian"`
	// LossChanges []float64 `json:"loss_changes"`
}

type treeInfoItem int

type treeTreeParam struct {
	NumFeature     string `json:"num_feature"`
	NumNodes       string `json:"num_nodes"`
	SizeLeafVector string `json:"size_leaf_vector"`
}

type xgboostSchema struct {
	Learner *learner `json:"learner"`
	Version []int    `json:"version"`
}

type xgboostSchemaTreeParam struct {
	NumFeature     string `json:"num_feature"`
	NumNodes       string `json:"num_nodes"`
	SizeLeafVector string `json:"size_leaf_vector"`
}