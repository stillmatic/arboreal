package arboreal

import (
	"encoding/json"
	"fmt"
	"os"
)

// NewGBDTFromXGBoostJSON loads an XGBoost model from a JSON file and resolves
// the objective into function pointers for fast prediction.
func NewGBDTFromXGBoostJSON(filename string) (*XGBoostSchema, error) {
	jsonIO, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %w", filename, err)
	}
	var schema XGBoostSchema
	if err := json.Unmarshal(jsonIO, &schema); err != nil {
		return nil, fmt.Errorf("couldn't unmarshal json: %w", err)
	}

	schema.perScore, schema.postProcess, schema.multiclass, err = resolveObjective(
		schema.Learner.Objective,
		schema.Learner.LearnerModelParam.BaseScore,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve objective: %w", err)
	}

	return &schema, nil
}
