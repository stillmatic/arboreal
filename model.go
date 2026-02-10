package arboreal

import (
	"encoding/json"
	"fmt"
	"os"
)

// NewGBDTFromXGBoostJSON loads an XGBoost model from a JSON file and resolves
// the objective into function pointers for fast prediction. Builds both SoA
// and AoS tree representations for benchmarking.
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

	// Build AoS model from the raw parsed trees (kept during unmarshal).
	if schema.Learner.rawModel != nil {
		schema.ModelAoS = OptimizedGBTModelAoS(schema.Learner.rawModel)
		schema.Learner.rawModel = nil // release parsed tree data
	}

	return &schema, nil
}
