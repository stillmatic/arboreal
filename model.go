package arboreal

import (
	"encoding/json"
	"github.com/pkg/errors"
	"os"
)

func NewGBDTFromXGBoostJSON(filename string) (*XGBoostSchema, error) {
	var schema *XGBoostSchema
	jsonIO, err := os.ReadFile(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open %s", filename)
	}
	err = json.Unmarshal(jsonIO, &schema)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't unmarshal json")
	}
	return schema, nil
}
