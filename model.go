package arboreal

import (
	"encoding/json"
	"io/ioutil"

	"github.com/pkg/errors"
)

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
