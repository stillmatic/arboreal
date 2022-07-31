package arboreal

import (
	"encoding/json"
)

// MustNotError is a generic function to get the output of a function that returns
// a value and an error. If the error is not nil, it will panic.
func MustNotError[T any](input T, err error) T {
	if err != nil {
		panic(err)
	}
	return input
}

func PrettyPrint(i interface{}) string {
	s, _ := json.MarshalIndent(i, "", "\t")
	return string(s)
}
