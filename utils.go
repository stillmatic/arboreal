package arboreal

// mustNotError is a generic function to get the output of a function that returns
// a value and an error. If the error is not nil, it will panic.
func mustNotError[T any](input T, err error) T {
	if err != nil {
		panic(err)
	}
	return input
}
