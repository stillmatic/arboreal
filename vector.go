package arboreal

type SparseVector map[int]float64

func SparseVectorFromArray(arr []float64) SparseVector {
	sv := make(SparseVector, len(arr))
	for i, v := range arr {
		sv[i] = v
	}
	return sv
}
