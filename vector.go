package arboreal

type SparseVector map[int]float32

func SparseVectorFromArray(arr []float32) SparseVector {
	sv := make(SparseVector, len(arr))
	for i, v := range arr {
		sv[i] = v
	}
	return sv
}
