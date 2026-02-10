# Struct Refactor: Benchmark Results

Benchmark environment: `INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.10GHz`, `linux/amd64`, Go 1.21.
All numbers are median of 5 runs with `-benchmem`.

## Summary

| Benchmark | Before | After (sparse) | After (dense) | Speedup |
|-----------|--------|-----------------|---------------|---------|
| Single tree traversal | 35.3 ns | 31.5 ns | **13.4 ns** | 2.6x (dense) |
| Full pipeline (100 trees) | 6505 ns | 5721 ns | **3616 ns** | 1.8x (dense) |
| End-to-end CSV data | 8817 ns | 7661 ns | **3044 ns** | 2.9x (dense) |
| Model loading | 8.98 ms / 852 KB / 6587 allocs | 8.58 ms / 691 KB / 4459 allocs | — | 19% less memory, 32% fewer allocs |

## What Changed

### 1. Struct-of-arrays tree layout

Before: `[]*NodeOptimized` — a slice of pointers to 80-byte structs scattered across the heap.

```go
type NodeOptimized struct {
    CategoricalSize   int       // 8 bytes
    Category          int       // 8 bytes
    CategoriesNode    int       // 8 bytes
    CategoriesSegment int       // 8 bytes
    LeftChild         int       // 8 bytes
    RightChild        int       // 8 bytes
    SplitIndex        int       // 8 bytes
    SplitType         int       // 8 bytes
    SplitCondition    float32   // 4 bytes
    DefaultLeft       bool      // 1 byte + padding
    IsLeaf            bool      // 1 byte + padding
}
// Total: 80 bytes per node, pointer-chasing to reach each one
```

After: struct-of-arrays with `int32` fields, contiguous memory per field.

```go
type TreeOptimized struct {
    LeftChild      []int32     // hot path
    RightChild     []int32
    SplitIndex     []int32     // hot path
    SplitCondition []float32   // hot path
    DefaultLeft    []bool      // hot path
    SplitType      []uint8     // cold (only if categorical)
    Category       []int32     // cold (only if categorical)
    HasCategorical bool
}
```

For a tree with 127 nodes, the hot-path arrays (`LeftChild`, `SplitIndex`, `SplitCondition`, `DefaultLeft`) total ~1.6 KB of contiguous memory vs ~10 KB of scattered node structs. Categorical fields are only allocated when the tree actually has categorical splits.

### 2. Dense `[]float32` prediction path

Before: all prediction went through `SparseVector` (`map[int]float32`), even when features were dense. `PredictFloats` converted `[]float32` → map → did map lookups.

After: `PredictDense([]float32)` does direct array indexing during tree traversal. Missing values are represented as NaN. No map allocation, no hash computation, no bucket scanning.

This is the single biggest win — the tree traversal inner loop goes from map lookups (~15-25 ns each with hash + bucket scan) to array indexing (~1 ns).

### 3. Objective resolved at load time

Before: 3 nested `switch` statements on `objective.Name` strings in the prediction hot path, evaluated on every call.

After: `resolveObjective()` runs once at model load and returns function pointers (`perScoreFn`, `postProcessFn`) stored on the schema. The prediction loop calls these directly — no string comparisons at inference time.

### 4. Proper error handling

Replaced `mustNotError()` (panics on bad input) with explicit error returns in `UnmarshalJSON`. A library should never panic on malformed data.

### 5. Dead code removed

- `tree.Predict()` that returned `0.0, nil` unconditionally
- `GBTree.Predict()` (dead after conversion to optimized)
- `OptimizedGBDTClassifier` (replaced by `XGBoostSchema.PredictDense`)
- Unused `Leaves []bool` field on `tree`
- Duplicate `sigmoidSingleOpt` / `sigmoidOpt`
- `mustNotError` utility
- Custom `max` generic (Go 1.21 builtin)
- `github.com/pkg/errors` and `golang.org/x/exp` dependencies

## Detailed Numbers

### Single tree prediction

```
BEFORE:
BenchmarkXGBoostTree-16     35.27 ns/op    0 B/op    0 allocs/op

AFTER:
BenchmarkXGBoostTree-16       31.47 ns/op    0 B/op    0 allocs/op  (sparse, 11% faster)
BenchmarkXGBoostTreeDense-16  13.41 ns/op    0 B/op    0 allocs/op  (dense, 62% faster)
```

The dense path is 2.6x faster per tree. Over 100 trees this compounds.

### Full pipeline — mortgage model (100 trees, binary classification)

```
BEFORE:
BenchmarkXGBoost-16            6505 ns/op    845 B/op    4 allocs/op
BenchmarkXGBoostOptimized-16   6023 ns/op     12 B/op    2 allocs/op

AFTER:
BenchmarkXGBoost-16            5721 ns/op    840 B/op    4 allocs/op  (sparse, 12% faster)
BenchmarkXGBoostDense-16       3616 ns/op    840 B/op    4 allocs/op  (dense, 44% faster than old sparse)
```

Note: the old `BenchmarkXGBoostOptimized` used `OptimizedGBDTClassifier` which skipped
objective handling (always applied sigmoid regardless of model type). The new dense path
is 40% faster AND handles objectives correctly.

### End-to-end with real data (CSV → predict)

```
BEFORE:
BenchmarkXGBEndToEnd-16              8817 ns/op    1674 B/op    6 allocs/op
BenchmarkXGBEndToEndOptimized-16     8077 ns/op    1209 B/op    4 allocs/op

AFTER:
BenchmarkXGBEndToEnd-16              7661 ns/op    1660 B/op    6 allocs/op  (sparse, 13% faster)
BenchmarkXGBEndToEndDense-16         3044 ns/op     420 B/op    2 allocs/op  (dense, 65% faster, 75% less memory)
```

The dense end-to-end path eliminates both the `SparseVectorFromArray` map allocation
and the per-lookup hash overhead. Allocs drop from 6 to 2 (just the tree results slice
and the per-class score slice).

### Model loading

```
BEFORE:
BenchmarkLoadXGBoost-16    8,979,216 ns/op    851,980 B/op    6,587 allocs/op

AFTER:
BenchmarkLoadXGBoost-16    8,583,944 ns/op    690,941 B/op    4,459 allocs/op
```

19% less memory allocated, 32% fewer allocations. The savings come from:
- Value slices instead of per-node pointer allocations (eliminates ~N allocs per tree)
- `int32` fields instead of `int` (halves index array memory)
- Categorical arrays only allocated when needed

## Benchmarks Removed

The concurrent benchmarks (`BenchmarkXGBoostConcurrent`, `BenchmarkXGBoostTreeConcurrent`,
`BenchmarkXGBEndToEndConcurrent`, `BenchmarkXGBEndToEndOptimizedConcurrent`) were removed.
They launched goroutines without synchronization (the guard channel was commented out),
so the benchmark completed before goroutines finished, producing unreliable numbers.
