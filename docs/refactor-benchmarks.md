# Struct Refactor: Benchmark Results

Benchmark environment: `INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.10GHz`, `linux/amd64`, Go 1.21.
All numbers are median of 5 runs with `-benchmem`.

## 3-Way Comparison: Original → SoA → Compact AoS

Three layouts tested, all with dense `[]float32` input and fused predict loops:

| Benchmark | Original (ptr AoS, sparse) | SoA Dense | Compact AoS Dense |
|-----------|---------------------------|-----------|-------------------|
| **Single tree** | 35.3 ns | 13.7 ns | **13.0 ns** |
| **Full pipeline** (100 trees) | 6505 ns / 845 B / 4 allocs | **2792 ns / 8 B / 2 allocs** | 3101 ns / 8 B / 2 allocs |
| **End-to-end** (CSV data) | 8817 ns / 1674 B / 6 allocs | **2552 ns / 4 B / 1 alloc** | 3085 ns / 4 B / 1 alloc |
| **Model loading** | 8.98 ms / 852 KB / 6587 allocs | 8.37 ms / 752 KB / 4561 allocs | (same load, builds both) |

### Key finding: AoS wins per-tree, SoA wins the ensemble

At the **single tree** level, compact AoS is ~5% faster (13.0 vs 13.7 ns). All node
fields are colocated — one struct load gets everything the traversal needs. This
validates the original cache locality rationale.

But at the **100-tree ensemble** level, SoA is ~10% faster (2792 vs 3101 ns). Why?
Per-tree memory footprint. For 127 nodes:

| Layout | Hot data per tree | Total for 100 trees |
|--------|------------------|---------------------|
| SoA | ~1.6 KB (4 separate arrays) | ~160 KB |
| Compact AoS | ~3.0 KB (one node array) | ~300 KB |

Both fit in L2 (2 MB on this Xeon), but SoA's smaller footprint means less cache
pressure when iterating across 100 different trees. Each SoA array (e.g., `SplitCondition`,
508 bytes for 127 nodes) fits in ~8 cache lines, so after the first node access in a tree,
subsequent accesses to the same field are likely L1 hits.

The AoS struct is 24 bytes/node, so a 64-byte cache line holds ~2.7 nodes. Tree
traversal follows a path through ~7 nodes at scattered indices. With the larger total
footprint, the AoS version sees more L2 misses as it moves between trees.

## What Changed (from original)

### 1. Data layout: pointer AoS → value SoA (primary) + compact AoS (comparison)

**Original**: `[]*NodeOptimized` — 80-byte structs, each a separate heap allocation.
Despite the AoS *intent* for cache locality, the pointer indirection scattered nodes
across the heap, defeating the optimization.

```go
// Original: 80 bytes/node, pointer per node
type NodeOptimized struct {
    CategoricalSize   int       // 8 bytes ← cold, always loaded
    Category          int       // 8 bytes ← cold, always loaded
    CategoriesNode    int       // 8 bytes ← cold, always loaded
    CategoriesSegment int       // 8 bytes ← cold, always loaded
    LeftChild         int       // 8 bytes
    RightChild        int       // 8 bytes
    SplitIndex        int       // 8 bytes
    SplitType         int       // 8 bytes
    SplitCondition    float32   // 4 bytes
    DefaultLeft       bool      // 1 byte + 3 padding
    IsLeaf            bool      // 1 byte + 7 padding
}
```

**SoA**: Separate contiguous arrays per field. `int32` fields halve index memory.
Categorical arrays only allocated when the tree has categorical splits.

```go
type TreeOptimized struct {
    LeftChild      []int32     // 508 bytes for 127 nodes
    RightChild     []int32
    SplitIndex     []int32
    SplitCondition []float32
    DefaultLeft    []bool      // 127 bytes
    SplitType      []uint8     // nil if no categorical
    Category       []int32     // nil if no categorical
    HasCategorical bool
}
```

**Compact AoS**: 24-byte value struct in a contiguous slice. Preserves the
single-dereference-per-node pattern from the original.

```go
type NodeAoS struct {
    SplitCondition float32  // 4 bytes
    LeftChild      int32    // 4 bytes
    SplitIndex     int32    // 4 bytes
    RightChild     int32    // 4 bytes
    Category       int32    // 4 bytes
    DefaultLeft    bool     // 1 byte
    IsLeaf         bool     // 1 byte
    SplitType      uint8    // 1 byte
    // 1 byte padding → 24 bytes total
}
```

### 2. Dense `[]float32` prediction path

All prediction previously went through `SparseVector` (`map[int]float32`). Even
`PredictFloats` converted `[]float32` → map → did map lookups.

`PredictDense([]float32)` does direct array indexing. Missing = NaN (`fval != fval`).
No map allocation, no hash computation. This is the single biggest win.

### 3. Fused predict loop (restoring original optimization)

The original `OptimizedGBDTClassifier.Predict` fused tree prediction with per-class
accumulation and sigmoid application in one loop — no intermediate `[]float32` for
all tree results. The optimization_notes.md documented this as "10-15% speedup and
99% reduction in memory overhead."

The initial SoA refactor lost this by allocating `make([]float32, len(trees))` then
iterating separately. Restoring the fused loop:

```
SoA Dense (non-fused):  3616 ns/op    840 B/op    4 allocs/op
SoA Dense (fused):      2792 ns/op      8 B/op    2 allocs/op
                        -------    ------    ------
                         23% faster   99% less    50% fewer
```

This exceeds the originally documented 10-15% because the dense path amplifies the
benefit — the intermediate allocation was a larger fraction of total cost once map
overhead was removed.

### 4. Objective resolved at load time

3 nested `switch` statements on `objective.Name` strings → function pointers
(`perScoreFn`, `postProcessFn`) resolved once at `NewGBDTFromXGBoostJSON` time.
Also fixes the old `OptimizedGBDTClassifier` bug where it always applied sigmoid.

### 5. Original optimizations preserved

- `rightChild = leftChild + 1` for numerical trees (both layouts)
- Separate `predictNumerical`/`predictCategorical` outlining (both layouts)
- `IsLeaf` precomputed at load time (AoS; SoA uses `leftChild < 0`)
- `node := t.Nodes[idx]` single dereference (AoS)

### 6. Proper error handling + dead code removed

- `mustNotError()` panic → explicit error returns in `UnmarshalJSON`
- Removed: `tree.Predict()` (returned 0.0), `GBTree.Predict()`, `OptimizedGBDTClassifier`,
  unused `Leaves` field, duplicate sigmoids, custom `max`, `github.com/pkg/errors`,
  `golang.org/x/exp`

## Detailed Numbers

### Single tree prediction

```
Original:                           35.27 ns/op    0 B/op    0 allocs/op
SoA sparse:                         31.24 ns/op    0 B/op    0 allocs/op
SoA dense:                          13.68 ns/op    0 B/op    0 allocs/op
Compact AoS dense:                  13.04 ns/op    0 B/op    0 allocs/op  ← fastest per-tree
```

### Full pipeline — mortgage model (100 trees, binary classification)

```
Original (sparse):                6505 ns/op    845 B/op    4 allocs/op
Original OptimizedClassifier:     6023 ns/op     12 B/op    2 allocs/op
SoA sparse:                       5827 ns/op    840 B/op    4 allocs/op
SoA dense (fused):                2792 ns/op      8 B/op    2 allocs/op  ← fastest ensemble
Compact AoS dense (fused):        3101 ns/op      8 B/op    2 allocs/op
```

### End-to-end with real data (CSV → predict)

```
Original (sparse):               8817 ns/op   1674 B/op    6 allocs/op
Original optimized:              8077 ns/op   1209 B/op    4 allocs/op
SoA sparse:                      7635 ns/op   1660 B/op    6 allocs/op
SoA dense (fused):               2552 ns/op      4 B/op    1 allocs/op  ← fastest E2E
Compact AoS dense (fused):       3085 ns/op      4 B/op    1 allocs/op
```

### Model loading

```
Original:     8,979,216 ns/op    851,980 B/op    6,587 allocs/op
Refactored:   8,370,042 ns/op    752,182 B/op    4,561 allocs/op
```

12% less memory, 31% fewer allocations.

## Benchmarks Removed

The concurrent benchmarks were removed. They launched goroutines without
synchronization (the guard channel was commented out), producing unreliable numbers.
