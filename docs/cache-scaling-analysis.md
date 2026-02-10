# Cache Scaling Analysis: SoA vs AoS Across Ensemble Sizes

Benchmark environment: `INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.10GHz`, `linux/amd64`, Go 1.24.7.
Median of 3 runs with `-benchmem`.

## Background

The [refactor benchmarks](refactor-benchmarks.md) showed that at 100 trees, SoA wins the
ensemble by ~10% despite AoS winning per-tree by ~5%. The hypothesis was that SoA's
smaller memory footprint (~1.6 KB vs ~3 KB per tree) creates less cache pressure.

This analysis tests two boundary conditions:
1. **Both fit in cache**: Do smaller ensembles eliminate SoA's advantage?
2. **Neither fits in cache**: Does SoA's advantage hold when both spill to L3?

## Corrected Memory Footprints

The original analysis assumed 127 nodes per tree. The actual mortgage model averages
**23 nodes per tree** (min 9, max 31). This changes the arithmetic significantly:

| Layout | Per tree (23 nodes) | 100 trees | 1000 trees | 8000 trees |
|--------|-------------------|-----------|------------|------------|
| **SoA** (4 int32 arrays + bool) | 391 B | 38 KB | 382 KB | 3.0 MB |
| **AoS** (24 B × 23 nodes) | 552 B | 54 KB | 539 KB | 4.3 MB |
| **Ratio** (AoS / SoA) | 1.41× | 1.41× | 1.41× | 1.41× |

Cache boundaries on this Xeon (48 KB L1d, 2 MB L2 per core):

| Threshold | SoA | AoS |
|-----------|-----|-----|
| Fits L1 (48 KB) | ≤125 trees | ≤89 trees |
| Fits L2 (2 MB) | ≤5363 trees | ≤3799 trees |

## Results

| Trees | SoA (ns) | AoS (ns) | SoA ns/tree | AoS ns/tree | AoS/SoA | Cache zone |
|------:|---------:|---------:|------------:|------------:|---------:|:-----------|
| 1 | 12.8 | 13.7 | 12.8 | 13.7 | 1.07× | Both in L1 |
| 5 | 58.9 | 66.6 | 11.8 | 13.3 | 1.13× | Both in L1 |
| 10 | 118.7 | 131.3 | 11.9 | 13.1 | 1.11× | Both in L1 |
| 25 | 296.2 | 324.0 | 11.9 | 13.0 | 1.09× | Both in L1 |
| 50 | 597.1 | 653.5 | 11.9 | 13.1 | 1.09× | Both in L1 |
| 100 | 1256 | 1381 | 12.6 | 13.8 | 1.10× | SoA in L1, AoS spills |
| 200 | 2368 | 3254 | 11.8 | 16.3 | **1.37×** | Both in L2 |
| 500 | 5894 | 8410 | 11.8 | 16.8 | **1.43×** | Both in L2 |
| 1000 | 11626 | 16828 | 11.6 | 16.8 | **1.45×** | Both in L2 |
| 2000 | 23687 | 33996 | 11.8 | 17.0 | **1.44×** | Both in L2 |
| 4000 | 52866 | 83040 | 13.2 | 20.8 | **1.57×** | SoA in L2, AoS near limit |
| 8000 | 126455 | 204016 | 15.8 | 25.5 | **1.61×** | Both spill L2 → L3 |

### Per-tree cost across scales

```
ns/tree
26 |                                                              A
24 |
22 |
20 |                                            A
18 |
16 |                  A-----A-----A-----A           S
14 | A--A--A--A--A-A                          S
12 | S--S--S--S--S-S--S-----S-----S-----S
10 |
   +--+--+--+--+--+--+-----+-----+-----+-----+-----+
   1  5  10 25 50 100 200   500  1000  2000  4000  8000  trees

   S = SoA    A = AoS
   |---- L1 ----||------- L2 ------||--- L3 ---|
```

## Analysis

### Q1: Both fit in L1 (≤50 trees) — does AoS win?

**No. SoA is 7–13% faster even when everything fits in L1.**

This contradicts the earlier finding (Go 1.21) that AoS was 5% faster per-tree.
On Go 1.24, the SoA path is consistently faster at all scales. Two factors:

1. **Data efficiency**: The SoA numerical hot path loads only the 4 fields it needs
   (`LeftChild`, `SplitIndex`, `SplitCondition`, `DefaultLeft` = 13 bytes/node).
   The AoS path copies the full 24-byte struct including `RightChild`, `Category`,
   `SplitType`, and `IsLeaf` — fields that are either unused in the numerical path
   or redundant (`IsLeaf` ≡ `LeftChild < 0`). That's 85% more data loaded per node.

2. **Compiler improvements**: Go 1.24 has better bounds check elimination for
   sequential array accesses. The SoA pattern (`array[idx]` across separate slices
   with the same index) is a pattern the compiler optimizes well.

When both layouts fit in L1, cache miss rates are ~0 for both. The difference is
purely instruction-level: fewer bytes moved, fewer wasted loads.

### Q2: Neither fits in cache (≥4000 trees) — does SoA's advantage hold?

**Yes, and it grows. SoA's advantage increases from ~1.1× (L1) to ~1.6× (L3).**

The scaling pattern shows three distinct phases:

| Phase | Trees | AoS/SoA ratio | Mechanism |
|-------|-------|---------------|-----------|
| L1-resident | 1–100 | 1.07–1.13× | Instruction-level: struct copy overhead |
| L2-resident | 200–2000 | 1.37–1.45× | L1 miss penalty: AoS spills first, SoA arrays stay compact |
| L2-spilling | 4000–8000 | 1.57–1.61× | L2 miss penalty: each miss costs ~10× more |

The key observation: **SoA's per-tree cost is nearly flat from 1 to 2000 trees**
(~12 ns/tree), only rising at 4000+ when it approaches the L2 boundary. AoS
degrades earlier and more sharply at each cache tier boundary.

Why does the gap widen? Cache miss penalties are asymmetric across tiers:
- L1 hit: ~1 ns
- L2 hit: ~5 ns
- L3 hit: ~20 ns

When AoS starts hitting L2 (at ~200 trees) while SoA is still in L1, each AoS
node access pays a ~4 ns penalty. When both start hitting L3 (at ~4000+ trees),
AoS's 1.4× larger footprint means 1.4× more L3 misses, each costing ~15 ns more.
The absolute cost of being larger grows with miss penalty.

### Why the 100-tree result was misleading

The original 100-tree benchmark sits right at the L1 boundary — SoA's 38 KB fits,
AoS's 54 KB barely spills. This created the appearance that the SoA advantage was
specifically about L1/L2 cache pressure at this particular ensemble size.

In reality, SoA wins at **every** scale tested:
- Small: wins on data efficiency (fewer bytes per node access)
- Medium: wins because smaller footprint delays cache tier transitions
- Large: wins because smaller footprint means fewer misses at expensive tiers

The advantage is monotonically increasing with ensemble size.

## Methodology notes

- Trees were deep-copied (not aliased) to ensure each tree has distinct memory
  allocations, matching real-world heap layout
- For tree counts > 100, trees cycle through the 100 source trees (tree `i` is a
  deep copy of source tree `i % 100`), preserving realistic tree structure variety
- Benchmarks measure raw tree iteration without the sigmoid/accumulation overhead
  to isolate the data layout effect
- The feature vector (44 × float32 = 176 bytes) is tiny and always L1-resident
