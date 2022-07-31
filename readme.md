# arboreal

> (chiefly of animals) living in trees.

![arboreal logo](arboreal.png)

Pure Go library for gradient boosted decision trees. The library is optimized for ease of use in a production web server, serving real-time inference requests. As a result, we do not support training models and eschew non-standard libaries for portability.

# Usage

```go
res, err := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
inpArr := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
inpVec := arboreal.SparseVectorFromArray(inpArr)
res.Predict(inpVec)
```

# Why?

Go is a great language for backend development, especially for web-facing apps. However, it is not a great language for machine learning. Machine learning and data science deal with messy data and benefit from the comparative flexibility of Python. Building models in Python is easy, but serving them is not particularly quick - Python's GIL and weak concurrency model make it difficult to serve many requests at high throughput.

This library aims to solve that problem. It is a pure Go implementation of gradient boosted decision trees, possibly the most popular machine learning model type within enterprise applications. It is optimized for serving inference requests, and is fast enough to be used in a production web server.

# Benchmarks

These are fairly simplistic and not very scientific. They are intended to give a rough idea of the performance of the library. The model tested has 100 trees of depth 4.

```
<<<<<<< HEAD

BenchmarkXGBoost-10 201556 5111 ns/op 1813 B/op 4 allocs/op
BenchmarkXGBoostConcurrent-10 1812044 660.5 ns/op 952 B/op 4 allocs/op

```

# The library is optimized for speed of inference. It takes ~5 microseconds to infer a row in a single-threaded setup and <1 microsecond in a concurrent setup. This benchmark is on my laptop and I encourage you to try your own.

BenchmarkXGBoost-10 196039 5223 ns/op 1829 B/op 4 allocs/op
BenchmarkXGBoostConcurrent-10 793015 1496 ns/op 969 B/op 4 allocs/op
BenchmarkLoadXGBoost-10 200 5957921 ns/op 1062448 B/op 17759 allocs/op

```

The library is optimized for speed of inference. It takes ~5 microseconds to infer a row in a single-threaded setup and ~1.5 microseconds in a concurrent setup. This benchmark is on my laptop and I encourage you to try your own.
>>>>>>> a0e600d (feat: initial commit)

On loading: this benchmark shows ~6ms to load a small model, including SSD I/O. You can make it faster by swapping out the default JSON parser for a faster one, e.g. go-json. In my testing, that shows close to a 4x speedup to ~1.5ms. However, go-json shows a 50% greater use of memory, and I'm not sure why. In some cases, you may want to optimize for loading speed - e.g. if you are constantly swapping models (e.g. limited memory and want to use a single box to load lots of models). But in most cases, you don't really need to shave milliseconds off loading time and would prefer to use less memory.

It's a bit annoying to compare directly against XGBoost and other libraries. These results are quite good though, and are on the same order of magnitude as FPGA algorithms. In practice, preprocessing the input data will dominate inference time, as will network latency. I will likely not pursue further optimizations unless there is a clear need.

<<<<<<< HEAD
### Wholely unnecessary optimizations

We do not concurrently process each tree in the ensemble. Each tree is generally very fast - e.g. 11 nanoseconds for a tree of depth 4 in this benchmark, so the overhead of spawning a goroutine is not worth it. It is possible that your use case will be different, e.g. with lots of very deep trees. Similarly, even if you move the concurrency up to the model level, and run a goroutine per tree while summing the results with a channel, there is quite a lot of overhead that makes it much slower than the serial case.

```

BenchmarkXGBoostTree-10 111218052 10.98 ns/op 0 B/op 0 allocs/op
BenchmarkXGBoostTreeConcurrent-10 4676050 247.3 ns/op 48 B/op 2 allocs/op

```

But, like, we can absolutely tune the hell out of this if we want.

```

BenchmarkXGBoost-10 246240 4803 ns/op 844 B/op 4 allocs/op
BenchmarkXGBoostOptimized-10 252561 4250 ns/op 12 B/op 2 allocs/op
BenchmarkXGBoostConcurrent-10 2146634 591.4 ns/op 469 B/op 4 allocs/op

```

Here, we've applied a few optimizations:

- `float32` instead of `float32`: XGBoost also uses single-precision floats, so this actually aligns us more closely with XGBoost's defaults. Both benchmarks above use this, it was just way easier to find/replace everywhere. This does make me a bit uncomfortable, as we're probably doing a bunch of back and forth translations, even with [math32](https://github.com/chewxy/math32).
- Inlined tree predict functions: we used 'outlining' (splitting functions into smaller ones) to increase performance, at the cost of increasing the binary size. This is reasonable, given that the tree predict functions are absolutely in the hot path and called often.
- Cache locality: XGBoost uses lists to represent its info, so you can represent a node as an integer index into each of its lists. The alternative is keeping a single list of structs which hold all of the info for a node. The latter is more efficient on modern architectures, as the data for a given node is stored close to each other, the CPU can also access it more quickly.
- Move transformation code into the predict loops. This is actually quite a large improvement, since we're no longer allocating a slice of floats into the heap to pass to transformations. This is probably worth it, as it's a 10-15% speedup and something like a 99% reduction in memory overhead, at the cost of making the predict code uglier - need to handle all the cases instead of delegating.

=======
We do not concurrently process each tree in the ensemble. Each tree is generally very fast - e.g. 11 nanoseconds for a tree of depth 4 in this benchmark, so the overhead of spawning a goroutine is not worth it. It is possible that your use case will be different, e.g. with lots of very deep trees.

```

BenchmarkXGBoostTree-10 111218052 10.98 ns/op 0 B/op 0 allocs/op
BenchmarkXGBoostTreeConcurrent-10 4676050 247.3 ns/op 48 B/op 2 allocs/op

```

>>>>>>> a0e600d (feat: initial commit)
# Caveats

- The XGBoost model format may change
- XGBoost internally uses single-precision float32's. Go's standard library uses float32's internally and it's a bit of a pain to use float32's (e.g. we're just casting back and forth). This may lead to slightly different results (and slightly worse performance than if we can use float32's directly).

# Learnings

Some observations

- I haven't applied many "clever" optimizations at all to this code. I aimed to write clean code that derived from the XGBoost export format and was easy to understand. This was pretty straightforward, and I was surprised at how easy it was to get good performance and to unmarshal the file format quickly (less than an hour, taking advantage of autogenerated structs). I'm sure there are some optimizations that could be made, but I'm not sure they would be worth it.
- Python's interop with C++ and Rust is great, but mainly that's a function of C++ and Rust offering a simple FFI. Because Go does not offer a similar setup, it's really difficult to use Go to optimize code that's later used in Python. While this library is designed to be plugged into a web server in Go, if I was going to write something more focused on the data science / ML workflow, I would do it in Rust for better portability.
- Likewise, the `cgo` API is a nightmare to work with. This is a shame, since sometimes you really do want to hyperoptimize a hot path or you want to use a library written in C++. Before writing this library, I tried to have a 'simpler' implementation that used `cgo` to call into XGBoost. But you're just trading off complexity in one arena for complexity elsewhere, and I found it to be a net negative.
- Go's philosophy of offering more batteries included than C++/Rust while still offering more control of the underlying system than Python is great for lots of things, but seems best if you limit your service's scope to problems that can be solved with pure Go. That seems relevant given how Go was developed for Google engineers to build services quickly, with reasonably good performance, and without having to worry about too many low-level things.
- The prediction code for XGBoost is actually quite simple, I'm a bit surprised that other people haven't written many similar implementations. Other efforts exist but I found them quite unintuitive to read and understand, especially since they rely on different file formats to parse the underlying XGBoost model. I made a more pronounced effort in this library to keep the code reasonably simple and readable to understand logically.
- On that point, I am a bit displeased with the XGBoost team's decisions around file formats. They have quite a few different implementations and even though the JSON format is considered to be their 'format of the future,' have even added another UBJSON format. The format is also not documented clearly; while there is a JSON schema, it is not easy to understand how it is actually meant to be used.
```
