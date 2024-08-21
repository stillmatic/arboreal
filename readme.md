# arboreal

> (chiefly of animals) living in trees.

![arboreal logo](arboreal.jpeg)

Pure Go library for inferencing gradient boosted decision trees. 

# Usage

```bash
go get github.com/stillmatic/arboreal
```

Regression exmaple

```go
schema, _ := arboreal.NewGBDTFromXGBoostJSON("testdata/regression.json")
inpArr := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
inpVec := arboreal.SparseVectorFromArray(inpArr)
res, _ := schema.Predict(inpVec)
```

Optimized classification example

```go
schema, _ := arboreal.NewGBDTFromXGBoostJSON("testdata/mortgage_xgb.json")
newRes := arboreal.NewOptimizedGBDTClassifierFromSchema(schema)
vec := make(arboreal.SparseVector, 44)
res, _ = newRes.Predict(vec)
```


# Why?

Go is a great language for backend development, especially for web-facing apps. However, it is not a great language for machine learning. Machine learning and data science deal with messy data and benefit from the comparative flexibility of Python. Building models in Python is easy, but serving them is not particularly quick - Python's GIL and weak concurrency model make it difficult to serve many requests at high throughput.

This library aims to solve that problem. It is a pure Go implementation of gradient boosted decision trees, possibly the most popular machine learning model type within enterprise applications. It is optimized for serving inference requests, and is fast enough to be used in a production web server.

