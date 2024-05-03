# tsgrf
This is a machine learning R package for time series based on grf.

Our experiment has proved that tsgrf performed better than grf on long time series with high volatility。

The current development version can be installed from source using devtools. If you don't install devtools, you can follow this:

```R
install.packages("devtools")
```

```R
devtools::install_github("soso010816/tsgrf", subdir = "tsgrf0.6.1")
```

There is a example for using it.

```R
n <- 500
p <- 10
X <- matrix(rnorm(n * p), n, p)
Y <- X[, 1] * rnorm(n)

# honesty.method : Specifies the method of honesty tree splitting, default is 4.
# nonlapping.block.size: Number of Blocks, The total number of blocks is calculated as n^{1/nonlapping_block_size}. 
r.forest <- regression_forest(X, Y, nonlapping.block.size=2, honesty.method=4)

# Predict using the forest.
X.test <- matrix(0, 101, p)
X.test[, 1] <- seq(-2, 2, length.out = 101)
r.pred <- predict(r.forest, X.test)

# Predict on out-of-bag training samples.
r.pred <- predict(r.forest)

# Predict with confidence intervals; growing more trees is now recommended.
r.forest <- regression_forest(X, Y, num.trees = 100)
r.pred <- predict(r.forest, X.test, estimate.variance = TRUE)
```
