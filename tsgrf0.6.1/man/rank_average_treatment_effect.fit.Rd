% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rank_average_treatment.R
\name{rank_average_treatment_effect.fit}
\alias{rank_average_treatment_effect.fit}
\title{Fitter function for Rank-Weighted Average Treatment Effect (RATE).}
\usage{
rank_average_treatment_effect.fit(
  DR.scores,
  priorities,
  target = c("AUTOC", "QINI"),
  q = seq(0.1, 1, by = 0.1),
  R = 200,
  sample.weights = NULL,
  clusters = NULL
)
}
\arguments{
\item{DR.scores}{A vector with the evaluation set scores.}

\item{priorities}{Treatment prioritization scores S(Xi) for the units in the evaluation set.
Two prioritization rules can be compared by supplying a two-column array or named list of priorities
(yielding paired standard errors that account for the correlation between RATE metrics estimated on
the same evaluation data).
WARNING: for valid statistical performance, these scores should be constructed independently from the evaluation
dataset used to construct DR.scores.}

\item{target}{The type of RATE estimate, options are "AUTOC" (exhibits greater power when only a small subset
of the population experience nontrivial heterogeneous treatment effects) or "QINI" (exhibits greater power
when the entire population experience diffuse or substantial heterogeneous treatment effects).
Default is "AUTOC".}

\item{q}{The grid q to compute the TOC curve on. Default is
(10\%, 20\%, ..., 100\%).}

\item{R}{Number of bootstrap replicates for SEs. Default is 200.}

\item{sample.weights}{Weights given to an observation in estimation.
If NULL, each observation is given the same weight. Default is NULL.}

\item{clusters}{Vector of integers or factors specifying which cluster each observation corresponds to.
Default is NULL (ignored).}
}
\value{
A list of class `rank_average_treatment_effect` with elements \itemize{
 \item estimate: the RATE estimate.
 \item std.err: bootstrapped standard error of RATE.
 \item target: the type of estimate.
 \item TOC: a data.frame with the Targeting Operator Characteristic curve
   estimated on grid q, along with bootstrapped SEs.
}
}
\description{
Provides an optional interface to \code{\link{rank_average_treatment_effect}} which allows for user-supplied
evaluation scores.
}
\examples{
\donttest{
# Train a causal forest to estimate a CATE based priority ranking
n <- 1500
p <- 5
X <- matrix(rnorm(n * p), n, p)
W <- rbinom(n, 1, 0.5)
event.prob <- 1 / (1 + exp(2*(pmax(2*X[, 1], 0) * W - X[, 2])))
Y <- rbinom(n, 1, event.prob)
train <- sample(1:n, n / 2)
cf.priority <- causal_forest(X[train, ], Y[train], W[train])

# Compute a prioritization based on estimated treatment effects.
# -1: in this example the treatment should reduce the risk of an event occuring.
priority.cate <- -1 * predict(cf.priority, X[-train, ])$predictions

# Train a separate CATE estimator for the evaluation set.
Y.forest.eval <- regression_forest(X[-train, ], Y[-train], num.trees = 500)
Y.hat.eval <- predict(Y.forest.eval)$predictions
W.forest.eval <- regression_forest(X[-train, ], W[-train], num.trees = 500)
W.hat.eval <- predict(W.forest.eval)$predictions
cf.eval <- causal_forest(X[-train, ], Y[-train], W[-train],
                         Y.hat = Y.hat.eval,
                         W.hat = W.hat.eval)

# Compute doubly robust scores corresponding to a binary treatment (AIPW).
tau.hat.eval <- predict(cf.eval)$predictions
debiasing.weights.eval <- (W[-train] - W.hat.eval) / (W.hat.eval * (1 - W.hat.eval))
Y.residual.eval <- Y[-train] - (Y.hat.eval + tau.hat.eval * (W[-train] - W.hat.eval))
DR.scores <- tau.hat.eval + debiasing.weights.eval * Y.residual.eval

# Could equivalently be obtained by
# DR.scores <- get_scores(cf.eval)

# Estimate AUTOC.
rate <- rank_average_treatment_effect.fit(DR.scores, priority.cate)

# Same as
# rank_average_treatment_effect(cf.eval, priority.cate)

# If the treatment randomization probabilities are known, then an alternative to
# evaluation via doubly robust scores is to use inverse-propensity weighting.
IPW.scores <- ifelse(W[-train] == 1, Y[-train]/0.5, -Y[-train]/0.5)
rate.ipw <- rank_average_treatment_effect.fit(IPW.scores, priority.cate)
}

}
