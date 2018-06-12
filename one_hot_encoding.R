# encoding categorical variables OR how to correctly using xgboost in R
# one hot encoding is everywhere in R

#--------------evidence from linear model
# one hot encoding in the variabla names chosen by a linear regression
dTrain <- data.frame(x = c('a', 'b', 'b', 'c'),
                     y = c(1, 2, 1, 2))
summary(lm(y ~ x, data = dTrain))
# most of the encoding in R is essentially based on "contrasts" implemented in stats::model.matrix()
# example:
data.matrix(dTrain)

# BUT When directly applying stats::model.matrix() 
# you can not safely assume the same formula applied to 
# two different data sets (say train and application or test) are using the same encoding!
# Example:
dTrain <- data.frame(x = c('a', 'b', 'c'), stringsAsFactors = F)
encTrain <- stats::model.matrix(~ x, dTrain)
print(encTrain)

dTest <- data.frame(x = c('b', 'c'), stringsAsFactors = F)
encTest <- stats::model.matrix(~ x, dTest)
print(encTest)

# a critical fraw when building a model then using it on new data
# encoding is hidden in model training, and how to encode new data is stored as part of the model
# xgboost requires data already to be encoded as a numeric matrix
# namely, xgboost requires a numeric matrix as its input
# in other words, we must manage the encoding plan ourselves
# so we should store the encoding plan somewhere(explicitly)
# R often hides its encoding plan in the trained model

# examples: using xgboost

# packageurl <- "http://cran.r-project.org/src/contrib/Archive/titanic/titanic_0.1.0.tar.gz"
# install.packages(packageurl, contriburl=NULL, type="source")
library(titanic)
data(titanic_train)
str(titanic_train)
summary(titanic_train)
outcome <- "Survived"
target <- 1
shouldBeCategorical <- c("PassengerId", "Pclass", "Parch")
for (v in shouldBeCategorical) {
  titanic_train[[v]] <- as.factor(titanic_train[[v]])
}
tooDetailed <- c("Ticket", "Cabin", "Name", "PassengerId")
vars <- setdiff(colnames(titanic_train), c(outcome,tooDetailed))
dTrain <- titanic_train

# design cross validation modeling experiment
library(xgboost)
library(sigr)
library(WVPlots)
library(vtreat)
set.seed(2333)
crossValPlan <- vtreat::kWayStratifiedY(nrow(dTrain),
                                        10,
                                        dTrain,
                                        dTrain[[outcome]])

evaluateModelingProcedure <- function(xMatrix, outcomeV, crossValPlan) {
  preds <- rep(NA_real_, nrow(xMatrix))
  for (ci in crossValPlan) {
    nrounds <- 1000
    cv <- xgb.cv(data = xMatrix[ci$train, ],
                 label = outcomeV[ci$train],
                 objective = "binary:logistic",
                 nrounds = nrounds,
                 verbose = 0,
                 nfold = 5)
    # nrounds <- which.min(cv$evaluation_log$test_rmse_mean) # regression
    nrounds <- which.min(cv$evaluation_log$test_error_mean) # classification
    model <- xgboost(data = xMatrix[ci$train, ],
                     label = outcomeV[ci$train],
                     objective = "binary:logistic",
                     nrounds = nrounds,
                     verbose = 0)
    preds[ci$app] <- predict(model, xMatrix[ci$app, ])
  }
  preds
}

set.seed(2333)
tplan <- vtreat::designTreatmentsZ(dTrain,
                                   vars,
                                   minFraction = 0,
                                   verbose = 0)

# restrict to common variable types
sf <- tplan$scoreFrame
newvars <- sf$valName[sf$code %in% c("lev", "clean", "isBAD")]
trainVtreat <- as.matrix(vtreat::prepare(tplan, dTrain, varRestriction = newvars))
print(dim(trainVtreat))

print(colnames(trainVtreat))

dTrain$predVtreatZ <- evaluateModelingProcedure(trainVtreat,
                                                dTrain[[outcome]] == target,
                                                crossValPlan)

sigr::permTestAUC(dTrain,
                  "predVtreatZ",
                  outcome,
                  target)

WVPlots::ROCPlot(dTrain,
                "predVtreatZ",
                outcome,
                target,
                "vtreat encoder performance")

set.seed(2333)
f <- paste("~ 0 +", paste(vars, collapse = " + "))
# model matrix skip rows with NAs by default
# get control of this through an option
oldOpt <- getOption("na.action")
options(na.action = "na.pass")
trainModelMatrix <- stats::model.matrix(as.formula(f),
                                        dTrain)

# note that model.matrix does not conveniently store the encoding plan
# so you may run into difficulty if you were to encode new data which didn't 
# have all the levels seen in the training data
options(na.action = oldOpt)
print(dim(trainModelMatrix))
print(colnames(trainModelMatrix))

dTrain$predModelMatrix <- evaluateModelingProcedure(trainModelMatrix,
                                                    dTrain[[outcome]] == target,
                                                    crossValPlan)
sigr::permTestAUC(dTrain, 
                  "predModelMatrix", 
                  outcome, 
                  target)

WVPlots::ROCPlot(dTrain,
                 "predModelMatrix",
                 outcome,
                 target,
                 "model.matrix encoder performance")

# using the encoder provided by package "caret"
# the encoding fuctionality can help properly split between training and application
# namely, caret::dummyVars() and predict()

library(caret)
set.seed(2333)
f <- paste("~", paste(vars, collapse = " + "))
encoder <- caret::dummyVars(as.formula(f), dTrain)
trainCaret <- predict(encoder, dTrain)
print(dim(trainCaret))
print(colnames(trainCaret))

dTrain$predCaret <- evaluateModelingProcedure(trainCaret,
                                              dTrain[[outcome]] == target,
                                              crossValPlan)
sigr::permTestAUC(dTrain,
                  "predCaret",
                  outcome,
                  target)

WVPlots::ROCPlot(dTrain,
                 "predCaret",
                 outcome,
                 target,
                 "caret encoder performance")

# CONCLUSION:
# vtreat::designTreatmentsZ has a number of useful properties:
# dose not look at the outcome values, so does not require extra care in cross-validation
# save its encoding, so can be used correctly on new data
# above 2 properties shared with caret::dummyVars()
# vtreat is designed "to always work" (always return a pure numeric data frame with no missing values)

# strength of this method:
# No NA values are passed through by vtreat::prepare().
# NA presence is added as an additional informative column.
# A few derived columns (such as pooling of rare levels are made available).
# Rare dummy variables are pruned (under a user-controlled threshold) 
#     to prevent encoding explosion.
# Novel levels (levels that occur during test or application, but not during training) 
#     are deliberately passed through as "no training level activated" by 
#     vtreat::prepare() (caret::dummyVars() considers this an error).










