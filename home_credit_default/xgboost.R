# Libraries
library(xgboost)
library(caret)

################ DATA PREPARATION ############

train_df <- read.csv('application_train_selected_feats.csv')
train_df <- train_df[, !names(train_df) %in% c('SK_ID_CURR')]

set.seed(100)

## 70% of the sample size
smp_size <- floor(0.70 * nrow(train_df))

## set the seed to make your partition reproducible
train_ind <- sample(seq_len(nrow(train_df)), size = smp_size)

X_train <- train_df[train_ind, ]
X_test <- train_df[-train_ind, ]

train_df <- X_train[, !(names(X_train) %in% c('TARGET'))]
xgmat <- xgb.DMatrix(as.matrix(sapply(train_df, as.numeric)), label = X_train$TARGET)

test_df <- X_test[, !(names(X_test) %in% c('TARGET'))]
xgmat_test <- xgb.DMatrix(as.matrix(sapply(test_df, as.numeric)), label = X_test$TARGET)

tgt_count <- table(X_train$TARGET)

########### MODEL FITTING #################
set.seed(8033)

param <- list("objective" = "binary:logistic",
              "scale_pos_weight" = tgt_count[names(tgt_count) == 0] / tgt_count[names(tgt_count) == 1],
              "bst:eta" = 0.05,
              "bst:max_depth" = 5,
              "subsample" = 0.8,
              "min_child_weight" = 8,
              "colsample_bytree" = 0.8,
              "eval_metric" = "auc",
              "objective" = "binary:logistic",
              "nthread" = 2,
              "lambda" = 1000)

watchlist = list('train' = xgmat, 'val' = xgmat_test)
nround = 1200
bst <- xgb.train(param, xgmat, nround, watchlist)


##### TRAIN PREDICTIONS
train_pred_probs <- predict(bst, as.matrix(sapply(train_df, as.numeric)))

require(ROCR)
train_pred <- prediction(train_pred_probs, X_train$TARGET)
perf <- performance(train_pred, measure = 'auc')
perf@y.values

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)

# Set our cutoff threshold
train_pred <- ifelse(train_pred_probs > 0.5, 1, 0)

library(MLmetrics)
F1_Score(X_train$severity_2017, train_pred)

# Create the confusion matrix
confusionMatrix(train_pred, X_train$severity_2017, positive="1")


### TEST PREDICTION
test_pred_probs <- predict(bst, xgmat_test)


# Set our cutoff threshold
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)

F1_Score(X_test$severity_2017, test_pred)

# Create the confusion matrix
confusionMatrix(test_pred, X_test$severity_2017, positive="1")

# ROC Curve    

test_pred <- prediction(test_pred_probs, X_test$TARGET)
perf <- performance(test_pred, measure = 'auc')
perf@y.values

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)

##### FEATURE IMPORTANCE ######
importance_matrix <- xgb.importance(colnames(train_df), model = bst)

par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,2)) # increase y-axis margin.
barplot(importance_matrix$Gain, main="Information Gain (XGBOOST)", horiz=TRUE, cex.names=0.8)


################# SUBMISSION ##############

submission_df <- read.csv('application_test_selected_feats.csv')
xgmat_submission <- xgb.DMatrix(as.matrix(sapply(submission_df[, !(names(submission_df) %in% c('SK_ID_CURR'))], as.numeric)))
submission_pred_probs <- predict(bst, xgmat_submission)

submission_results <- data.frame('SK_ID_CURR' = submission_df$SK_ID_CURR, 'TARGET' = submission_pred_probs)
write.csv(submission_results, 'RESULTS/xgboost_submission.csv')
