library(xgboost)
library(tidyverse)
library(pROC)


pathname = "/Users/dylanlee/GitHub/stat441-W20-project/"

train <- read.csv(paste0(pathname, "03-data-creation/train_FE_final.csv"), header = TRUE)
test <- read.csv(paste0(pathname, "03-data-creation/test_FE_final.csv"), header = TRUE)

train_ids <- train[,"id"]
train_satisfied <- train[,"satisfied"]
test_ids <- test[,"id"]
train <- train %>% select(-"satisfied", -"X", -"id")
test <- test %>% select (-"X", -"id")

train_matrix <- data.matrix(train)
test_matrix <- data.matrix(test)

# 70/30 training test split on training set to measure performance before applying to actual test set
set.seed(224)
sample_size <- floor(nrow(train_matrix)*.7)
train_ind <- sample(seq_len(nrow(train_matrix)), size = sample_size)
training_data <- train_matrix[train_ind, ]
training_satisfied <- train_satisfied[train_ind]
testing_data <- train_matrix[-train_ind,]
testing_satisfied <- train_satisfied[-train_ind]

dtrain <- xgb.DMatrix(data = training_data, label = training_satisfied)
dtest <- xgb.DMatrix(data = testing_data, label = testing_satisfied)

cv <- xgb.cv(data = dtrain, nrounds = 100, nthread = 2, nfold = 5, metrics = list("auc"),
             max_depth = 5, eta = 1, objective = "binary:logistic", early_stopping_rounds = 20)
print(cv)
# best iteration = 3

# print(cv, verbose=TRUE)

# can tune parameters here 

model <- xgb.train(data = dtrain, nrounds = cv$best_iteration, objective = "binary:logistic", max.depth = 5)

pred <- predict(model, dtest)
err <- mean(as.numeric(pred > 0.5) != testing_satisfied)
print(paste("test-error =", err))
roc_obj <- roc(testing_satisfied, pred)
auc(roc_obj)

importance_matrix <- xgb.importance(names(train_matrix), model = model)
xgb.plot.importance(importance_matrix)


important_features <- importance_matrix[importance_matrix$Gain>=0.001,]$Feature
training_data_reduced <- training_data[,important_features]
testing_data_reduced <- testing_data[,important_features]
dtrain_reduced <- xgb.DMatrix(data = training_data_reduced, label = training_satisfied)
dtest_reduced <- xgb.DMatrix(data = testing_data_reduced, label = testing_satisfied)

model_reduced <- xgb.train(data = dtrain_reduced, nrounds = cv$best_iteration, objective = "binary:logistic", max.depth = 5)

pred_reduced <- predict(model_reduced, dtest_reduced)
err <- mean(as.numeric(pred_reduced > 0.5) != testing_satisfied)
print(paste("test-error =", err))
roc_obj <- roc(testing_satisfied, pred_reduced)
auc(roc_obj)

# reduced model performs better

dtest_submit <- xgb.DMatrix(data = test_matrix[,important_features])
pred_submit <- predict(model_reduced, dtest_submit)
# pred_submit <- as.numeric(pred_submit>0.5)

predictions <- data.frame(id = test_ids, Predicted = pred_submit)
write.csv(predictions,paste0(pathname,"/04-modeling/stacking_predictions/fe_data_Final_xgboost_with_CV_no_rounding.csv"), row.names = FALSE)

# Score : 0.79661