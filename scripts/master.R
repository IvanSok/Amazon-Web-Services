###############################################################################

pacman::p_load(doParallel, readr, rstudioapi, dplyr, plotly, corrplot, caret, rpart, rpart.plot, C50,
               randomForest, e1071, kknn)
detectCores()

###############################################################################
# Github setup ------------------------------------------------------------
current_path <- getActiveDocumentContext()$path

setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
#Cluster --------------
cl <- makeCluster(6) #creating a cluster
registerDoParallel(cl)
getDoParWorkers()

###############################################################################
#Import&Preprocessing:

iphone_matrix <- read_csv("datasets/iphone_smallmatrix_labeled_8d.csv")
#iphone_matrix <- distinct(iphone_matrix)
#str(iphone_matrix)
#summary(iphone_matrix)
#sum(is.na(iphone_matrix))

#plot_ly(iphone_matrix, x= ~iphone_matrix$iphonesentiment, type='histogram')


#M <- cor(iphone_matrix)
#options(max.print = 1000000)
#M
#corrplot(M, method = "circle") #samsunggalaxy and sonyxperia have negative correlations with iphonesentiment
iphone_matrix$iphonesentiment <- as.factor(iphone_matrix$iphonesentiment)


#Feature Variance:

#nzvMetrics <- nearZeroVar(iphone_matrix, saveMetrics = TRUE)
#nzvMetrics

#nzv <- nearZeroVar(iphone_matrix, saveMetrics = FALSE) #identifying index with newar zero var as a vector
#nzv

#iphone_matrix <- iphone_matrix[,-nzv] #removing nearzerovar variables
#str(iphone_matrix)


###############################################################################
#Feature engineering:
iphoneFE <- iphone_matrix
iphoneFE$iphonesentiment <- recode(iphoneFE$iphonesentiment, "0" = 1, "1" = 1,
                   "2" = 2, "3" = 3, "4" = 4, "5" = 4)
#summary(iphoneFE)
#str(iphoneFE)
iphoneFE$iphonesentiment <- as.factor(iphoneFE$iphonesentiment)

#plot_ly(iphoneFE, x= ~iphoneFE$iphonesentiment, type='histogram')


###############################################################################
###Creating Testing and Training Sets:
set.seed(123) #a starting point used to create a sequence of random numbers

trainSize<-round(nrow(iphoneFE)*0.7) #calculating the size of training set (70%)
testSize<-nrow(iphoneFE)-trainSize #calculating the size of testing set (30%)

training_indices<-sample(seq_len(nrow(iphoneFE)),size =trainSize) #creating training and testing datasets
trainSet<-iphoneFE[training_indices,]
testSet<-iphoneFE[-training_indices,] 

trainSet_down <- downSample(x = trainSet, y = trainSet$iphonesentiment) #downsampling the dataset
trainSet_down$Class <- NULL
testSet_down <- downSample(x = testSet, y = testSet$iphonesentiment)
testSet_down$Class <- NULL

trainSet_up <- upSample(x = trainSet, y = trainSet$iphonesentiment) #downsampling the dataset
trainSet_up$Class <- NULL
testSet_up <- upSample(x = testSet, y = testSet$iphonesentiment)
testSet_up$Class <- NULL


# Decision tree:
#dt <- rpart(iphonesentiment~., data = iphone_matrix, cp = .008)
#rpart.plot(dt, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
#varImp(dt)



#Random Forest: best model
set.seed(124)
rf_mod <- randomForest(iphonesentiment ~ ., data = trainSet, importance = TRUE,
                       ntree = 100, mtry = 9, method = "rf")
rf_mod
rf_predict <- predict(rf_mod, newdata = testSet)
rf_postres <- postResample(rf_predict, testSet$iphonesentiment)
rf_postres

rf_pred_all <- predict(rf_mod, newdata = iphoneFE)
rf_postres_all <- postResample(rf_pred_all, iphoneFE$iphonesentiment)
rf_postres_all

#downsampled dataset:
#rf_mod2 <- randomForest(iphonesentiment ~ ., data = trainSet_down, importance = TRUE,
#                       ntree = 100, mtry = 9, method = "rf")
#rf_mod2
#rf_predict2 <- predict(rf_mod2, newdata = testSet_down)
#rf_postres2 <- postResample(rf_predict2, testSet_down$iphonesentiment)
#rf_postres2

#upsampled dataset:
#rf_mod3 <- randomForest(iphonesentiment ~ ., data = trainSet_up, importance = TRUE,
#                        ntree = 100, mtry = 9, method = "rf")
#rf_mod3
#rf_predict3 <- predict(rf_mod3, newdata = testSet_up)
#rf_postres3 <- postResample(rf_predict3, testSet_up$iphonesentiment)
#rf_postres3



#SVM:
#svm_mod1 <- svm(iphonesentiment ~ ., data = trainSet)
#svm_mod1
#svm_mod_pred1 <- predict(svm_mod1, testSet)
#svm_mod_postres1 <- postResample(svm_mod_pred1, testSet$iphonesentiment)
#svm_mod_postres1

#downsampled:
#svm_mod2 <- svm(iphonesentiment ~ ., data = trainSet_down)
#svm_mod2
#svm_mod_pred2 <- predict(svm_mod2, testSet_down)
#svm_mod_postres2 <- postResample(svm_mod_pred2, testSet_down$iphonesentiment)
#svm_mod_postres2


#kknn:
#kknn_mod1 <- train.kknn(formula = iphonesentiment ~ ., data = trainSet, kmax = 9)
#kknn_mod1
#kknn_predict1 <- predict(kknn_mod1, testSet )
#kknn_postres1 <- postResample(kknn_predict1, testSet$iphonesentiment)
#kknn_postres1

#downsampled
#kknn_mod2 <- train.kknn(formula = iphonesentiment ~ ., data = trainSet_down, kmax = 9)
#kknn_mod2
#kknn_predict2 <- predict(kknn_mod2, testSet_down )
#kknn_postres2 <- postResample(kknn_predict2, testSet_down$iphonesentiment)
#kknn_postres2
###############################################################################
# Applying the model:
iphoneLargeMatrix <- read_csv("datasets/iphoneLargeMatrix.csv")
iphoneLargeMatrix <- distinct(iphoneLargeMatrix)
iphoneLargeMatrix$iphonesentiment <- as.factor(iphoneLargeMatrix$iphonesentiment)



rf_pred_Large <- predict(rf_mod, newdata = iphoneLargeMatrix)
rf_postres_Large <- postResample(rf_pred_Large, iphoneLargeMatrix$iphonesentiment)
rf_postres_Large

rf_pred_Large <- as.numeric(rf_pred_Large)
hist(rf_pred_Large)
###############################################################################
stopCluster(cl) #stopping the cluster
