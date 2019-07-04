###############################################################################
#Libraries/Cores:--------------
pacman::p_load(doParallel, readr, rstudioapi, dplyr, plotly, corrplot, caret,
               rpart, rpart.plot, C50, randomForest, e1071, kknn)
detectCores()

###############################################################################
# Github setup --------------
current_path <- getActiveDocumentContext()$path

setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
#Cluster --------------
cl <- makeCluster(6) #creating a cluster
registerDoParallel(cl)
getDoParWorkers()

###############################################################################
#Import&Preprocessing:--------------

iphone_matrix <- read_csv("datasets/iphone_smallmatrix_labeled_8d.csv")
galaxy_matrix <- read_csv("datasets/galaxy_smallmatrix_labeled_9d.csv")

iphone_matrix$iphonesentiment <- as.factor(iphone_matrix$iphonesentiment)
galaxy_matrix$galaxysentiment <- as.factor(galaxy_matrix$galaxysentiment)

###############################################################################
#Feature engineering:--------------
iphoneFE <- iphone_matrix
#iphoneFE$iphonesentiment <- recode(iphoneFE$iphonesentiment, "0" = 1, "1" = 1,
#                   "2" = 2, "3" = 3, "4" = 4, "5" = 4)

galaxyFE <- galaxy_matrix
#galaxyFE$galaxysentiment <- recode(galaxyFE$galaxysentiment, "0" = 1, "1" = 1,
#                                   "2" = 2, "3" = 3, "4" = 4, "5" = 4)


#summary(iphoneFE)
#str(iphoneFE)
iphoneFE$iphonesentiment <- as.factor(iphoneFE$iphonesentiment)
galaxyFE$galaxysentiment <- as.factor(galaxyFE$galaxysentiment)

#plot_ly(iphoneFE, x= ~iphoneFE$iphonesentiment, type='histogram')
#plot_ly(galaxyFE, x= ~galaxyFE$galaxysentiment, type='histogram')


###############################################################################
#Train/Test sets: --------------
###Creating Testing and Training Sets: (iphone)
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


###Creating Testing and Training Sets: (galaxy)
set.seed(321) #a starting point used to create a sequence of random numbers

trainSize_g<-round(nrow(galaxyFE)*0.7) #calculating the size of training set (70%)
testSize_g<-nrow(galaxyFE)-trainSize_g #calculating the size of testing set (30%)

training_indices_g<-sample(seq_len(nrow(galaxyFE)),size =trainSize_g) #creating training and testing datasets
trainSet_g<-galaxyFE[training_indices_g,]
testSet_g<-galaxyFE[-training_indices_g,] 
###############################################################################
#PCA reduction:--------------
#PCA:(iphone)
preprocessParams <- preProcess(trainSet[,-59], method = c("center", "scale", "pca"),
                               thresh = 0.95)
print(preprocessParams)

train.pca <- predict(preprocessParams, trainSet[,-59])
train.pca$iphonesentiment <- trainSet$iphonesentiment

test.pca <- predict(preprocessParams, testSet[,-59])
test.pca$iphonesentiment <- testSet$iphonesentiment

str(train.pca)
str(test.pca)

#PCA:(galaxy)
preprocessParams_g <- preProcess(trainSet_g[,-59],
                                 method = c("center", "scale", "pca"),
                                 thresh = 0.95)
print(preprocessParams_g)

train.pca_g <- predict(preprocessParams_g, trainSet_g[,-59])
train.pca_g$galaxysentiment <- trainSet_g$galaxysentiment

test.pca_g <- predict(preprocessParams_g, testSet_g[,-59])
test.pca_g$galaxysentiment <- testSet_g$galaxysentiment

str(train.pca_g)
str(test.pca_g)

#PCA on the whole iphone matrix:(iphone)
preprocessParams_all <- preProcess(iphoneFE[,-59], 
                                   method = c("center", "scale", "pca"),
                                   thresh = 0.95)

iphoneFE.pca <- predict(preprocessParams_all, iphoneFE[,-59])
iphoneFE.pca$iphonesentiment <- iphoneFE$iphonesentiment
str(iphoneFE.pca)

#PCA on the whole iphone matrix:(galaxy)
preprocessParams_all_g <- preProcess(galaxyFE[,-59],
                                   method = c("center", "scale", "pca"),
                                   thresh = 0.95)

galaxyFE.pca <- predict(preprocessParams_all_g, galaxyFE[,-59])
galaxyFE.pca$galaxysentiment <- galaxyFE$galaxysentiment
str(galaxyFE.pca)
###############################################################################
#Modelling: --------------
#Random Forest: (iphone)
set.seed(432)
rf_mod <- randomForest(iphonesentiment ~ ., data = train.pca, importance = TRUE,
                       ntree = 100, mtry = 9, method = "rf")
rf_mod
rf_predict <- predict(rf_mod, newdata = test.pca)
rf_postres <- postResample(rf_predict, test.pca$iphonesentiment)
rf_postres

#Random Forest: (galaxy)
set.seed(432)
rf_mod_g <- randomForest(galaxysentiment ~ ., data = train.pca_g, importance = TRUE,
                       ntree = 100, mtry = 9, method = "rf")
rf_mod_g
rf_predict_g <- predict(rf_mod_g, newdata = test.pca_g)
rf_postres_g <- postResample(rf_predict_g, test.pca_g$galaxysentiment)
rf_postres_g

###############################################################################
# Applying the model:--------------

LargeMatrix_i <- read_csv("datasets/allfactors.csv")
LargeMatrix_g <- read_csv("datasets/allfactors.csv")
LargeMatrix_i[,c("X1", "id")] <- NULL
LargeMatrix_g[,c("X1", "id")] <- NULL
LargeMatrix_i$iphonesentiment <- NA
LargeMatrix_i$iphonesentiment <- as.factor(LargeMatrix_i$iphonesentiment)
LargeMatrix_g$galaxysentiment <- NA
LargeMatrix_g$galaxysentiment <- as.factor(LargeMatrix_g$galaxysentiment)

#PCA on the whole Large matrix:

LargeMatrix.pca_i <- predict(preprocessParams_all, LargeMatrix_i[,-59])
LargeMatrix.pca_g <- predict(preprocessParams_all_g, LargeMatrix_g[,-59])

LargeMatrix.pca_i$iphonesentiment <- LargeMatrix_i$iphonesentiment
LargeMatrix.pca_g$galaxysentiment <- LargeMatrix_g$galaxysentiment

str(LargeMatrix.pca_i)
str(LargeMatrix.pca_g)

rf_pred_Large_i <- predict(rf_mod, newdata = LargeMatrix.pca_i)
rf_pred_Large_i

rf_pred_Large_g <- predict(rf_mod_g, newdata = LargeMatrix.pca_g)
rf_pred_Large_g

LargeMatrix_i$iphonesentiment <- rf_pred_Large_i
LargeMatrix_g$galaxysentiment <- rf_pred_Large_g


summary(LargeMatrix_i$iphonesentiment)
summary(LargeMatrix_g$galaxysentiment)

write_csv(LargeMatrix_i, "LargeMatrixPredicted_iphone.csv")
write_csv(LargeMatrix_g, "LargeMatrixPredicted_galaxy.csv")

LargeMatrixPredicted_iphone <- read_csv("LargeMatrixPredicted_iphone.csv")
LargeMatrixPredicted_galaxy <- read_csv("LargeMatrixPredicted_galaxy.csv")

LargeMatrixPredicted_iphone$iphonesentiment <- as.numeric(LargeMatrixPredicted_iphone$iphonesentiment)
LargeMatrixPredicted_galaxy$galaxysentiment <- as.numeric(LargeMatrixPredicted_galaxy$galaxysentiment)

iphone_sentiment_level <- (sum(LargeMatrixPredicted_iphone$iphonesentiment))/nrow(LargeMatrixPredicted_iphone)
iphone_sentiment_level

galaxy_sentiment_level <- (sum(LargeMatrixPredicted_galaxy$galaxysentiment))/nrow(LargeMatrixPredicted_galaxy)
galaxy_sentiment_level

###############################################################################
stopCluster(cl) #stopping the cluster
