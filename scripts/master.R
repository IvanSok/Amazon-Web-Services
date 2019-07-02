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
iphone_matrix$iphonesentiment <- as.factor(iphone_matrix$iphonesentiment)
iphone_matrix <- distinct(iphone_matrix)
str(iphone_matrix)
summary(iphone_matrix)
sum(is.na(iphone_matrix))

plot_ly(iphone_matrix, x= ~iphone_matrix$iphonesentiment, type='histogram')


M <- cor(iphone_matrix)
options(max.print = 1000000)
M
corrplot(M, method = "circle") #samsunggalaxy and sonyxperia have negative correlations with iphonesentiment

#iphone_matrix_v2 <- iphone_matrix #creating a new df
#iphone_matrix_v2$featuretoremove <- NULL #removing highly correlated variables

#Feature Variance:

nzvMetrics <- nearZeroVar(iphone_matrix, saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(iphone_matrix, saveMetrics = FALSE) #identifying index with newar zero var as a vector
nzv

iphoneNZV <- iphone_matrix[,-nzv] #removing nearzerovar variables
str(iphoneNZV)

#Recursive Feature Elimination:

set.seed(123)

iphoneSample <- iphone_matrix[sample(1:nrow(iphone_matrix), 1000, replace = FALSE),]

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)  #train control

ctrl2 <- trainControl(sampling = "down",
                      method = "repeatedcv", repeats = 5)

trainSet_down <- downSample(x = trainSet, y = trainSet$iphonesentiment) #downsampling the dataset

# Using rfe and omitting the response variable (attribute 59 iphonesentiment) 
#rfeResults <- rfe(iphoneSample[,1:58], 
#                  iphoneSample$iphonesentiment, 
#                  sizes=(1:58), 
#                  rfeControl=ctrl)

#rfeResults
#plot(rfeResults, type = c("g", "o"))

#iphoneRFE <- iphone_matrix[,predictors(rfeResults)] #newdf with rfe recommended features
#iphoneRFE$iphonesentiment <- iphone_matrix$iphonesentiment #adding the dependent variable to iphoneRFE
#str(iphoneRFE)




###############################################################################
###Creating Testing and Training Sets:
set.seed(123) #a starting point used to create a sequence of random numbers

trainSize<-round(nrow(iphone_matrix)*0.7) #calculating the size of training set (70%)
testSize<-nrow(iphone_matrix)-trainSize #calculating the size of testing set (30%)

training_indices<-sample(seq_len(nrow(iphone_matrix)),size =trainSize) #creating training and testing datasets
trainSet<-iphone_matrix[training_indices,]
testSet<-iphone_matrix[-training_indices,] 

# Decision tree:
dt <- rpart(iphonesentiment~., data = iphone_matrix, cp = .008)
rpart.plot(dt, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
varImp(dt)

#C50:
#c50_mod <- C5.0(x = trainSet, y = trainSet$iphonesentiment, control = ctrl2)

#Random Forest:
set.seed(124)
rf_mod <- randomForest(iphonesentiment ~ ., data = trainSet, importance = TRUE,
                       ntree = 100, mtry = 9)
rf_mod

#SVM:
svm_mod <- svm(iphonesentiment ~ ., data = trainSet)
summary(svm_mod)

#kknn:
kknn_mod <- kknn(iphonesentiment ~ ., train = trainSet, test = testSet )
summary(kknn_mod)
###############################################################################
stopCluster(cl) #stopping the cluster
