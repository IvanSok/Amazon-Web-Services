###############################################################################

pacman::p_load(doParallel, readr, rstudioapi, dplyr, plotly, corrplot, caret)
detectCores()

###############################################################################
# Github setup ------------------------------------------------------------
current_path <- getActiveDocumentContext()$path

setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
#Cluster --------------
cl <- makeCluster(3) #creating a cluster
registerDoParallel(cl)
getDoParWorkers()

###############################################################################
#Import&Preprocessing:

iphone_matrix <- read_csv("datasets/iphone_smallmatrix_labeled_8d.csv")

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

# Using rfe and omitting the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

rfeResults
plot(rfeResults, type = c("g", "o"))

###############################################################################

stopCluster(cl) #stopping the cluster
