library(caret)
library(neuralnet)
load("~/occupancy.RData")

set.seed(123)                                     # Train Test split 60/40
trainRowNumbers <- createDataPartition(data$Occupancy, p=0.6, list=FALSE)
trainData <- data[trainRowNumbers,]
testData <- data[-trainRowNumbers,]

max_data_train <- apply(trainData[,1:5], 2, max)  # Normalize train data
min_data_train <- apply(trainData[,1:5], 2, min)
data_scaled_train <- scale(trainData[,1:5],
                           center = min_data_train, 
                           scale = max_data_train - min_data_train)
Occupancy = as.numeric(trainData$Occupancy)-1
train<-as.data.frame(cbind(data_scaled, Occupancy))

max_data_test <- apply(testData[,1:5], 2, max)    # Normalize test data
min_data_test <- apply(testData[,1:5], 2, min)
data_scaled_test <- scale(testData[,1:5],
                          center = min_data_test, 
                          scale = max_data_test - min_data_test)
Occupancy = as.numeric(testData$Occupancy)-1
test<-as.data.frame(cbind(data_scaled_test,Occupancy))

deep_net = neuralnet(Occupancy~.,data = train, # Train neural network
                     hidden = 4,               # K=4
                     linear.output=F, 
                     lifesign = "full",        # print during calculation
                     err.fct = "ce",           # Cross entrophy error function
                     act.fct = "logistic",     # Sigmoid activation function
                     stepmax= 500000)          # Increase steps for it to converge 
plot(deep_net, rep="best", show.weights = F, information=F) # Plot

predicted_data <- neuralnet::compute(deep_net,test) #Predictions test data
predicted_data$net.result <- sapply(predicted_data$net.result,round,digits=0)

confusionMatrix(as.factor(predicted_data$net.result),
                as.factor(test$Occupancy), positive = "1")

predicted_data_train <- neuralnet::compute(deep_net,train)#Predictions train data
predicted_data_train$net.result <- sapply(predicted_data_train$net.result,round,digits=0)

confusionMatrix(as.factor(predicted_data_train$net.result),
                as.factor(train$Occupancy), positive = "1")