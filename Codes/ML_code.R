#install and load the below required libraries
library(dplyr)
library(caret)
library(mgcv)
library(randomForest)
library(xgboost)
library(caret)
library(MLmetrics) 
library(Metrics)
library(raster) 
library(tidyverse) 
#################################################################################################
#Sample input data 
#load the sample data
setwd("D:/PHD data/Github projects/ML_models/sample_data")
sample <- read.csv("sample.csv")
sample2 <- sample[,c(1,2:8)] #load the response and predictor variables
x <- as.matrix(sample2[,c(2:8)]) # convert the dataframe to a matrix
xs <- scale(x,center = TRUE, scale = TRUE) #scale/standardize the data using 'scale' function
y <- sample2$Percent #load the response variable into a variable named 'y'
dataset <- data.frame(cbind(y,xs))#combine the scaled x and y variables
summary(dataset)#view the summary of the dataset

################################################################################
# Splitting training and testing set (80/20)
set.seed(30000)
idx <- caret::createDataPartition(dataset$y, p = 0.8, list = FALSE)
train <- dataset[idx, ]
test  <- dataset[-idx, ]
###############################################################################################
# Random Forest Regression 

set.seed(30000)
#use the tuneRF function from randomForest library to identify the best subset of variables under each tree
mtry <- randomForest::tuneRF(train[,2:8],train$y ,  mtryStart = 2,ntreeTry=2000, stepFactor=2, improve=0.05,
                             trace=TRUE, plot=TRUE, doBest=FALSE)
best.m <- mtry[mtry[, 2] == min(mtry[,2]), 1]#load the best subset variable number into the variable 'best.m'
print(mtry)
print(best.m) 
#############################################################################################
#specify the repeated cross validation with number of folds and repeats and hyper parameter optimization technique
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 5,
                        search = 'grid')
#create tunegrid for the RF model
tunegrid <- expand.grid(.mtry = best.m)
modellist <- list()

#train with different ntree parameters, mtry and CV method and save the model into the variable named 'fit'
for (ntree in c(1000,1500,2000,2500,3000)){
  set.seed(1000)
  fit <- train(y ~ .,
               data = train,
               method = 'rf',
               metric = 'RMSE',
               tuneGrid = tunegrid,
               trControl = control,
               ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}


#load the CV values and results
results <- resamples(modellist)
summary(results) #view summary of fit statistics for different ntree values
##############################################################################################
# use the fitted model to predict for the test data
predicted <- predict(fit,test)
observed<-test[,1]
#load the following libraries to calculate the fit statistics

#RMSE
sqrt(mean((predicted - observed)^2))
#correlation coefficient
(cor(predicted,observed))^2 
#Correlation
cor(predicted,observed)
#MAPE
MAPE(predicted, observed)
#MAE
caret::MAE(observed, predicted)

##################################################################################################
# use the below code to develop the final map using the predictor variable rasters
#rasters not uploaded due to storage limitations
setwd("D:/PHD data/Github projects/ML_models/example_predictors")

img<-list.files(pattern='*.tif$')#load the tiff files

rstack <- raster::stack(img)
# Rename layers to match your CSV / Excel predictor names
names(rstack) <- c("Dist_roads","Elevation","MAP","MAPcv","MAT",
                   "NDMI","SnowDepth")

# Now reorder to your desired order
ordered_names <- c("NDMI","Dist_roads","Elevation","SnowDepth",
                   "MAT","MAP","MAPcv")
# Check for mismatches before subsetting
setdiff(ordered_names, names(rstack))   # should return character(0)

# Reorder
stack <- rstack[[ordered_names]]

names(stack)  # verify final order
#############################################################################
# retrieve the names of the raster layers in the raster stack object xs that have been scaled and centered.
rasterNames = names(attr(x = xs,which = "scaled:center"))
# retrieve the center values used for scaling and centering each raster layer in the stack xs.
center = attr(x = xs,which = "scaled:center")
#retrieve the scaling factors used for scaling each raster layer in the stack xs.
scales =  attr(x = xs,which = "scaled:scale") 

#The following loop takes each raster layer (stack[[rasterNames[i]]]), subtracts corresponding center value (center[i]), and then divides by corresponding scaling factor (scales[i]). 
# Finally, the updated raster layer is assigned back to the same position in the stack object.
for(i in 1:7){
  stack[[rasterNames[i]]] = (stack[[rasterNames[i]]]-center[i])/scales[i]
  
}

setwd("D:/PHD data/Github projects/Output")
#apply the developed model on the stack to predict the final raster layer
predict(stack,fit,filename="predictedcover_RF_map.tif",na.rm=TRUE, progress="text",format="GTiff", overwrite=TRUE)
#################################################################################################
#GAM 

# y in [0,100] -> p in (0,1)
n <- nrow(dataset)
p_raw <- dataset$y / 100
dataset$p <- (p_raw * (n - 1) + 0.5) / n   # Smithson & Verkuilen style squeeze

set.seed(30000)
idx <- caret::createDataPartition(dataset$y, p = 0.8, list = FALSE)
train <- dataset[idx, ]
test  <- dataset[-idx, ]
#################################################################
forms <- list(
  m1 = p ~ s(NDMI, k=10) + s(Dist_roads, k=10) + Elevation + SnowDepth + MAT + MAP + MAPcv,
  m2 = p ~ s(NDMI, k=15) + s(Dist_roads, k=15) + Elevation + SnowDepth + MAT + MAP + MAPcv,
  m3 = p ~ s(NDMI, k=10) + s(Dist_roads, k=10) + s(Elevation, k=10) + SnowDepth + MAT + MAP + MAPcv,
  m4 = p ~ NDMI + Dist_roads + Elevation + SnowDepth + MAT + MAP + MAPcv,
  m5 = p ~ s(NDMI, k=10) + s(Dist_roads, k=10) + s(Elevation, k= 10) + s(SnowDepth, k = 10) + s(MAT, k= 10)  + s(MAP, k = 10) + s(MAPcv, k = 10) 
)

score_model <- function(form) {
  cv <- cv_gam_beta(train, form, repeats = 10, folds = 10)
  
  cv %>%
    summarise(
      RMSE = mean(RMSE),
      MAE  = mean(MAE),
      R2   = mean(R2)
    ) %>%
    mutate(
      model = paste(deparse(form), collapse = " ")
    )
}

scores <- dplyr::bind_rows(lapply(forms, score_model))
scores <- scores %>% arrange(RMSE)
scores

best_form <- forms[[ which.min(scores$RMSE) ]]

gam_beta_final <- gam(
  best_form,
  data   = train,
  family = betar(link = "logit"),
  method = "REML"
)
##################################################################
# use the fitted model to predict for the test data
predicted <- predict(gam_beta_final,test)
observed <-test[,1]
#load the following libraries to calculate the gam_beta_final statistics

#RMSE
sqrt(mean((predicted - observed)^2))
#correlation coefficient
(cor(predicted,observed))^2 
#Correlation
cor(predicted,observed)
#MAPE
MAPE(predicted, observed)
#MAE
caret::MAE(observed, predicted)
######################################################################
#rasters not uploaded due to storage limitations
#apply the developed model on the stack to predict the final raster layer
setwd("D:/PHD data/Github projects/Output")
pred <- raster::predict(
  stack,
  gam_beta_final,
  fun      = predict,
  type     = "response",     # IMPORTANT for beta GAM: returns p in (0,1)
  na.rm    = TRUE,
  progress = "text",
  filename = "predictedcover_GAM_map.tif",
  format   = "GTiff",
  overwrite= TRUE
)

# Convert to 0–100 cover and save
pred_pct <- pred_p * 100
writeRaster(pred_pct, "predictedcover_GAM_map_pct.tif",
            format = "GTiff", overwrite = TRUE)  

##################################################################
#Cubist model

set.seed(123)

# predictors used in your GAM
xvars <- c("NDMI","Dist_roads","Elevation","SnowDepth","MAT","MAP","MAPcv")

# Make sure response is numeric (0–100 is fine for Cubist)
train <- train %>%
  dplyr::select(all_of(c("y", xvars))) %>%
  na.omit()

ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5,
  savePredictions = "final"
)

# Cubist tuning grid:
# committees = number of model committees (like boosting)
# neighbors  = number of nearest neighbors used to adjust predictions
grid <- expand.grid(
  committees = c(1, 5, 10, 20, 50),
  neighbors  = c(0, 3, 5, 9)
)

cubist_cv <- caret::train(
  x = train[, xvars],
  y = train$y,
  method = "cubist",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "RMSE"
)

cubist_cv
best_params <- cubist_cv$bestTune
best_params
#########################################################################

cubist_final <- Cubist::cubist(
  x = train[, xvars],
  y = train$y,
  committees = best_params$committees,
  neighbors  = best_params$neighbors
)
#########################################################################

# use the fitted model to predict for the test data
predicted <- predict(cubist_final,test)
observed <-test[,1]
#load the following libraries to calculate the gam_beta_final statistics
#RMSE
sqrt(mean((predicted - observed)^2))
#correlation coefficient
(cor(predicted,observed))^2 
#Correlation
cor(predicted,observed)
#MAPE
MAPE(predicted, observed)
#MAE
caret::MAE(observed, predicted)
######################################################################

# Ensure layer names match predictors EXACTLY
# names(stack) <- xvars

pred_fun_cubist <- function(model, data) {
  as.numeric(predict(model, newdata = data))
}

pred_y <- raster::predict(
  stack,
  cubist_final,
  fun       = pred_fun_cubist,
  na.rm     = TRUE,
  progress  = "text",
  filename  = "predictedcover_Cubist_map.tif",
  format    = "GTiff",
  overwrite = TRUE
)

# Optional but recommended: clamp predictions to 0–100
pred_y_clamped <- raster::clamp(pred_y, lower = 0, upper = 100, useValues = TRUE)
setwd("D:/PHD data/Github projects/Output")
writeRaster(pred_y_clamped, "predictedcover_Cubist_map_clamped.tif",
            format="GTiff", overwrite=TRUE)
#################################################################################
#XGBoost model 
#changing format of input

x_train <- as.matrix(data.frame(lapply(train[, 2:8], as.numeric)))
storage.mode(x_train) <- "double"
y_train <- as.numeric(train$y)

x_test <- as.matrix(data.frame(lapply(test[, 2:8], as.numeric)))
storage.mode(x_test) <- "double"
y_test <- as.numeric(test$y)
######################################################

xgbGrid <- expand.grid(nrounds = c(100,200,500,1000), # this is n_estimators in the python code above 
                       max_depth = c(10, 15, 20, 25), 
                       colsample_bytree = seq(0.5, 0.9, length.out = 5), 
                       ## The values below are default values 
                       eta = 0.1, 
                       gamma=0, 
                       min_child_weight = c(1,25), 
                       subsample = c(0.25,1))

##################################################

xgb_trcontrol <- trainControl(
  method = "cv",
  number = 2,
  allowParallel = TRUE,
  verboseIter = FALSE
)
##################################################
set.seed(123)
xgb_model <- train(
  x = x_train,
  y = y_train,
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  metric = "RMSE"
)
########################################################################################
xgb_model$bestTune
xgb_model$results$Rsquared
xgb_model$results$RMSE
xgb_model$results$MAE
########################################################################################
predicted_xgb = predict(xgb_model, X_test)
residuals = y_test - predicted_xgb
RMSE = sqrt(mean(residuals^2))

predicted_xgb = predict(xgb_model, X_test)
#RMSE
sqrt(mean((predicted_xgb - observed)^2))
#correlation coefficient
(cor(predicted_xgb,observed))^2 
#Correlation
cor(predicted_xgb,observed)
#MAPE
MAPE(predicted_xgb, y_test)
#MAE
mae(y_test,predicted_xgb)
caret::MAE(y_test, predicted_xgb) 

########################################################################################
setwd("D:/PHD data/Github projects/Output")
xgb_reg <- predict(stack,xgb_model,filename="predictedcover_xgb_map.tif",na.rm=TRUE, progress="text",format="GTiff", overwrite=TRUE)  

########################################################################################