rm(list = ls())
suppressMessages(library(Matrix))
suppressMessages(library(parallel))
suppressMessages(library(compiler))
suppressMessages(library(lars))
suppressMessages(library(elasticnet))
options(warn=-1)
#####################################################################################
source('Auxiliar.r')
source('elastic_net_fit.r')
source('LOOCV.r')
source('KernelFunctions.r')
source('OutOfSample.r')
source('Interactions.R')
source('TrainingError.r')
source('ridge.r')
################# Regularized S-map for inference and forecasting ###################
cat('Example: here we use the kernel and the alpha selected in model selection.r\n')
ShowPlot = T
options.models = c('inputFiles/PredatorPrey.txt', 'inputFiles/RPS.txt', 
                   'inputFiles/Chaotic_LV.txt')
options.jacobian = c('inputFiles/jacobian_predator.txt', 'inputFiles/jacobian_rps.txt', 
                     'inputFiles/jacobian_chaos.txt')
### The regression kernel should be chosen from 'model_selection.r'
choice = 2
kernel.choice = 1
ObservationalNoise = F
regularization.type = 1
### alpha should be chosen from 'model_selection.r'
alpha = 0.95
####
Regularization.schemes = c('ridge_fit', 'ELNET_fit')
RegressionType = Regularization.schemes[regularization.type]
###################################################################
FileName = options.models[choice]
JacobianName = options.jacobian[choice]
####################################
Kernel.Options = c('Exponential.Kernel', 'Epanechnikov.Kernel', 'TriCubic.Kernel')
Regression.Kernel = Kernel.Options[kernel.choice]
###################################
cat('Model:',  FileName, '\nKernel:', Regression.Kernel,'\nObservational Noise:', 
    ObservationalNoise, '\nRegularization:', Regularization.schemes[regularization.type], '\n')
###################################
logspace <- function(d1, d2, n) exp(log(10)*seq(d1, d2, length.out=n)) 
############# Parameters for cross validation
lambda = logspace(-3,0,15)                       
tht = logspace(-2,1,15)        
parameters_on_grid = expand.grid(tht, lambda)     
##### length of training and test set
length.training = 100
length.testing = 30
### Read Time series
d = ReadTimeSeries(FileName)
### Use all the species
Embedding = LETTERS[1:ncol(d)]
TargetList = Embedding
######################
dfdx = expand.grid(TargetList, TargetList)
######################
d = d[, Embedding]

### Just as example
t.min = 1
#### If you want to measure performance across random starting points initialized randomly
#t.min = floor(runif(1,1, nrow(d) - length.training - length.testing-1))

#### Random Chunk of length:
length_of_interval = length.training + length.testing
t.max = t.min + length_of_interval - 1
interval = t.min:t.max
interval_training = 1:length.training
interval_testing = (length.training + 1):length_of_interval
#### Subset the chunk
d_intact = d[interval,]
d.training = d_intact[interval_training, ]
#### Make noise if you want
if(ObservationalNoise == T){
  d.training = d.training + matrix(rnorm(length(d.training), 0,mean(d.training)*0.05), 
                                              nrow(d.training), ncol(d.training))   
}
### Preserve the training set to standardize the test set
d.train.to.test.set = d.training
d.training = Standardizza(d.training)
#### Prepare for the fit in parallel (keep one core out)
Lavoratori = detectCores() - 1
cl <- makeCluster(Lavoratori, type = "FORK")
BestModel = LOO.CV(cl, d.training, TargetList, Embedding, 
                                parameters_on_grid, 
                                RegressionType,alpha, Regression.Kernel)
stopCluster(cl)
#### Coefficient of the Jacobian and intercept
BestCoefficients = BestModel$BestCoefficients
#### Bandwith of the kernel and regularization parameter
BestParameters = BestModel$BestParameters
### Out-of-sample forecast
out.of.samp = forecast.function(BestCoefficients, 
                                BestParameters$BestTH,
                                BestParameters$BestLM, 
                                alpha,
                                d.training, length.testing, Regression.Kernel)
prd = out.of.samp$out_of_samp
#### The test set is only defined here after the predictions
d.testing = d_intact[interval_testing, ]
d.testing = Standardizza.test(d.testing,d.train.to.test.set)
#### Compute the naive forecast
naive.prediction = matrix(0,nrow(d.testing), ncol(d.testing))
for(coll in 1:ncol(naive.prediction)){
  naive.prediction[,coll] = rep(d.training[nrow(d.training),coll], nrow(d.testing))
}
###### Now check the in-sample error
TrainErr = ComputeTrainingError(d.training, BestCoefficients)
reconstruction = ReconstructionOfTrainingSet(d.training, BestCoefficients)
###### Now check the quality of the out-of-sample forecast 
Test.rho = MeanCorrelation(d.testing, prd)$correlation
Test.rmse = MeanCorrelation(d.testing, prd)$rmse
###### Now check the quality of the inference
rho = jacobiano_analitico(JacobianName,
                                BestCoefficients$J, 
                                ncol(d), t.min, t.min+length.training-1)
matrix.inference.rho = rho$correlation.matrix
matrix.inference.rmse = rho$rmse.matrix
#####
training.error.rho = TrainErr$rho
training.error.rmse = compute.rmse.train(d.training,reconstruction)

  
jacobian.inference.rho = mean(matrix.inference.rho[!is.na(matrix.inference.rho)])
jacobian.inference.rmse = mean(matrix.inference.rmse[!is.na(matrix.inference.rmse)])
naive.error.rmse = sqrt(mean((naive.prediction - d.testing)^2))

##### Make a couple of plots to visualize the results
if(ShowPlot == TRUE){
  source('PlotFunctions.r')
  par(mfrow = c(ncol(d.training), 1))
  for(i in 1:ncol(d.training)){
    Put_train_test_together(d.training[2:nrow(d.training),],d.testing,
                            reconstruction, prd, i)
  }
  par(mfrow = c(1,1))
}
cat('###########################\nTrained on:', length.training, 'data points\nTested on:', 
    length.testing, 'data points', 
    '\n######### Performance on test set\nCorrelation coefficient:', Test.rho, 
    '\nRMSE: ', Test.rmse, '\nNaive RMSE:', naive.error.rmse,'\n')
