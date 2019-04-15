rm(list = ls())
suppressMessages(library(Matrix))
suppressMessages(library(parallel))
suppressMessages(library(compiler))
suppressMessages(library(lars))
suppressMessages(library(elasticnet))
options(warn=-1)
#####################################################################################
source('Functions/Auxiliar.r')
source('Functions/elastic_net_fit.r')
source('Functions/LOOCV.r')
source('Functions/KernelFunctions.r')
source('Functions/OutOfSample2.r')
source('Functions/Interactions.R')
source('Functions/TrainingError.r')


Kernel.Options = c('Exponential.Kernel', 'Epanechnikov.Kernel', 'TriCubic.Kernel')
kernel.choice = 1
alpha = 1
#### Prepare for the fit in parallel (keep one core out)
Lavoratori = detectCores() - 1
cl <- makeCluster(Lavoratori, type = "FORK")

RegressionType = 'ELNET_fit'
###################################################################
FileName = 'inputFiles/deterministic_chaos_k.txt'
JacobianName = 'inputFiles/jacobian_chaos_k.txt'
####################################
Regression.Kernel = Kernel.Options[kernel.choice]
###################################
logspace <- function(d1, d2, n) exp(log(10)*seq(d1, d2, length.out=n)) 
############# Parameters for cross validation
parameters_on_grid = expand.grid(logspace(-1.5,1.2,15) , logspace(-3,0,15))     
##### length of training and test set
length.training = 400
length.testing = 100
### Read Time series
d = ReadTimeSeries(FileName)
### Use all the species
Embedding = LETTERS[1:ncol(d)]
TargetList = Embedding
######################
dfdx = expand.grid(TargetList, TargetList)
######################
d = d[, Embedding]
t.min = 1
#### Subset the chunk
d.training = d[t.min:(t.min + length.training - 1), ]
### Preserve the training set to standardize the test set
d.train.to.test.set = d.training
d.training = Standardizza(d.training)


##### Here start the proper function:
##### First infer the jacobian coefficients
##### Second forecast the species abundance


#### Inference
BestModel = LOO.CV(cl, d.training, TargetList, Embedding, 
		               parameters_on_grid, 
		               RegressionType,alpha, Regression.Kernel)
stopCluster(cl)
#### Coefficient of the Jacobian and intercept
BestCoefficients = BestModel$BestCoefficients
#### Bandwith of the kernel and regularization parameter
BestParameters = BestModel$BestParameters
###### Now check the quality of the inference
rho = jacobiano_analitico(JacobianName,
			                    BestCoefficients$J, 
			                    ncol(d), t.min, t.min+length.training-1)
matrix.inference.rho = rho$correlation.matrix
#### 
cat('Trace inference:', rho$trace.correlation, '\n')
plot(scale(rho$true.trace), type = 'l', lwd = 3)
lines(scale(rho$inferred.trace), lwd = 3, col = 'red')



#### Forecast
out.of.samp = forecast.function(BestCoefficients, 
                                BestParameters$BestTH,
                                BestParameters$BestLM, 
                                alpha,
                                d.training, length.testing, Regression.Kernel)
prd = out.of.samp$out_of_samp
d.testing =  d[(t.min + length.training):(t.min + length.training + length.testing - 1), ]
d.testing = Standardizza.test(d.testing, d.train.to.test.set)
correlation = mean(unlist(lapply(1:ncol(d.testing), function(x, X, Y) cor(X[,x], Y[,x]), prd, d.testing)))
cat('correlation:', correlation, '\n')
idx = 1
plot(d.testing[,idx], type = 'l', lwd = 3)
lines(prd[,idx], col = 'red', lwd = 3)
