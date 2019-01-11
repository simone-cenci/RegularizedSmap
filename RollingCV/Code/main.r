rm(list = ls())
suppressMessages(library(Matrix))
suppressMessages(library(quantreg))
suppressMessages(library(parallel))
suppressMessages(library(compiler))
suppressMessages(library(lars))
suppressMessages(library(tictoc))
suppressMessages(library(elasticnet))
options(warn=-1)
#####################################################################################
source('Auxiliar.r')
source('elastic_net_fit.r')
source('moving_window.r')
source('KernelFunctions.r')
source('OutOfSample.r')
source('ridge.r')

Kernel.Options = c('Exponential.Kernel', 'Epanechnikov.Kernel', 'TriCubic.Kernel')
###################################
logspace <- function(d1, d2, n) exp(log(10)*seq(d1, d2, length.out=n)) 
######
length.testing = 30
#### Create parameters for cross-validation
choice = c(1,2,3)
lambda = logspace(-3,0,15)
tht = logspace(-2,1.2,15)
parameters_on_grid = expand.grid(tht, lambda, choice)
######
Lavoratori = detectCores()-1
cl <- makeCluster(Lavoratori, type = "FORK")

example = 1
num.files = 200
tic('Ensemble performance')
for(example in 1:num.files)
{
  cat('Running ', example, '/', num.files, '\n', sep = '')
  name.for.training = paste('../DataForComparison/TrainingData/TrainingData_Example_', example,'.txt', sep = '')
  d.training = as.matrix(read.table(name.for.training))
  colnames(d.training) = LETTERS[1:ncol(d.training)]
  #### Run rolling window cross validation
  output.training = mv.training(cl, d.training, parameters_on_grid, Kernel.Options)
  #### Jacobian coefficients
  Jacobian.TS = output.training$jacobiano
  #### Predict
  prd = forecast.function(rep(output.training$best.theta,ncol(d.training)),
                          rep(output.training$best.lambda,ncol(d.training)),
                          d.training, length.testing,
                          output.training$Regression.Kernel)
  #### Save file
  file.to.save = paste('../SimonePredictions/TestDataSimone_', example, '.txt', sep ='')
  write.table(prd, file = file.to.save, row.names = F, col.names = F)
}
toc()
stopCluster(cl)
