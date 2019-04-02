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
source('Functions/Auxiliar.r')
source('Functions/moving_window.r')
source('Functions/KernelFunctions.r')
source('Functions/Interactions.R')
source('Functions/OutOfSample.r')
source('Functions/ridge.r')

Kernel.Options = c('Exponential.Kernel', 'Epanechnikov.Kernel', 'TriCubic.Kernel')
###################################
logspace <- function(d1, d2, n) exp(log(10)*seq(d1, d2, length.out=n)) 
######
length.training = 400
length.testing = 100
#### Create parameters for cross-validation
choice = c(1,2,3)
lambda = logspace(-5,0,15)
tht = logspace(-2,1.2,15)
parameters_on_grid = expand.grid(tht, lambda, choice)
######
Lavoratori = detectCores()-3
cl <- makeCluster(Lavoratori, type = "FORK")

input.file = 'inputFiles/deterministic_chaos_k.txt'
JacobianName = 'inputFiles/jacobian_chaos_k.txt'
t.min = 1
d.training = as.matrix(read.table(input.file))[t.min:length.training,]
d.train.to.test.set = d.training
d.training = scale(d.training)
colnames(d.training) = LETTERS[1:ncol(d.training)]
#### Run rolling window cross validation
output.training = mv.training(cl, d.training, parameters_on_grid, Kernel.Options)
stopCluster(cl)

#### Inference
Jacobian.TS = output.training$jacobiano
rho = jacobiano_analitico(JacobianName,
                          Jacobian.TS,
                          ncol(d.training), t.min, t.min+length.training-1)
matrix.inference.rho = rho$correlation.matrix
#### 
cat('Trace inference:', rho$trace.correlation, '\n')


#### Predict
prd = forecast.function(rep(output.training$best.theta,ncol(d.training)),
                        rep(output.training$best.lambda,ncol(d.training)),
                        d.training, length.testing,
                        output.training$Regression.Kernel)
d.testing =  as.matrix(read.table(input.file))[(t.min + length.training):(t.min + length.training + length.testing - 1),]
d.testing = Standardizza.test(d.testing, d.train.to.test.set)
correlation = mean(unlist(lapply(1:ncol(d.testing), function(x, X, Y) cor(X[,x], Y[,x]), prd, d.testing)))
cat('correlation:', correlation, '\n')

toc()

#### Make some plot
## VCR
plot(scale(rho$true.trace), type = 'l', lwd = 3)
lines(scale(rho$inferred.trace), lwd = 3, col = 'red')

## Forecast
idx = 2
plot(d.testing[,idx], type = 'l', lwd = 3)
lines(prd[,idx], col = 'red', lwd = 3)
