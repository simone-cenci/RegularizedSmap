update.ridge <- function(time.series, theta, lambda, Regression.Kernel)
{
  Edim <- ncol(time.series)
  block <- cbind(time.series[2:dim(time.series)[1],],time.series[1:(dim(time.series)[1]-1),])
  block <- as.data.frame(apply(block, 2, function(x) (x-mean(x))/sd(x)))
  lib <- 1:dim(block)[1]
  pred <- 1:dim(block)[1]
  lm_regularized <- function(y, x, ws, lambda, dimension, subset = seq_along(y)){
    #########################################################
    WWs = diag(ws)
    Xx = cbind(1, as.matrix(x))
    coeff <- solve(t(Xx) %*% WWs %*% Xx + lambda*nrow(Xx)*diag(1,dimension + 1)) %*% t(Xx) %*%(WWs %*% y)
    return(t(coeff))
  }
  ipred = length(pred)
  libs = lib[-pred[ipred]]
  q <- matrix(as.numeric(block[pred[ipred],(Edim+1):dim(block)[2]]),
              ncol=Edim, nrow=length(libs), byrow = T)
  distances <- sqrt(rowSums((block[libs,(Edim+1):dim(block)[2]] - q)^2))
  Krnl = match.fun(Regression.Kernel)
  Ws = Krnl(distances, theta)
  fit_ <- lm_regularized(as.matrix(block[libs,1:Edim]),
                        as.matrix(block[libs,(Edim+1):dim(block)[2]]),Ws, lambda, Edim)
  colnames(fit_) <- c('c0', letters[1:Edim])
  return(list(J = fit_[,2:(Edim+1)], c0 = fit_[,1]))
}
#### Forecast
forecast.function <- function(th, lm, ts_training, num_points, 
                              Regression.Kernel){
  out_of_samp = c()
  ### Take the last point in the training set
  new_point = ts_training[nrow(ts_training), ]
  updated.fit = update.ridge(ts_training, th, lm, Regression.Kernel)

  for(j in 1:num_points){
    ### Predict the first point in the training set and then allthe others
    new_point = Testing(updated.fit$J, updated.fit$c0, new_point)
    out_of_samp = rbind(out_of_samp, t(new_point))
    ts_training = Add_to_TS(ts_training, t(new_point))
    updated.fit = update.ridge(ts_training, th, lm, Regression.Kernel)
  }
  return(out_of_samp)
}
