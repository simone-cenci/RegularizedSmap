ridge_fit <- function(time.series, theta, lambda, Regression.Kernel)
  {
  Edim <- ncol(time.series)
  block <- cbind(time.series[2:dim(time.series)[1],],time.series[1:(dim(time.series)[1]-1),])
  block <- as.data.frame(block)
  
  lib <- 1:dim(block)[1]
  pred <- 1:dim(block)[1]
  Js = c0s = list()
  
  lm_regularized <- function(y, x, ws, lambda, dimension, subset = seq_along(y)){
    #########################################################
    WWs = diag(ws)
    Xx = cbind(1, as.matrix(x))
    coeff <- solve(t(Xx) %*% WWs %*% Xx + lambda*nrow(Xx)*diag(1,dimension + 1)) %*% t(Xx) %*%(WWs %*% y)
    return(t(coeff))
  }
  for (ipred in 1:length(pred)){
    libs = lib[-pred[ipred]]
    q <- matrix(as.numeric(block[pred[ipred],(Edim+1):dim(block)[2]]),
                ncol=Edim, nrow=length(libs), byrow = T)
    distances <- sqrt(rowSums((block[libs,(Edim+1):dim(block)[2]] - q)^2))
    Krnl = match.fun(Regression.Kernel)
    Ws = Krnl(distances, theta)
    fit <- lm_regularized(as.matrix(block[libs,1:Edim]),
                          as.matrix(block[libs,(Edim+1):dim(block)[2]]),Ws, lambda, Edim)
    Js[[ipred]] <- fit[,2:(Edim+1)]
    c0s[[ipred]] <- fit[,1]
  }
  return(Js)
}

