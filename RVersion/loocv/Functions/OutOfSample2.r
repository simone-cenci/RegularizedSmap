update.elnet <- function(time.series, targ_col, Embedding, theta, lambda,alp)
  {
  Edim <- length(Embedding)
  coeff_names <- sapply(colnames(time.series),function(x) paste("d", targ_col, "d", x, sep = ""))
  block <- cbind(time.series[2:dim(time.series)[1],targ_col],time.series[1:(dim(time.series)[1]-1),])
  block <- as.data.frame(apply(block, 2, function(x) (x-mean(x))/sd(x)))
  
  lib <- 1:dim(block)[1]
  pred <- 1:dim(block)[1]
  
  coeff <- array(0,dim=c(1,Edim + 1))
  colnames(coeff) <- c('c0', coeff_names)
  coeff <- as.data.frame(coeff)
  ipred = length(pred)
  libs = lib[-pred[ipred]]
  q <- matrix(as.numeric(block[pred[ipred],2:dim(block)[2]]),
              ncol=Edim, nrow=length(libs), byrow = T)
  distances <- sqrt(rowSums((block[libs,2:dim(block)[2]] - q)^2))
  ### Kernel
  Krnl = match.fun(Regression.Kernel)
  Ws = Krnl(distances, theta)
  ############ Fit function
  x = as.matrix(block[libs,2:dim(block)[2]])
  y = as.matrix(block[libs,1])
  x = x[seq_along(y), ]
  y = y[seq_along(y)]
  Ws = sqrt(Ws[seq_along(y)])
  x = Ws * cbind(1, x)
  y = Ws * y
  fit <- enet(x, y, lambda = lambda, normalize = TRUE, intercept = FALSE)
  coeff[1,] <- predict(fit, s = alp, type="coefficients", mode="fraction")$coefficients 
  return(coeff)
}


update.ridge <- function(time.series, targ_col, Embedding, theta, lambda, alp, Regression.Kernel)
{
  Edim <- length(Embedding)
  coeff_names <- sapply(colnames(time.series),function(x) paste("d", targ_col, "d", x, sep = ""))
  block <- cbind(time.series[2:dim(time.series)[1],targ_col],time.series[1:(dim(time.series)[1]-1),])
  block <- as.data.frame(apply(block, 2, function(x) (x-mean(x))/sd(x)))
  
  lib <- 1:dim(block)[1]
  pred <- 1:dim(block)[1]
  
  coeff <- array(0,dim=c(1,Edim + 1))
  colnames(coeff) <- c('c0', coeff_names)
  coeff <- as.data.frame(coeff)
  
  lm_regularized <- function(y, x, ws, lambda, dimension, subset = seq_along(y)){
    x <- x[subset,]
    y <- y[subset]
    ws <- ws[subset]
    #########################################################
    WWs = diag(ws)
    Xx = as.matrix(x)
    Xx = cbind(1, Xx)
    coeff <- solve(t(Xx) %*% WWs %*% Xx + lambda*nrow(Xx)*diag(1,dimension + 1)) %*% t(Xx) %*%(ws * y)
    coeff <- t(coeff)
    return(coeff)
  }
  ipred = length(pred)
  libs = lib[-pred[ipred]]
  q <- matrix(as.numeric(block[pred[ipred],2:dim(block)[2]]),
              ncol=Edim, nrow=length(libs), byrow = T)
  distances <- sqrt(rowSums((block[libs,2:dim(block)[2]] - q)^2))
  Krnl = match.fun(Regression.Kernel)
  Ws = Krnl(distances, theta)
  fit <- lm_regularized(block[libs,1],block[libs,2:dim(block)[2]],Ws, lambda, Edim)
  coeff[1,] <- fit

  return(coeff)
}


new.coefficients <- function(X, TargetList, Embedding, th, lm, alpha){
  J = c0 = list()
  n_ = 1
  for(df in TargetList){
    ########## Now compute the optimum regularized coefficients
    if(RegressionType == 'ELNET_fit'){
      J[[n_]]  = update.elnet(X, df, Embedding, th[n_], lm[n_],alpha)
    }else{
      J[[n_]]  = update.ridge(X, df, Embedding, th[n_], lm[n_],alpha, Regression.Kernel)
    }
    c0[[n_]] = J[[n_]]$c0
    J[[n_]] = J[[n_]][-1]
    n_ = n_ + 1
  }
  return(list(J = J, c0 = c0))
}
######################################################################
next.Jacobian <- function(X, TargetList, Embedding, th, lm, alpha){
  mine_output = new.coefficients(X, TargetList, Embedding, th, lm, alpha)
  mine_c0  = mine_output$c0
  mine_output = mine_output$J
  
  J = list()
  c0 = do.call(cbind, lapply(1:ncol(X), function(x, M) unlist(M[[x]]), mine_c0))
  colnames(c0) = sapply(TargetList,function(x) paste("c0_", x, sep = ""))
  k = 1
  J[[k]] = do.call(rbind, lapply(1:ncol(X), function(x, M, i) unlist(M[[x]][i,]), mine_output, k))
  rownames(J[[k]]) = LETTERS[1:ncol(X)]
  colnames(J[[k]]) = LETTERS[1:ncol(X)]

  return(list(J = J, c0 = c0))
}
#### Forecast
forecast.function <- function(Jacobiano, th, lm, alpha, ts_training, num_points, ...){
  out_of_samp = c()
  coeff = list()
  ### Take the last point in the training set
  new_point = ts_training[nrow(ts_training), ]
  for(j in 1:num_points){
    ### Predict the first point in the training set and then allthe others
    new_point = Testing(Jacobiano$J, Jacobiano$c0, new_point)
    out_of_samp = rbind(out_of_samp, t(new_point))
    ts_training = Add_to_TS(ts_training, t(new_point))
    updated.fit = next.Jacobian(ts_training, TargetList, Embedding, th, lm, alpha)
    Jacobiano$J[length(Jacobiano$J)+1] = updated.fit$J
    Jacobiano$c0 = rbind(Jacobiano$c0, updated.fit$c0)
  }
  return(list(out_of_samp = out_of_samp))
}