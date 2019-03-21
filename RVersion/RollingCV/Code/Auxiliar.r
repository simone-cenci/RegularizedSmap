### To make next step prediction
Testing <- function(J, c0, X){
  return(c0 + J%*%X)
}
Add_to_TS <- function(TS, x){
  return(rbind(TS, x))
}

Standardizza <- function(X){
  for(i in 1:ncol(X)){
    X[,i] = (X[,i]- mean(X[,i]))/sd(X[,i])
  }
  return(X)
}
Standardizza.test <- function(X, Y){
  ### X = test set
  ### Y = training set
  ### This return y = (x-meanY)/stdY
  for(i in 1:ncol(X)){
    X[,i] = (X[,i]- mean(Y[,i]))/sd(Y[,i])
  }
  return(X)
}
