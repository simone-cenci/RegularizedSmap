############ For the plotting of the time series
Put_train_test_together <- function(d.tr, d.ts, species){
  title = paste('Species', species)
  
  plot(c(d.tr[,species],d.ts[,species]), type = 'l', lwd = 2, ylim = c(-2,2), 
       main = title, ylab = expression('x'[species]))
  abline(v=(nrow(d.tr)+1), col="blue", lty = 2, lwd = 1.5)
}
ReadTimeSeries <- function(Nome){
  X = as.matrix(read.table(Nome, header = F))
  colnames(X) =  LETTERS[1:ncol(X)]
  return(X)
}
Standardizza <- function(X){
  ### This return y = (x-meanx)/stdx
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