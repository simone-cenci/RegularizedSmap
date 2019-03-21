MeanCorrelation <- function(TS, X){
  rho  = rmse = c()
  for(i in 1:ncol(X)){
    rho = c(rho, cor(TS[,i], X[,i]))
    rmse = c(rmse, sqrt(mean((TS[,i] - X[,i])^2)))
  }
  return(list( correlation = mean(rho), rmse = mean(rmse)))
}
test.error.rho.simone = test.error.rmse.simone = c()
num.of.files = length(list.files(path = '../PredictedData/'))
for(example in 1:num.of.files)
{
	name.test.data = paste('../DataForComparison/TestData/TestingData_Example_', example,'.txt', sep = '')
	name.simone.data = paste('../PredictedData/TestDataSimone_', example,'.txt', sep = '')
	######
	test.data = as.matrix(read.table(name.test.data))
	test.simone = as.matrix(read.table(name.simone.data ))
	######	
	test.error.rho.simone = c(test.error.rho.simone, MeanCorrelation(test.data, test.simone)$correlation)
	test.error.rmse.simone = c(test.error.rmse.simone, MeanCorrelation(test.data, test.simone)$rmse)
	
}
cat('Performance on test set\n')
cat('Median correlation Simone:', median(test.error.rho.simone), '\n')
cat('######\n')
cat('Median RMSE Simone:', median(test.error.rmse.simone), '\n')
cat('######\n')
hist(test.error.rho.simone, breaks = 30)
hist(test.error.rmse.simone)
