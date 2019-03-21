rm(list = ls())
source('PlotFunctions.r')

options.models = c('inputFiles/PredatorPrey.txt', 'inputFiles/RPS.txt', 
                   'inputFiles/Chaotic_LV.txt')
length.training = 100
length.testing = 30
ObservationalNoise = T
cat('Observational Noise in training data:', ObservationalNoise)
for(k in 1:200){
  ##### Choose a random time series
  FileName = options.models[floor(runif(1, 1, 4))]
  ### Read Time series
  d = ReadTimeSeries(FileName)
  ### Use all the species
  Embedding = LETTERS[1:ncol(d)]
  TargetList = Embedding
  ######################
  dfdx = expand.grid(TargetList, TargetList)
  ######################
  d = d[, Embedding]
  ####
  t.min = floor(runif(1,1, nrow(d) - length.training - length.testing-1))
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
  #### The test set is only defined here after the predictions
  d.testing = d_intact[interval_testing, ]
  d.testing = Standardizza.test(d.testing,d.train.to.test.set)
  to.save.training = paste('TrainingData/TrainingData_Example_', k, '.txt', sep = '')
  write.table(d.training, file = to.save.training, row.names = F, col.names = F)
  to.save.testing = paste('TestData/TestingData_Example_', k, '.txt', sep = '')
  write.table(d.testing, file = to.save.testing, row.names = F, col.names = F)
}
par(mfrow = c(ncol(d.training), 1))
for(i in 1:ncol(d.training)){
  Put_train_test_together(d.training[2:nrow(d.training),],d.testing,i)
}
par(mfrow = c(1,1))

