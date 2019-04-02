mv <- function(s, X, grid)
{
  #### Rolling window cross validation
  #### -----------0***  |
  #### ------------0*** |   
  #### -------------0***|
  #### -------------0****
  #### Look carefully at the way you divide training and validation
  length.validating = 10
  repetitions = 10
  validation.error = rep(0,repetitions+1)
  for(i in 0:repetitions)
  {

  x.training = X[1:(nrow(X) - length.validating - repetitions + i),]
  x.validating = X[(nrow(X) - length.validating - repetitions+i+1):(nrow(X)- length.validating - repetitions+i+length.validating),]

  out.of.samp = forecast.function(grid[s,1], grid[s,2], x.training, 
                                  nrow(x.validating),
                                  Kernel.Options[grid[s,3]])
  validation.error[i] = sqrt(mean((out.of.samp - x.validating)**2))
  }
  return(list(bndwth = grid[s,1], lmb = grid[s,2], krn = grid[s,3], rmse = mean(validation.error)))
}

MV.CV <- function(cl, data.df, grid){
  S_target <- parLapply(cl, 1:nrow(grid), mv, data.df, grid)
  error.mat = cbind(unlist(S_target)[attr(unlist(S_target),"names") == "bndwth"],
                    unlist(S_target)[attr(unlist(S_target),"names") == "lmb"],
                    unlist(S_target)[attr(unlist(S_target),"names") == "krn"],
                    unlist(S_target)[attr(unlist(S_target),"names") == "rmse"])
  rownames(error.mat) = c()
  error.mat[,4] = round(error.mat[,4],3)
  error.mat = error.mat[order(error.mat[,4]), ]
  idx = which(error.mat[,4] == min(error.mat[,4]))
  idx.th = which(grid[idx,1] == min(grid[idx,1]))
  idx = idx[min(idx.th)]
  return(list(BestTH = error.mat[idx,1],
              BestLM = error.mat[idx,2],
              BestKRN = error.mat[idx,3],
              min.val.err = error.mat[idx,4],
              val.err = error.mat))
}
mv.training <- function(cl, X, grid, Krn.Options)
{
  ### X = training data 
  fit_ = MV.CV(cl, X, grid)
  #### Make coefficients
  Jacobian.TS = ridge_fit(X,fit_$BestTH, fit_$BestLM, Krn.Options[fit_$BestKRN])
  return(list(best.theta = fit_$BestTH, best.lambda = fit_$BestLM,
              Regression.Kernel = Kernel.Options[fit_$BestKRN],
              validation.error = fit_$val.err,
              jacobiano = Jacobian.TS))
}
