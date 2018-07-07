corGBM <- function(n, r, t=1/365, plot=TRUE) {
  #n is number of samples 
  #r is correlation
  #t is tick step
  x <- rnorm(n, mean=0, sd= 1)
  se <- sqrt(1 - r^2) #standard deviation of error
  e <- rnorm(n, mean=0, sd=se)
  y <- r*x + e
  
  X <- cumsum(x* sqrt(t))
  Y <- cumsum(y* sqrt(t))
  Max <- max(c(X,Y))
  Min <- min(c(X,Y))
  
  if(plot) {
    plot(X, type="l", ylim=c(Min, Max))
    lines(Y, col="blue")
  }
  return(cor(x,y))
}

#sample result
corGBM(10000,.85)
