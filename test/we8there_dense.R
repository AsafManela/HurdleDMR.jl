# convert we8there data to something useful
library(textir)
library(Matrix)

## some multinomial inverse regression
## we'll regress counts onto 5-star overall rating
data(we8there)

## cl=NULL implies a serial run. 
## To use a parallel library fork cluster, 
## uncomment the relevant lines below. 
## Forking is unix only; use PSOCK for windows
#cl <- NULL
cl <- makeCluster(detectCores(), type="FORK")
## small nlambda for a fast example
fits <- mnlm(cl, we8thereRatings[,'Overall',drop=FALSE], 
             we8thereCounts, bins=5, gamma=1, nlambda=10)
fits_binless <- mnlm(cl, we8thereRatings[,'Overall',drop=FALSE], 
             we8thereCounts, bins=NULL, gamma=1, nlambda=10)
stopCluster(cl)

## plot fits for a few individual terms
terms <- c("first date","chicken wing",
           "ate here", "good food",
           "food fabul","terribl servic")
par(mfrow=c(3,2))
for(j in terms)
{ 	plot(fits[[j]]); mtext(j,font=2,line=2) }

## extract coefficients
B <- coef(fits)
B_binless <- coef(fits_binless)
A <- as.matrix(cbind(t(B),t(B_binless)))

mean(B[2,]==0) # sparsity in loadings
## some big loadings in IR
B[2,order(B[2,])[1:10]]
B[2,order(-B[2,])[1:10]]

## do MNIR projection onto factors
z <- srproj(B,we8thereCounts) 

## fit a fwd model to the factors
summary(fwd <- lm(we8thereRatings$Overall ~ z)) 

## truncate the fwd predictions to our known range
fwd$fitted[fwd$fitted<1] <- 1
fwd$fitted[fwd$fitted>5] <- 5
## plot the fitted rating by true rating
par(mfrow=c(1,1))
plot(fwd$fitted ~ factor(we8thereRatings$Overall), 
     varwidth=TRUE, col="lightslategrey")