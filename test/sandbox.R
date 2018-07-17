# get source
pkgname <- "pscl"
destdir <- "~/Dropbox/KellyManelaMoreira/HurdleDMR/R"
untar(download.packages(pkgs = pkgname,destdir = destdir,type = "source")[,2],exdir = destdir)

library(MASS)
library(textir)
data(fgl)
covars = fgl[,1:9]
counts = fgl[,10:10]
cl <- makeCluster(detectCores(), type="FORK")
col <- collapse(covars,counts)
fits = dmr(cl,covars, counts, verb=1)
fits2 = mnlm(cl,covars, counts, verb=1)
stopCluster(cl)
col$counts
coef(fits)
z <- srproj(fits,col$counts)
# dim(z1)
# obj <- coef(fits)
# obj2 <- obj[-1, , drop = FALSE]
# obj
# obj2
# m <- rowSums(col$counts)
# z2 <- cbind(tcrossprod(col$counts,obj),m)
# all(z1==z2)
#
# untar()

## some multinomial inverse regression
## we'll regress counts onto 5-star overall rating
data(we8there)

cl <- makeCluster(detectCores(), type="FORK")

ptm <- proc.time()
fits <- dmr(cl, we8thereRatings[,'Overall',drop=FALSE], we8thereCounts, bins=NULL, gamma=1, nlambda=100)
B <- coef(fits)
proc.time() - ptm

stopCluster(cl)

B
## plot fits for a few individual terms
terms <- c("first date","chicken wing",
           "ate here", "good food",
           "food fabul","terribl servic")
par(mfrow=c(3,2))
for(j in terms)
{ 	plot(fits[[j]]); mtext(j,font=2,line=2) }

## extract coefficients
B <- coef(fits)
B
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

#####################################################
# hurdle
#####################################################
library(pscl)
data("bioChemists", package = "pscl")

## logit-poisson
## "art ~ ." is the same as "art ~ . | .", i.e.
## "art ~ fem + mar + kid5 + phd + ment | fem + mar + kid5 + phd + ment"
fm_hp1 <- hurdle(art ~ ., data = bioChemists)
summary(fm_hp1)

fm_hp1 <- hurdle(art ~ fem + mar + kid5 + phd + ment, data = bioChemists)
summary(fm_hp1)

fm_hp2 <- hurdle(art ~ fem + mar + kid5 | phd + ment, data = bioChemists)
summary(fm_hp2)


#####################################################
# pospoisson
#####################################################
library(VGAM)

# Data from Coleman and James (1961)
cjdata <- data.frame(y = 1:6, freq = c(1486, 694, 195, 37, 10, 1))
fit <- vglm(y ~ 1, pospoisson, data = cjdata, weights = freq)
Coef(fit)
summary(fit)
fitted(fit)

set.seed(13)
pdata <- data.frame(x2 = runif(nn <- 1000))  # Artificial data
pdata <- transform(pdata, lambda = exp(1 - 2 * x2))
pdata <- transform(pdata, y1 = rpospois(nn, lambda))
with(pdata, table(y1))
fit <- vglm(y1 ~ x2, pospoisson, data = pdata, trace = TRUE, crit = "coef")
coef(fit, matrix = TRUE)
