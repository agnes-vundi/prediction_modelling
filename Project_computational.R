
### Data.ML-390 Spring 2022
### Project

############################
### Question 3

#Install from github the datarium package: https://rdrr.io/github/kassambara/datarium/
#Load the marketing data set in R by: data("marketing", package = "datarium")
#. (I) Discuss the meaning of the data.
#. (II) Find the best linear regression model to predict sales by using 10-fold CV
#and LOOCV. Estimate in addition to the MSE its standard error.
#. (III) Discuss the results.

# loading the data
#install.packages("remotes")
#remotes::install_github("kassambara/datarium")
data("marketing", package = "datarium")

# checking the data set
head(marketing)  # checking the data contents

#   youtube facebook newspaper sales
# 1  276.12    45.36     83.04 26.52
# 2   53.40    47.16     54.12 12.48
# 3   20.64    55.08     83.16 11.16
# 4  181.80    49.56     70.20 22.20
# 5  216.96    12.96     70.08 15.48
# 6   10.44    58.68     90.00  8.64

dim(marketing) # 200 by 4 dimensions
names(marketing) # column names: "youtube"   "facebook"  "newspaper" "sales"
sum(is.na(marketing))  # checking for missing values [1] 0


########## I) Discussion of the marketing dataset
# The marketing data set (datarium package) contains the impact of three 
# advertising medias (youtube, facebook and newspaper) on sales. These are used 
# for predicting sales units on the basis of the amount of money in thousands of  
# dollars spent in the three advertising medias. Data are the advertising budget  
# in thousands of dollars along with the sales. The advertising experiment has  
# been repeated 200 times with different budgets and the observed sales have been 
# recorded.


####### TASK: Fitting a multiple linear regression model to predict sales by using 10-fold
####### and LOOCV methods for data resampling i.e the partitioning the data into training and
####### testing subsets of the dataset. The equation of the model is of the form y= b0+b1x1+b2x2+b3x3
####### where y:response variable (sales), x1,x2 and x3 are the covariate variables i.e
####### youtube, facebook and newspaper respectively.

###################################################
### 1. leave-one-out method (LOOCV) to fit the model.
####################################################
n = nrow(marketing)

# placeholder for storing the i-th prediction

preds <- rep(0, n)
sample.mse <- rep(0, n)

for(i in 1:n) {
  
  data.train <- marketing[-i, ]
  data.test <- marketing[i, ]
  loocv.fit <- lm(sales ~ youtube + facebook + newspaper, data=data.train)
  preds[i] <- predict(loocv.fit , data.test)
  
}

# calculating the mean square error of the total sum of residual
nsize = length(preds)
RSS = sum((marketing$sales - preds)^2)
MSE.loocv = RSS/nsize  # 4.243536
#mean((marketing$sales - preds)^2)  4.243536

# Residual standard Error 
RSE = sqrt(RSS/(nsize-2))   # 2.070362
# sigma(loocv.fit) # Residual standard error: 2.022

# Standard Error of the MSE: SE=sigma/sqrt(sample size:N)
SE_Loocv = sd((marketing$sales - preds)^2)/sqrt(n)  # 0.7066973

############################## 
## Analysis and visualizing the model fit and parameters plus residuals
##############################

summary(loocv.fit)
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -10.5835  -1.0241   0.2889   1.4161   3.3750 

# Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  3.542593   0.374468   9.460   <2e-16 ***
#   youtube      0.045879   0.001399  32.804   <2e-16 ***
#   facebook     0.188062   0.008620  21.818   <2e-16 ***
#   newspaper   -0.001368   0.005877  -0.233    0.816    

#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 2.022 on 195 degrees of freedom
# Multiple R-squared:  0.8978,	Adjusted R-squared:  0.8962 
# F-statistic:   571 on 3 and 195 DF,  p-value: < 2.2e-16

# fitted values and the residuals
plot(fitted(loocv.fit), residuals(loocv.fit), main="Fitted values vs residuals")
# observed values and the predicted values
plot(marketing$sales, preds, xlab='actual', ylab='predicted', main="Observed values vs predicted")
plot(loocv.fit, main='Model summary')

# Residuals plots
library(ggplot2)
qqnorm(loocv.fit$residuals, main='Residuals Normal Q-Q Plot')
# plots of the variables: predictors and response
ggplot( marketing, aes( x = youtube, y = sales)) +
  geom_point() + 
  stat_smooth() + ggtitle("youtube vs sales")

ggplot( marketing, aes( x = facebook, y = sales)) +
  geom_point() + 
  stat_smooth() + ggtitle("facebook vs sales")

ggplot( marketing, aes( x = newspaper, y = sales)) +
  geom_point() + 
  stat_smooth() + ggtitle("newspaper vs sales")

# checking normality of the residuals: H0 : distribution is normal
r<-residuals(loocv.fit)
shapiro.test(r)
# Shapiro-Wilk normality test

# data:  r
# W = 0.91602, p-value = 3.215e-09


# Durbin-Watson test for the correlation of e_i vs e_i+1
library(lmtest)
dwtest(loocv.fit, alternative="two.sided")
# Durbin-Watson test

# data:  loocv.fit
# DW = 2.0742, p-value = 0.6003
# alternative hypothesis: true autocorrelation is not 0


########################################################
### 2. K-Fold cross validation method for K=10
#######################################################
set.seed(123)

# Number of k-fold
k = 10

# sample size in each fold
fold.size = n/k

# placeholder for storing the MSE values.
MSE.10Fold <- rep(0, k)
MSE.10Fold
# 2.582685 3.922308 4.244028 3.480843 3.148358 3.303171 3.366896 3.751054 9.426113 5.046522

# placeholder for storing RSE values
RSE.10Fold <- rep(0, k)
RSE.10Fold
# 1.694004 2.087611 2.171540 1.966622 1.870341 1.915774 1.934165 2.041528 3.236272 2.367963

#Randomly shuffle the data
randomized.data<-marketing[sample(nrow(marketing)),]

#Creating 10 equally size folds
folds <- cut(seq(1,nrow(randomized.data)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:k){
  
  #Segment the data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- randomized.data[testIndexes, ]
  trainData <- randomized.data[-testIndexes, ]
  
  # Training the model using the training dataset
  kfold.model <- lm(sales ~ youtube + facebook + newspaper, data = trainData)
  
  # making predictions using the validation set
  sales_pred <- predict(kfold.model, newdata = testData)
  
  MSE.10Fold[i] <- mean((testData$sales - sales_pred)^2)
  
  rss <- sum((testData$sales - sales_pred)^2)
  
  RSE.10Fold[i] <- sqrt(rss/(fold.size-2))
}

summary(kfold.model)
#Residuals:
#  Min       1Q   Median       3Q      Max 
#-10.8633  -1.0563   0.3144   1.4052   3.3777 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept) 3.578684   0.380220   9.412   <2e-16 ***
#  youtube     0.044361   0.001475  30.073   <2e-16 ***
#  facebook    0.192022   0.008927  21.510   <2e-16 ***
#  newspaper   0.004068   0.006235   0.652    0.515    
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 2.011 on 176 degrees of freedom
#Multiple R-squared:  0.9009,	Adjusted R-squared:  0.8993 
#F-statistic: 533.6 on 3 and 176 DF,  p-value: < 2.2e-16


# The average mean standard error for the 10-fold mean errors
mse.kfold <- (sum(MSE.10Fold))/k   # 4.227198

# The average residual standard error for the mean
rse.kfold <- (sum(RSE.10Fold))/k  # 2.128582
sigma(kfold.model) # Residual standard error: 2.011

# The average standard error of the mse
SE_10Fold <- sd(MSE.10Fold)/sqrt(k)   # 0.6149912

all.results <- data.frame(name = c("LOOCV","10-FOLD"),
                          mse=c(MSE.loocv,mse.kfold), rse=c(RSE,rse.kfold),
                          se=c(SE_Loocv, SE_10Fold))
#      name      mse      rse        se
# 1   LOOCV 4.243536 2.070362 0.7066973
# 2 10-FOLD 4.227198 2.128582 0.6149912

# III) Discussion of results above
# For both fitted models the F-statistics and the associated p-value in the
# model summaries is < 2.2e-16, which is highly significant. This means that, at
# least, one of the predictor variables is significantly related to the outcome
# variable.
# The p-value for newspaper in both models is >0.05 test statistic
# significant level, hence we can conclude that newspaper is not significant in
# predicting sales variable in the model. therefore, it can be removed from the 
# model.
# Comparing goodness-of-fit for both models to find the best model to predict sales;
# The R-Squared value for the model fitted using LOOCV method for data resampling 
# is 0.8978 and with 10-Fold method is 0.9009, this means that the model
# fitted with data resampled using 10-fold cv method will best fit the data and 
# predict the observed variable sales since it has a higher value of r-squared
# and also its MSE and standard error values are smaller compared to the other model.
# since a low mse and rse values the better the model fits the data
# Hence the model fitted using 10-fold cv method is the best linear regression model to
# predict sales given the advertising budget of youtube, facebook and newspaper 
# media outlets.


###########################################
#### QUESTION 4
###########################################

# Formulation of null hypothesis: H0
# p1 : probability to get the flu having received the flu shot
# p2 : probability to get the flu having not received the flu shot
# H0 : p1 = p2
# H1 : p1 > p2

# hypothesis test to use: Fisher's test
# this is because the variable of interest involved is proportional ie the
# number of people who got the flu when they had a flu shot and those who did not
# have a flu shot. Also since the two groups are mutually exclusive.


# Using the fisher's test method
data.shot <- matrix(c(3, 15, 10, 13), 2)
fisher.test(data.shot, alternative= "greater")
# Fisher's Exact Test for Count Data

# data:  data.shot
# p-value = 0.9868
# alternative hypothesis: true odds ratio is not equal to 1
# 95 percent confidence interval:
# 0.05274483        Inf
# sample estimates:
# odds ratio 
# 0.2686768

# comment: The p-value >0.05 test significance level hence we fail to reject the
# null hypothesis, the flu shot does not have an impact or effect on getting 
# the flu.

# Estimate p-value, p, from the sampling distribution (hypergeometric dist):
p <- vector(mode = "numeric", length = 18-3)

for (i in 3:18)
{
  p[i] <- dhyper(i, 13, 28, 18, log = FALSE)
}

# sum of the probabilities to get the p-value:
sum(p)   # 0.9868134

# comment: The p-value is same as to the one when using fisher's test which is 
# also greater than 0.05 test significant level. Hence accepting the null 
# hypothesis. Therefore getting the flu shot is not statistically significant.
# the proportions of the two groups do not significantly differ from each other.
