### prediction using linear regression model 

**Goal:**
Fitting a multiple linear regression model to predict sales by using 10-fold
and LOOCV methods for data resampling i.e the partitioning the data into training and
testing subsets of the dataset. The equation of the model is of the form y= b0+b1x1+b2x2+b3x3
where y:response variable (sales), x1,x2 and x3 are the covariate variables i.e
youtube, facebook and newspaper respectively.

**Model performance:** The mean squared error (MSE) and its standard error were used to compare perfomance of the two models.

**Dataset:**
The data used was that of a marketing dataset (datarium package) which contains the impact of three advertising medias (youtube, facebook and newspaper) on sales. These are used 
for predicting sales units on the basis of the amount of money in thousands of dollars spent in the three advertising medias. Data are the advertising budget in thousands of dollars along with the sales. The advertising experiment has been repeated 200 times with different budgets and the observed sales have been 
# recorded.
