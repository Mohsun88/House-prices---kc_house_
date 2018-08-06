#Prediction of house prices

# Here I will try to analyse the features that impact house prices and build some models to predict the prices. I will use RMSE (root mean squared error) to measure the performance of the models.
# Lets importing some libraries 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # for plotting 
import pandas as pd # for manipulating datasets
import seaborn as sb
from pylab import rcParams
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter

%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# Laod the dataset
df = pd.read_csv('../input/kc_house_data.csv')

df.head()

# Lets see the some important stats
df.describe()

df.info()

#df correlation matrix shows relations between all variables in the dataset  
f,ax = plt.subplots(figsize=(17, 14))
sb.heatmap(df.corr(), annot=True,annot_kws={'size': 12}, linewidths=.5, fmt='.2f', ax=ax)

# Lets select the main variable (important features) and review the insigts by using pairplots 
sb.set()
cols = df[['price','sqft_living','grade','sqft_above','bathrooms','sqft_living15']]
sb.pairplot(cols, size = 2.5)

# Correlation matrix between the target value (price) and important predictors.
k = 10 #number of variables for heatmap
corrmat = df.corr()
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(df[cols].values.T)
sb.set(font_scale=1.25)
f,ax = plt.subplots(figsize=(17, 14))
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', ax=ax, annot_kws={'size': 12}, linewidths=.7, yticklabels=cols.values, xticklabels=cols.values)

#scatter plot 'sqft_living'/'price'
var = 'sqft_living'
data = pd.concat([df['price'], df[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(0,8000000));
# there is a strong relationship between 'sqft_living' and 'price'

#boxplot 'grade'/'price'
var1 = 'grade'
data = pd.concat([df['price'], df[var1]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sb.boxplot(x=var1, y="price", data=data)
fig.axis(ymin=0, ymax=8000000);

#scatter plot 'sqft_living15'/'price'
var2 = 'sqft_living15'
data = pd.concat([df['price'], df[var2]], axis=1)
data.plot.scatter(x=var2, y='price', ylim=(0,8000000));
# There is a positive relationship between 'sqft_living15' and 'price' but not as well as 'sqft_living' 

#scatter plot 'sqft_above'/'price'
var3 = 'sqft_above'
data = pd.concat([df['price'], df[var3]], axis=1)
data.plot.scatter(x=var3, y='price', ylim=(0,8000000));

var4 = 'bathrooms'
data = pd.concat([df['price'], df[var4]], axis=1)
data.plot.scatter(x=var4, y='price', ylim=(0,8000000));

var5 = 'view'
data = pd.concat([df['price'], df[var5]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sb.boxplot(x=var5, y="price", data=data)
fig.axis(ymin=0, ymax=8000000)

# ratios
corrmat['price'].sort_values(ascending = False)

X = df[[var,var1,var2,var3,var4,var5,'sqft_basement','bedrooms','lat','waterfront','floors','yr_renovated']]
y = df['price']

from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 1)

# Standardizing makes our data distributed normal and it is very usefull for a good prediction
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)

# Fitting Multiple Linear Regression to the Training set
LinReg = LinearRegression()
LinReg.fit(X_train,y_train)
y_predict = LinReg.predict(X_test)

LinReg.score(X_train,y_train) # R squared 

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std()
          
          
lin_scores = cross_val_score(LinReg, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
          
from sklearn.metrics import mean_squared_error

housing_predictions = LinReg.predict(X_train)
lin_mse = mean_squared_error(y_train, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

          
          
  
