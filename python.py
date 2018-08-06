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

