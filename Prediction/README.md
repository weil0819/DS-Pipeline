# Pipeline for Prediction 

## Time Series Prediction
Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values.  

Time series are widely used for non-stationary data, like economic, weather, stock price, and retail sales in this post.  

### Models: 
1. Logistic Regression
2. (Multinomial) Naive Bayes
3. Linear Support Vector Machine
4. Random Forest
5. Keras

### Libraries:
#### Math Libraries:
- `import pandas as pd`
- `import numpy as np`
- `from math import sqrt`
- `from statsmodels.distributions.empirical_distribution import ECDF`

#### Plot Libraries:
- `import seaborn as sns`
- `import matplotlib.pyplot as plt`
- `import plotly.express as px `
- `import warnings`
- `warnings.filterwarnings("ignore")`

#### Feature Extraction/Selection:
- `from sklearn.feature_selection import chi2`
- `from sklearn.feature_extraction.text import CountVectorizer`
- `from sklearn.feature_extraction.text import TfidfVectorizer`

#### Model Selection:
- `from sklearn.pipeline import Pipeline`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.model_selection import cross_val_score`
- `from sklearn.naive_bayes import MultinomialNB`
- `from sklearn.linear_model import LogisticRegression`
- `from sklearn.ensemble import RandomForestClassifier`
- `from sklearn.svm import LinearSVC`

#### Metric Libraries:
- `from sklearn.metrics import confusion_matrix`
- `from sklearn.metrics import classification_report`
- `from sklearn.metrics import accuracy_score`
- `from sklearn.metrics import mean_squared_error`

#### Time Series Libraries:
- `from fbprophet import Prophet`


### Ref:
> Sales Forecasting with Price & Promotion effects  
https://towardsdatascience.com/sales-forecasting-with-price-promotion-effects-b5d70207b128

> Time Series Analysis, Visualization & Forecasting with LSTM  
https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba

> An End-to-End Project on Time Series Analysis and Forecasting with Python  
https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b  

> A Guide For Time Series Prediction Using Recurrent Neural Networks (LSTMs)  
https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f  

================================================================================ 

## Price Prediction


### Ref
> Modeling Price with Regularized Linear Model & Xgboost  
https://towardsdatascience.com/modeling-price-with-regularized-linear-model-xgboost-55e59eae4482

> How Taxis Arrive at Fares? — Predicting New York City Yellow Cab Fares  
https://towardsdatascience.com/how-taxis-arrive-at-fares-predicting-new-york-city-yellow-cab-fares-71a8c43b7c50 


## Ad Prediction

### Ref
> Ad Demand Forecast with Catboost & LightGBM  
https://towardsdatascience.com/ad-demand-forecast-with-catboost-lightgbm-819e5073cd3e

> Mobile Ads Click-Through Rate (CTR) Prediction  
https://towardsdatascience.com/mobile-ads-click-through-rate-ctr-prediction-44fdac40c6ff

## Booking Prediction

### Ref
> Predict Where a New User Will Book Their First Travel Experience  
https://towardsdatascience.com/predict-where-a-new-user-will-book-their-first-travel-experience-e6c9ada67cf4 

> Predicting Hotel Bookings with User Search Parameters  
https://towardsdatascience.com/predicting-hotel-bookings-with-user-search-parameters-8c570ab24805  

## Search Prediction

### Ref
> Predict Search Relevance Using Machine Learning for Online Retailers  
https://towardsdatascience.com/predict-search-relevance-using-machine-learning-for-online-retailers-5d3e47acaa33  

> Machine Learning Model for Predicting Click-Through in Hotel Online Ranking  
https://towardsdatascience.com/machine-learning-model-for-predicting-click-through-in-hotel-online-ranking-d55fc18c8516  




