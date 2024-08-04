#!/usr/bin/env python
# coding: utf-8

# # **Time Series Forecasting with ARIMA Model**

# ARIMA - Auto Regressive Integrated Moving Average
# 
# An ARIMA model is characterized by 3 terms: p, d, q
# <br><br>
# * **AR (p) Autoregression:** An auto regressive AR(p) term refers to the number of lags of Y to be used as predictors. In other words, the number of terms that are used from the past values in the time series for forecasting the future values. It's the part of the model that captures the relationship between a current observation and its predecessors. A high *p* value can indicate a strong influence of past values on current values.
# <br><br>
# * **I (d) Integration:** This parameter indicates the number of times the original time series data should be differenced to make it stationary. Stationarity is a critical aspect of time series analysis where the statistical properties of the series (mean, variance) do not change over time. Differencing is a method of transforming a non-stationary time series into a stationary one by subtracting the previous observation from the current observation. The *d* value helps in removing trends and seasonal structures from the time series, making it easier to model.
# <br><br>
# * **MA (q) Moving Average:**  ‘q’ is the order of the ‘Moving Average’ (MA) term. 'q' refers to the number of past errors (residuals) that are used to improve forecasts. The MA component models the relationship between an observation and a residual error from a moving average model applied to lagged observations. In simpler terms, it accounts for the influence of past prediction errors on the current observation. The errors referred to here are the differences between the actual values and the predictions that a simple moving average model would have made. The choice of *q* has a direct impact on how the ARIMA model predicts future values by considering how past errors influence current predictions. It helps in capturing the autocorrelation in the residuals of the series that is not explained by the autoregressive (AR) part.
# <br><br>
# 
# ### Model Types:
# 
# **ARIMA** - Non-seasonal auto regressive integrated moving average
# 
# <span style="color:crimson">**ARIMAX** - ARIMA with exogenous variable</span>
# 
# **SARIMA** - Seasonal ARIMA
# 
# **SARIMAX** - Seasonal ARIMA with exogenous variable

# In[ ]:





# In[ ]:


# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


#     

# In[ ]:


# load the dataset.
# This dataset contains the stock price for 'Apple' and the stock price for 'TXN'
data = pd.read_csv('AAPL.csv')
print(data.dtypes)


# The column `Date` is an 'object'. We need to change it to `datetime` data type.

# In[ ]:


# convert 'Date' to datetime type
data['Date'] = pd.to_datetime(data['Date'])


# In[ ]:


# check the datatype
print(data.dtypes)


# In[ ]:


data


# **AAPL** - Ticker symbol for Apple Inc. and it is traded on the NASDAQ stock exchange.<br>
# **TXN** - Ticker symbol for Texas Instruments (TXN): Major supplier for Apple known to supply Apple with several critical components used in Apple's products. One of the main components supplied by Texas Instruments to Apple is semiconductor chips, particularly for power management. These chips are crucial for managing battery life and power consumption in Apple's devices such as iPhones, iPads, and MacBooks.

# # Univariate - Using only one variable

# In[ ]:


# Univariate analysis - We will only use 'Apple' variable.
df = data.iloc[:-2,0:2]


# In[ ]:


# check the data
df.tail()


# In[ ]:


# set the 'Date' column as index
df = df.set_index('Date')


# In[ ]:


#create seaborn lineplot
plot = sns.lineplot(df['AAPL'])

#rotate x-axis labels
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


# ### Decomposition

# In[ ]:


# Extract and plot trend, seasonal and residuals.
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(df['AAPL'])


# In[ ]:


trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid


# In[ ]:


plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['AAPL'], label='Original', color='black')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='blue')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='black')
plt.legend(loc='upper left')
plt.show()


#     

# If the **residual plot** is a straight line at a non-zero value (in this case, 2), it indicates the presence of some systematic bias or pattern in the residuals, which means the model is not accounting for all the information in the data. This could be due to various reasons such as missing variables, incorrect model selection, or a non-linear relationship between the variables.

#     

# <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html>

# ## **Find p, d, q**

# ### To find p and q use ACF and PACF plots from statsmodels library

# In[ ]:


# for finding p,q using PACF and ACF plots respectively
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(7,4), 'figure.dpi':80})

# plot ACF
plot_acf(df['AAPL'].dropna());

# plot PACF
plot_pacf(df['AAPL'].dropna(), lags=11);

plt.show()


#     

# **q** is given by the ACF plot, where you count the number of spikes outside of the confidence interval (blue area). The first one which is 0, is the current value which will always be correlated to itself i.e. value is 1. In this case we only 1 spike outside the confidence interval. Therefore, `q = 1`
# 
# 
# **p** is given by the PACF plot, where you count the number of spikes outside of the confidence interval (blue area). The first one which is 0, is the current value which will always be correlated to itself i.e. value is 1. In this case we only have 1 spike outside the confidence interval. Therefore, `p = 1`
# That spike value in pacf (i.e. 0.76 are the coeficient values (beta1, beta2, etc.) in the linear regression equation)
# 
# We can use `q = 1` and `p = 1`

#     

# The key difference between the **Partial Autocorrelation Function (PACF)** and the **Autocorrelation Function (ACF)** is that the **PACF** measures the relationship between a time series and its lagged values after removing the effects of the intervening lags, while the **ACF** measures the relationship between a time series and its lagged values without removing the effects of the intervening lags.
# 
# For example, let's say we have a time series of monthly sales data for a store. If we are considering the correlation between sales in January and sales in March, then February is the intervening lag. The effect of the February sales on the correlation between January and March sales is accounted for when we use the PACF, as it controls for the effect of the intervening lags.

#     

# ### To find d, we have to first check if the series is stationary

# 
# 
# The first step to build an ARIMA model is to make the time series stationary.
# 
# Because, term ‘Auto Regressive’ in ARIMA means it is a linear regression model that uses its own lags as predictors. Linear regression models, work best when the predictors are not correlated and are independent of each other.
# 
# To make a series stationary, the most common approach is to difference (d) it. The value of `d`, therefore, is the minimum number of differencing needed to make the series stationary. And if the time series is already stationary, then d = 0.

# In[ ]:


# Augmented Dickey-Fuller (ADF) test to check if the data is stationary. We already know it is not by looking at the chart above.
from statsmodels.tsa.stattools import adfuller


# In[ ]:


# if p value < 0.05 the series is stationary
results = adfuller(df['AAPL'])
print('p-value:', results[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_


#     

# STEPS:
# * First check the p-value. If p-value is <0.05, your series is stationary.
# * If not, do a first order differencing
# * Perform the ADF test on the differenced series to check the p-value
# * If p-value is <0.05, your series is stationary
# * If not, do a second order differencing.
# * Whatever order of differencing you do, that will be your value of 'd' in ARIMA model
# 
# p-value is 0.99. Therefore, the series is **not stationary**. Therefore, we will need to make it stationary.

# ### Differencing - To make series stationary

# In[ ]:


# 1st order differencing
v1 = df['AAPL'].diff().dropna()

# adf test on the new series. if p value < 0.05 the series is stationary
results1 = adfuller(v1)
print('p-value:', results1[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_


# In[ ]:


# Plot the differenced series
plt.plot(v1)
plt.title('1st order differenced series')
plt.xlabel('Date')
plt.xticks(rotation=30)
plt.ylabel('Price (USD)')
plt.show()


# In[ ]:


# the mean for above series is
(v1.values).mean()


# The right order of differencing is the minimum differencing required to get a near-stationary series.

#     

# In[ ]:


# if reqired, 2nd order differencing would be
# v2 = v1.diff().dropna()
# results2 = adfuller(v2)
# print('p-value:', results2[1]) # adf, pvalue, usedlag_, nobs_, critical_values_, icbest_


#     

# ## **Training and Forecasting**

# In[ ]:


# Use statsmodels 0.11 or later
# to check the version after import use: print(statsmodels.__version__)
# to upgrade use: !pip install statsmodels --upgrade

# Install statsmodel
#!pip install statsmodels==0.13
import statsmodels


# In[ ]:


# Import ARIMA Model
from statsmodels.tsa.arima.model import ARIMA


# In[ ]:


# 1,1,1 ARIMA Model
arima = ARIMA(df.AAPL, order=(1,1,1))
ar_model = arima.fit()
print(ar_model.summary())


# ### 1. Coefficients:
# 
# **ar.L1 (AR coefficient at lag 1):** The coefficient value 0.8450 with a standard error 0.522 suggests the presence of an autoregressive term of order 1. However, the p-value (P>|z|) is 0.106, which is greater than the typical significance level of 0.05, indicating that this coefficient is not statistically significant. <br><br>
# **ma.L1 (MA coefficient at lag 1):** The coefficient value -0.9996 with a very large standard error 73.545 suggests a moving average effect. The p-value is very high (0.989), indicating that this coefficient is also not statistically significant.<br><br>
# **sigma2:** This is the estimated variance of the error term, with a value of 157.6766. The standard error of this estimate is quite large relative to the coefficient.<br><br>
# 
# 
# ### 2. Goodness of Fit:
# 
# **Log Likelihood:** The value -91.365 indicates the log-likelihood of the model, which is a measure of the fit of the model to the data. It’s used for comparing different models; higher values indicate a better fit.<br><br>
# **AIC (Akaike Information Criterion):** The value 188.731 is a measure used to compare models with a penalty for the number of parameters to prevent overfitting. Lower AIC values indicate a better model.<br><br>
# **BIC (Bayesian Information Criterion):** Similar to AIC with a stricter penalty for the number of parameters, the BIC value is 192.137. Again, lower is better.<br><br>
# **HQIC (Hannan-Quinn Information Criterion):** Another criterion for model selection, with a value of 189.588.
# 
# ### 3. Diagnostics:
# 
# **Ljung-Box (Q):** Tests for lack of fit. A high p-value (0.91) suggests that there is little evidence of lack of fit in the model.<br><br>
# **Jarque-Bera (JB):** Tests the assumption of normality of the residuals. A high p-value (0.52) suggests that the residuals are normally distributed.<br><br>
# **Heteroskedasticity (H):** Tests whether the variance of the errors is constant across observations. A high p-value (0.58) suggests no heteroskedasticity.<br><br>
# 
# 
# ### 4. Interpretation:
# 
# * The model coefficients are not statistically significant (based on the p-values), which may imply that the model might not be the best fit for this data.
# 
# * The model diagnostics suggest that the residuals do not violate the assumptions of normality and constant variance, which is good. However, the model's predictive accuracy is in question due to the coefficient significance.
# 
# * The very large standard error for the MA coefficient suggests that there may be issues with the data or the model's specification.
# 
# Notice here the coefficient of the MA.L1 and MA.L2 term is close to zero and the P-Value in ‘P>|z|’ column is highly insignificant. It should ideally be less than 0.05 for the respective X to be significant.
# 
# Insignificant p-values imply that the autoregressive and moving average terms are not statistically significant and therefore should be removed from the model.
# 
# Alternatively, it could mean that the model needs to be refined by including additional variables or using a different order of differencing or a higher order ARIMA model.
# 
# It's also possible that the data itself may not be suitable for modeling using an ARIMA model, and a different approach may be needed to model the time series.

# In[ ]:


# Forecast
forecast = ar_model.get_forecast(2)
ypred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)


# In[ ]:


ypred


# #### Check the actual values on Yahoo Finance
# 
# https://finance.yahoo.com/quote/AAPL/history?period1=1640908800&period2=1707868800&interval=1mo&filter=history&frequency=1mo&includeAdjustedClose=true

# In[ ]:


# creating a new Dataframe dp with the prediction values.
Date = pd.Series(['2024-01-01', '2024-02-01'])
price_actual = pd.Series(['184.40','185.04'])
price_predicted = pd.Series(ypred.values)
lower_int = pd.Series(conf_int['lower AAPL'].values)
upper_int = upper_series = pd.Series(conf_int['upper AAPL'].values)

dp = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index =['Date','price_actual', 'lower_int', 'price_predicted', 'upper_int']).T
dp = dp.set_index('Date')
dp.index = pd.to_datetime(dp.index)
dp


# In[ ]:


data = data.set_index('Date')


# In[ ]:


# Plot
plt.plot(data.AAPL)
plt.plot(dp.price_predicted, color='orange')
plt.fill_between(dp.index,
                 lower_int,
                 upper_int,
                 color='k', alpha=.15)


plt.title('Model Performance')
plt.legend(['Actual','Prediction'], loc='lower right')
plt.xlabel('Date')
plt.xticks(rotation=30)
plt.ylabel('Price (USD)')
plt.show()


# In[ ]:


# import evaluation matrices
from sklearn.metrics import mean_absolute_error

# Evaluate the model
print('ARIMA MAE = ', mean_absolute_error(dp.price_actual, dp.price_predicted))


#     

# # Bivariate - Using the exogenous variable - **'TXN'**

# In[ ]:


# Bivariate analysis - We will use both 'Apple' and 'TexasIns' variables.
# Training data - January 2020 to January 2022
# Testing data - February 2022 - May 2022
dfx = data.iloc[0:-2,0:3]


# In[ ]:


dfx


# In[ ]:


# 1,1,1 ARIMAX Model with exogenous variable
model2 = ARIMA(dfx.AAPL, exog=dfx.TXN, order=(1,1,1))
arimax = model2.fit()
print(arimax.summary())


# In[ ]:


# Forecast
forecast = arimax.get_forecast(2)
ypred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)


# In[ ]:


data.tail()


# In[ ]:


# we have the values for TXN
data.TXN.iloc[-2:]


# In[ ]:


# put those values in a variable
ex = data.TXN.iloc[-2:].values
ex


# In[ ]:


# Forecast
forecast = arimax.get_forecast(2, exog=ex)
ypred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)


# In[ ]:


# creating a new Dataframe dp with the prediction values.
Year = pd.Series(['2024-01-01', '2024-02-01'])
price_actual = pd.Series(['184.40','185.04'])
price_predicted = pd.Series(ypred.values)
lower_int = pd.Series(conf_int['lower AAPL'].values)
upper_int = upper_series = pd.Series(conf_int['upper AAPL'].values)


dpx = pd.DataFrame([Date, price_actual, lower_int, price_predicted, upper_int], index =['Date','price_actual', 'lower_int','price_predicted','upper_int' ]).T
dpx = dpx.set_index('Date')
dpx.index = pd.to_datetime(dpx.index)
dpx


# In[ ]:


# Plot
plt.plot(data.AAPL)
plt.plot(dpx.price_predicted, color='orange')
plt.fill_between(dpx.index,
                 lower_int,
                 upper_int,
                 color='k', alpha=.15)
plt.title('Model Performance')
plt.legend(['Actual','Prediction'], loc='lower right')
plt.xlabel('Date')
plt.xticks(rotation=30)
plt.ylabel('Price (USD)')
plt.show()


# In[ ]:


# import evaluation matrices
from sklearn.metrics import mean_absolute_error

# Evaluate the model
print('ARIMAX MAE = ', mean_absolute_error(dpx.price_actual, dpx.price_predicted))


#   

# For further learning on time series models:<br>
# 
# * https://www.machinelearningplus.com/time-series/time-series-analysis-python/<br>
# * https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# 

# # Preparaing Data for ML Models

# In[ ]:


import yfinance as yf


# In[ ]:


data = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
data.head()


# In[ ]:


data['Next_day'] = data['Close'].shift(-1)
data.head()


# #### Lets make this a classification task

# In[ ]:


# If the price next day was greater than previous day, we flag it as 1
data['Target'] = (data['Next_day'] > data['Close']).astype(int)
data.head()


# In[ ]:


# Let's check the trend
from matplotlib import pyplot as plt
data['Close'].plot(kind='line', figsize=(8, 4), title='line Plot')
plt.gca().spines[['top', 'right']].set_visible(False)


# ### Let's train a XGBoost Model

# In[ ]:


# Train test split. Note, this is a time series data.
train = data.iloc[:-30]
test = data.iloc[-30:]

# Be carefull not to use the next_day feature
features = ['Open', 'High', 'Low', 'Close', 'Volume']


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


# Instantiate a model
model1 = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)


# In[ ]:


# Train the baseline model
model1.fit(train[features], train['Target'])


# In[ ]:


# Make predictions
model1_preds = model1.predict(test[features])

# Convert numpy array to pandas series
model1_preds = pd.Series(model1_preds, index=test.index)


# In[ ]:


#model1_preds


# In[ ]:


# Evaluate the model
from sklearn.metrics import precision_score
precision_score(test['Target'], model1_preds)


# In[ ]:


# Plot test['Target'] vs model1_preds
plt.plot(test['Target'], label='Actual')
plt.plot(model1_preds, label='Predicted')
plt.legend()
plt.show()


# ## Create a backtesting function

# In[ ]:


# First create a predict function
def predict(train, test, features, model):
  model.fit(train[features], train['Target'])
  model_preds = model.predict(test[features])
  model_preds = pd.Series(model_preds, index=test.index, name='predictions')
  combine = pd.concat([test['Target'], model_preds], axis=1)
  return combine


# In[ ]:


#data.head()


# In[ ]:


x = data.loc['2000-01-01':'2019-12-31']
x.shape


# In[ ]:


r = data.loc['2000-01-01':'2000-01-31']
r.shape


# In[ ]:


# Create a backtest function
def backtest(data, model, features, start=5031, step=120):
  all_predictions = []

  for i in range(start, data.shape[0], step):
    train = data.iloc[:i].copy()
    test = data.iloc[i:(i+step)].copy()
    model_preds = predict(train, test, features, model)
    all_predictions.append(model_preds)

  return pd.concat(all_predictions)


# In[ ]:


# backtest
predictions = backtest(data, model1, features)


# In[ ]:


predictions


# In[ ]:


#Evaluate the model
precision_score(predictions['Target'], predictions['predictions'])


# In this example, we demonstrated the basic application of ARIMA and XGBoost models for stock price prediction using the APPL stock dataset with open, close, and volume features.
# 
# While our initial results may not have achieved high accuracy, this is not unexpected. Stock market predictions are notoriously challenging, and our simple model is just a starting point.
# 
# To improve the accuracy of our predictions, incorporate additional relevant features, such as technical indicators (e.g., moving averages, RSI), sentiment analysis, or economic indicators (e.g., GDP, inflation rate), relevant stock data.

# In[ ]:




