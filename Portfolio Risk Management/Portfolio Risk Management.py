# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:50:16 2018

@author: Lenovo
"""

# Import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fpath_csv = "C:\\Users\\Lenovo\\Desktop\\ML\\Finance-Assignment\\Portfolio Risk Management\\data\\MSFTPrices.csv"

# Read in the csv file
StockPrices = pd.read_csv(fpath_csv, parse_dates=['Date'])

# Ensure the prices are sorted by Date
StockPrices = StockPrices.sort_values(by='Date')

StockPrices['Returns'] = StockPrices['Open'].pct_change()
# Convert the decimal returns into percentage returns
percent_return = StockPrices['Returns']*100

# Drop the missing values
returns_plot = percent_return.dropna()

# Plot the returns histogram
plt.hist(returns_plot,bins=75)
plt.show()

# Calculate the average daily return of the stock
mean_return_daily = np.mean(StockPrices['Returns'])
print(mean_return_daily)

# Calculate the implied annualized average return
mean_return_annualized = ((1+mean_return_daily)**252)-1
print(mean_return_annualized)

# Calculate the standard deviation of daily return of the stock
sigma_daily = np.std(StockPrices['Returns'])
print(sigma_daily)

# Calculate the daily variance
variance_daily = sigma_daily**2
print(variance_daily)

# Annualize the standard deviation
sigma_annualized = sigma_daily*np.sqrt(252)
print(sigma_annualized)

# Calculate the annualized variance
variance_annualized = sigma_annualized**2
print(variance_annualized)

# Import skew from scipy.stats
from scipy.stats import skew

# Drop the missing values
clean_returns = StockPrices['Returns'].dropna()

# Calculate the third moment (skewness) of the returns distribution
returns_skewness = skew(clean_returns)
print(returns_skewness)

# Import stats from scipy
from scipy.stats import shapiro

# Run the Shapiro-Wilk test on the stock returns
shapiro_results = shapiro(clean_returns)
print("Shapiro results:", shapiro_results)

# Extract the p-value from the shapiro_results
p_value = shapiro_results[1]
print("P-value: ", p_value)

###############################################################################
'''Portfolio Composition'''

StockReturns = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\Finance-Assignment\\Portfolio Risk Management\\data\\Big9Returns2017.csv", parse_dates=['Date'])

# Finish defining the portfolio weights as a numpy array
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

StockReturns.index = pd.to_datetime(StockReturns['Date'])
StockReturns = StockReturns.drop(['Date'],axis=1)

# Calculate the weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Calculate the portfolio returns
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)

# Plot the cumulative portfolio returns over time
CumulativeReturns = ((1+StockReturns["Portfolio"]).cumprod()-1)
CumulativeReturns.plot()
plt.show()

# How many stocks are in your portfolio?
numstocks = 9

# Create an array of equal weights across all assets
portfolio_weights_ew = np.repeat(1/9,numstocks)

# Calculate the equally-weighted portfolio returns
StockReturns['Portfolio_EW'] = StockReturns.iloc[:,0:9].mul(portfolio_weights_ew, axis=1).sum(axis=1)

# Create an array of market capitalizations (in billions)
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])

# Calculate the market cap weights
mcap_weights = market_capitalizations/sum(market_capitalizations)

# Calculate the market cap weighted portfolio returns
StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(mcap_weights, axis=1).sum(axis=1)

StockReturns = StockReturns.drop(['Portfolio','Portfolio_MCap'],axis=1)
correlation_matrix = StockReturns.cov()
cov_mat_annual = correlation_matrix*252

# Import seaborn as sns
import seaborn as sns

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()

# Calculate the portfolio standard deviation
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
print(portfolio_volatility)

##############################################################################
'''Markowitz portfolio'''

RandomReturns = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\Finance-Assignment\\Portfolio Risk Management\\data\\EfficientFrontierPortfoliosSlim.csv")

# Risk free rate
risk_free = 0

# Calculate the Sharpe Ratio for each asset
RandomReturns['Sharpe'] = (RandomReturns['Returns'] - risk_free )/ RandomReturns['Volatility']

# Print the range of Sharpe ratios
print(RandomReturns['Sharpe'].describe()[['min', 'max']])

# Sort the portfolios by Sharpe ratio
sorted_portfolios = RandomReturns.sort_values(by=['Sharpe'], ascending=False)

'''Maximum sharpie ratio'''

# Extract the corresponding weights
MSR_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the MSR weights as a numpy array
MSR_weights_array = np.array(MSR_weights)

# Calculate the MSR portfolio returns
StockReturns['Portfolio_MSR'] = StockReturns.iloc[:, 0:numstocks].mul(MSR_weights_array, axis=1).sum(axis=1)

'''Global Minimum Volatility'''

# Sort the portfolios by volatility
sorted_portfolios = RandomReturns.sort_values(by=['Volatility'], ascending=True)

# Extract the corresponding weights
GMV_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the GMV weights as a numpy array
GMV_weights_array = np.array(GMV_weights)

# Calculate the GMV portfolio returns
StockReturns['Portfolio_GMV'] = StockReturns.iloc[:, 0:numstocks].mul(GMV_weights_array, axis=1).sum(axis=1)

###############################################################################

'''CAPM Model to determine beta'''

PortfolioReturns = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\ML\\Finance-Assignment\\Portfolio Risk Management\\data\\FamaFrenchFactors.csv")

# Risk free rate
risk_free = 0

# Calculate excess portfolio returns
PortfolioReturns['Portfolio_Excess'] = PortfolioReturns['Portfolio'] - PortfolioReturns['RF']

# Calculate the co-variance matrix between Portfolio_Excess and Market_Excess
covariance_matrix = PortfolioReturns[['Portfolio_Excess', 'Market_Excess']].cov()

# Extract the co-variance co-efficient
covariance_coefficient = covariance_matrix.iloc[0, 1]
print(covariance_coefficient)

# Calculate the benchmark variance
benchmark_variance = PortfolioReturns['Market_Excess'].var()
print(benchmark_variance)

# Calculating the portfolio market beta
portfolio_beta = covariance_coefficient/benchmark_variance
print(portfolio_beta)

# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
CAPM_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess', data=PortfolioReturns)

# Fit the regression and print adjusted r-squared
CAPM_fit = CAPM_model.fit()
print(CAPM_fit.rsquared_adj)

# Extract the beta
regression_beta = CAPM_fit.params[1]
print(regression_beta)

###############################################################################

'''Alpha and multifactor models'''

#Explains the variance better
#More risk more reward., going for smaller stocks instead of premium stocks

#Alpha is the error time + alpha - scale returns
# - alpha - weighter returns 
#alpha - missing components in the model

# Define the regression formula
PortfolioReturns_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML', data=PortfolioReturns)

# Fit the regression
PortfolioReturns_fit = PortfolioReturns_model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = PortfolioReturns_fit.rsquared_adj
print(regression_adj_rsq)

