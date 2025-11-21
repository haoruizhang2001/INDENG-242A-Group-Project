# INDENG-242A-Group-Project
Goal: comparative study of different ML methods in a specific area of the financial market
Possible subgoal: find a more nuanced algorithm for arbitrage/making profits, tailored towards the specific kind of market/futures/options/stock.

Discussion we want to touch upon:
1. Effectiveness of different ML methods
2. Rethinking the performance metrics
  	- Conjecture: high accuracy does not directly translate to good returns in portfolios sometimes


## Markets Suggested
- Oil Market (Crude oil CL=F)
	- Structural, systematic, easy-to-understand, highly connected with macro-economy
- Crypto
	- Full of data sources, subject to sentiment and many other variables one could call
- Beans/Corn (ZS, ZC)
	- Extremely seasonal, can discuss stationarity/seasonality stuff

## Data sources
- yfiance
- quantnet
- Xianyu
- Binance API
- Alphavantage


### Tentative Research Plan

#### Abstract
This study aims to construct and compare systematic trading strategies for WTI Crude Oil (CL) futures. The research investigates whether non-linear machine learning models (XGBoost, LSTM) can generate superior risk-adjusted returns (Sharpe Ratio) compared to traditional linear benchmarks (Lasso Regression) by effectively integrating Term Structure signals with fundamental oil-market specific variables (Crack Spreads, Inventory Data).
#### Details
- We estimate whether the price will go up (1) or go down (0).
- Model
  	- Benchmark Model (Linear): Lasso Regression. Represents the traditional econometric approach, assuming linear relationships between fundamental factors and oil returns.
  	- Main Model (Tree-Based): RF and Boosting (XGBoost specifically). Chosen for its ability to capture regime-switching behavior (e.g., "Inventory data matters more when Volatility is low").
  	- Challenger Model?? : NN, LSTM, etc.? We choose.

- Features:
	- Time-Series Momentum: Returns over the past 1-month, 3-months, and 12-months.
	- Volatility Normalized Momentum: Return divided by the standard deviation of returns (Sharpe-like metric).
		- Logic: A positive carry signal is stronger if the price trend is also Up.
	- DXY (Dollar Index): Correlation with commodities is generally negative.
	- US10Y (10-Year Treasury Yield): Represents the opportunity cost of money and inflation expectations.
		- Logic: Commodities often struggle when the Dollar is strengthening rapidly.








--


Baseline model: ARIMA
Linear regression (PCA)
RF/Gradient boosting
Neural Network
Time series model

Novelty:
Neural network
specific/useful factors in the regression
240 project (optimization method)
room for discussion and exploration

Final results:
talk about the effectiveness(accuracy)/ mechanism (why this work/why not)
discussion engaging with the financial knowledge

Steps/Workload

—> Choose topic! Financial market?
1. data (raw data, control variable, proxy variables, others)
	→ Big shitty work
2. Baseline model construction
	→ algorithm
NN, Factors
Disucssion
regarding ML methods effectiveness
discussion related to finance
