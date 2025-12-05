
<img width="352" height="188" alt="Gemini_Generated_Image_8ik9ws8ik9ws8ik9" src="https://github.com/user-attachments/assets/4a3bd7de-2753-4da5-a8e0-643062aea46c" />


# Meeting notes - Dec.5th
- Baseline models
	- Linear Regression (without feature selection)
	- Random Walk Model
- Processing data
  - 4 Datasets:
  - PCA, Lasso, Elastic net,  Original Dataset
- Advanced models
 	- Lasso regression
 	- Random forest
 	- Boosting (could check XGBoost, which is a version of boosting more useful for quant finance)
 	- ARIMA
	 - Neural Network (?)

---
# Meeting Notes - Dec. 4th

11 a.m. Section Gathering

"If you can do the data pipeline plus one well-designed linear vs tree-based comparison with proper backtesting, you’ll already have a very strong project. Anything beyond that is a bonus."

Everyone works on their own ideas:
- Wish: PCA vs. linear/feature selection by hand
- Monthly frequencies v.s. daily frequencies
- Baseline model
- Forest
  - Boosting
- ARIMA
- XGBoost???
- Factors (based on the financial knowledge)


  


# INDENG242A Group Project Outline
Goal: comparative study of different ML methods in a specific area of the financial market

Possible subgoal: find a more nuanced algorithm for arbitrage/making profits, tailored towards the specific kind of market/futures/options/stock.

Discussion we want to touch upon:
1. Effectiveness of different ML methods
2. Rethinking the performance metrics
  	- Conjecture: high accuracy does not directly translate to good returns in portfolios sometimes
3. Connect the model's findings with mathematical/financial principles.
	- Conjecture: We are likely to discuss some rules derived by Boosting/RF and see how they can be connected with math principles. When some variables are extremely high, the previous criteria become ineffective.
4. Application insights: data collection time v.s. quality
   	- One interesting thing to do is to check if the low-quality/less precise/less sensitive data could bring up a fairly similar model with one based on a high-quality model.


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
- Binance API
- Alphavantage


### Tentative Research Plan

#### Abstract
This study aims to construct and compare systematic trading strategies for WTI Crude Oil (CL) futures. The research investigates whether non-linear machine learning models (XGBoost, LSTM) can generate superior risk-adjusted returns (Sharpe Ratio) compared to traditional linear benchmarks (Lasso Regression) by effectively integrating Term Structure signals with fundamental oil-market specific variables (Crack Spreads, Inventory Data, Macro-economic data).


#### Target variables

Essentially, we believe the majority of methods learnt in the course could be somewhat applied. As long as the predicted probabilities could be generated, the signals of buying/selling/holding could be set up through a certain numeric threshold, and furthermore, one could design nuanced strategies (Position sizing corresponding to the probabilities, for example)

Therefore, we will try abusing the models to produce a binary variable such that
$Y_{t} = 1$ if $\text{Return}_{t+5} > 0$ else $0$


#### Models
- Benchmark Model: Represents the traditional econometric approach, assuming linear relationships between fundamental factors and oil returns.
  	- Simple linear regression
  	- Lasso Regression.
  	- ARIMA
  	- Logistic regression
- Funny Model: Attempts to find a high-dimensional space to separate the time period when going upwards and downwards.
  	- KNN
  	- SVM
- Main Model (Tree-Based): Chosen for its ability to capture regime-switching behavior (e.g., "Inventory data matters more when Volatility is low").
  	- Random Forest
  	- Boosting (XGBoost specifically).
- Challenger Model (We expect them to be more powerful, plus they have cool names):  (MLP?). Depends on our workload.
  	- Neural Network
  	- MLP
	- LSTM



#### Features
There will be 4 big parts of features:
1. Term Structure / Carry
   - Roll yield
   - Curve slope
   - Curve curvature
2. Oil fundamentals
   - Crack spread ($P_{Gasoline} - P_{crude}$)
   - Inventory Change (EIA Crude Oil Stocks Change, need forward fill)
   - Open interest change ($\frac{OI_t - OI_{t-1}}{OI_{t-1}}$)
3. Technical & Momentum
   	- Time Series Momentum
   	- Vol_Adjusted_Mom$\frac{\text{Return}}{\text{Volatility}}$
   	- RSI
   	- Basis Momentum
4. Macro & Sentiment
   	- DXY Return
   	- US10Y
   	- OVX

There are definitely a lot more interesting and arguably reasonable data sources, but we also have to be wise about the price–performance ratio; if they are highly correlated with other variables, we definitely don't want to spend days wrangling data.

While that also leads to another interesting question: would a proxy variable (correlating with a lot) outperform a more accurate/arguably more powerful, specific variable in the model?

#### Backtesting
- Walk-Forward Validation: Rolling window training (e.g., 3-year train, 1-year test) to prevent look-ahead bias.
- Scenario analysis: Carefully select bullish/bearish time periods to prevent feeding models with overly optimistic/pessimistic information. 
- Transaction Costs: Simulating realistic slippage and commission fees typical for retail/institutional futures trading.

#### Performance metrics
The model is not only about predictions, but it's more about profiting. That leads to 2 dimensions of metrics we have to check:
1. Statistical / ML Metrics
- Precision
  	- Primary indicator: one doesn't have to capture every increase, but definitely needs to make a good shot when making the decisions. 
- Accuracy
  - Be careful about that, and we have to set a Benchmark Accuracy. If the stock market is in a bull scenario, blindly guessing a skyrocketing price will generate good accuracy.
- Log-Loss
  - High accuracy but low confidence is still not good. 
- AUC-ROC
2. Financial / Strategy Metrics
- Risk-Adjusted Returns
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
- Absolute Risk
  - Max Drawdown
  - Volatility
- Execution Metrics (advanced)
  - Win Rate vs. Profit/Loss Ratio
  - Turnover Rate

#### Basic To-Do List

1. Settle the market (the current structure of the tentative plan can be transferred into any market).
2. Settle the time period (Different datasets have different units of time, and this may affect the quality of training)
3. Collecting data 
- Time-consuming work (each guy focuses on different stuff)
- Need to be careful about the units of time
- Could look up some interesting variables/ design some interesting variables (say, network variable across different sections of supply chain)
4. Literature review on the application of ML to financial markets.
  - Performance metrics?
  - Which model works best in what situations?
  - How to really apply them? How do they form the execution decisions, really?
