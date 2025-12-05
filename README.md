# Baseline Models

To begin with, we attempted several basic models simply based on our own understanding and reasonable conjectures about the financial market.

The **first part is a daily linear regression analysis**. 

It's expected that we will show that blindly modeling daily price movements with a Linear Regression is noisy.

While the **second part is a weekly analysis including a variety of random walks**. 

It's mainly due to the fact that the dataset contains a series of weekly data denoting the demand, supply, and other fundamental aspects of crude oil. Thus, there is a fairly solid motive for us to assume the **great inertia (or autocorrelation in statistical terms) and time dependency of crude oil price**. In that way, this week's oil price is likely to depend on previous weeks' situations, simulating a random walk.

---
# Findings based on the primary models

This set of results is classic and carries **profound financial significance**.

### Core Finding: High Market Efficiency (Long Live Random Walk King)
Note that Model 2 (Random Walk) defeated all competitors with an RMSE of 2.5835.

This validates the Efficient Market Hypothesis (EMH) within the weekly crude oil timeframe.

It implies that the current week's price already encapsulates almost all available information.

Trying to beat "simple inertia" using publicly available EIA supply/demand data (which everyone sees) is incredibly difficult. And this suggests that crude oil prices at this frequency are primarily driven by Trend and Momentum, rather than weekly micro-changes in inventory.

### The "Spurious Regression" Trap
Model 1 (Linear Regression) collapsed with a massive RMSE of 34.58.

This is, of course, a textbook level of Overfitting.

When one throws dozens of fundamental indicators (Inventory, Production, etc.) directly into a linear model to predict Absolute Price, the model forces itself to memorize every bit of noise in the training set.


Once applied to the testing set (unseen data), these memorized relationships completely break down.

### Time Value: Recency Bias (Newer is Better)
Model 3 (Smoothed RW) performed significantly worse (RMSE 2.80) than Model 2 (2.58).

Therefore, **smoothing (averaging) actually destroyed value**.

By averaging the last two weeks, one artificially introduced Lag.

In a volatile market like crude oil, "The Latest Information" (Current Price) is far more valuable than "Old Information" (Last Week's Price). While smoothing reduces noise, it also dilutes critical trend signals.

### Model 4's Convergence 
Model 4 (Calibrated RW) produced an RMSE (2.59) almost identical to Model 2 (2.58), differing by only 0.01.

It proves that Lasso worked exactly as intended—it likely compressed the coefficients of the vast majority of your fundamental features to zero.


The tiny 0.01 difference suggests that Lasso kept one or two very weak features that turned out to be noise in the test set, slightly dragging down performance.

---

# 1. Linear Regression (Baseline ML) 

This model attempts to predict the next day/week's price directly using a weighted sum of all available features (Inventory, Production, RSI, etc.) from the current day/week.

$$\hat{P}_{t+1} = \beta_0 + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \dots + \beta_n X_{n,t} + \epsilon$$

Where:

$\hat{P}_{t+1}$: Predicted Price for next day/week.

$X_{i,t}$: Value of feature $i$ (e.g., Inventory) at current day/week $t$.

$\beta_i$: Learned coefficients.

# 2. Random Walk (Naive Baseline)

This model assumes the market is efficient and the best predictor for next week's price is simply this week's price.$$\hat{P}_{t+1} = P_t$$

Where:

$P_t$: Actual Price at current week $t$.

# 3. Smoothed Random Walk (2-Week Average)

This model assumes the current price might contain noise, so it uses the average of the last two weeks as the prediction anchor.$$\hat{P}_{t+1} = \frac{P_t + P_{t-1}}{2}$$

# 4.  Supply/Demand Calibrated Random Walk (Augmented)

This model uses the Random Walk as a base ($P_t$) but "calibrates" it by predicting the percentage return ($\hat{r}_{t+1}$) based on changes in supply and demand fundamentals.$$\hat{r}_{t+1} = \alpha + \sum_{i=1}^{k} \lambda_i \Delta F_{i,t}$$

$$\hat{P}_{t+1} = P_t \times (1 + \hat{r}_{t+1})$$

Where:


$\hat{r}_{t+1}$: Predicted return for next week.


$\Delta F_{i,t}$: Percentage change in fundamental factor $i$ (e.g., % change in Inventory).

$\lambda_i$: Coefficients learned via Lasso Regularization (sparse selection).

