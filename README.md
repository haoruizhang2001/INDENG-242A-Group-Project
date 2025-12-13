

# WTI Crude Oil Prediction — Modeling Results Summary

## 1. Key Insights (Executive Summary)

1. **XGBoost is the strongest overall model**

   * Sharpe Ratio: **6.02** (vs. −1.36 for Linear Regression)
   * Annual Return: **3.40**
   * Best performance in both Bull and Bear regimes

2. **Fundamental supply–demand variables dominate predictability**

   * Top features come from **PADD3 refinery production** and **Alaska crude-in-transit inventories**.
   * These same variables were also selected by **Lasso**, confirming their stability.

3. **Residual (imbalance) features are highly predictive**

   * XGBoost ranks supply–demand residuals among the most important predictors.
   * These represent **non-linear regime-switching signals** that linear models cannot capture.

4. **XGBoost outperforms Buy-and-Hold**

   * Sharpe Ratio: **6.02 vs −0.16**
   * Remains profitable under transaction costs.

5. **Strong Bear Market robustness**

   * Sharpe Ratio (Bear): **6.29**
   * Max Drawdown: **−0.36**, significantly lower than linear baselines.

---

# 2. Overall Model Performance

| Model             | Accuracy | Precision | AUC-ROC | Sharpe Ratio | Annual Return | Max Drawdown | Volatility | Win Rate |
| ----------------- | -------- | --------- | ------- | ------------ | ------------- | ------------ | ---------- | -------- |
| XGBoost           | 0.592    | 0.574     | 0.623   | 6.02         | 3.40          | -0.43        | 0.565      | 0.476    |
| Smoothed RW       | 0.588    | 0.602     | 0.639   | 5.18         | 1.42          | -0.09        | 0.275      | 0.116    |
| Calibrated RW     | 0.488    | 0.488     | 0.500   | -0.93        | -0.48         | -0.56        | 0.618      | 0.512    |
| Linear Regression | 0.452    | 0.474     | 0.444   | -1.36        | -0.83         | -0.88        | 0.609      | 0.448    |
| Random Walk       | 0.500    | 0.500     | 0.500   | 0.00         | 0.00          | 0.00         | 0.000      | 0.000    |

### Observations

* XGBoost produces the **highest Sharpe and Annual Return**.

