
<img width="352" height="188" alt="Gemini_Generated_Image_8ik9ws8ik9ws8ik9" src="https://github.com/user-attachments/assets/4a3bd7de-2753-4da5-a8e0-643062aea46c" />

# Data Processing on EIA Weekly Variables

## Design Highlights

### 1. Resolution Alignment to Mitigate Spurious Correlation

To address the frequency mismatch between daily price action and weekly fundamental releases (EIA data), I rejected the common practice of naive forward-filling (up-sampling), which artificially inflates sample size ($N$) and induces severe serial correlation in the error term. Instead, I employed a pre-selection temporal aggregation strategy. By down-sampling the target variable to match the native weekly resolution of the covariates, I ensured that the feature selection algorithms (LASSO/Elastic Net) operated on statistically independent signals rather than autocorrelated noise, thereby preserving the integrity of the t-statistics and coefficient estimates.

### 2. Preservation of Temporal Causality (Out-of-Sample Validity)
Given the non-stationary nature of financial time series, standard $K$-fold cross-validation with random shuffling introduces significant look-ahead bias. My design enforces a strict chronological split (first 80% for training, subsequent 20% for testing). Furthermore, the standardization parameters ($\mu, \sigma$) were derived exclusively from the training window and applied forward to the test set. This ensures that the evaluation metrics ($MSE_{test}$, $R^2_{test}$) represent a mathematically honest estimate of the model's generalization error in a real-world forecasting micro-structure.

### 3. Robustness via Target Specification Sensitivity 
To differentiate between structural signal failure and target-specific noise, I designed a multi-horizon sensitivity analysis (Schemes A, B, and C). Rather than optimizing for a single metric, I tested the linear hypothesis across varying temporal lags (1-week vs. 2-week cumulative returns) and functional forms (continuous regression vs. binary classification). This approach isolates whether the explanatory power of fundamental variables decays over time or transforms into directional probability, providing a comprehensive audit of the linear relationship between supply/demand imbalances and price formation.

### 4. Ensemble Feature Selection in High-Dimensional ($p > n$) Space
Faced with a high-dimensional dataset ($p=406$) relative to a limited sample size ($n \approx 256$), coupled with inherent multicollinearity among inventory metrics, single-model selection is prone to instability. I implemented an Intersection-Based Selection Criterion. By retaining only the subset of features selected by both LASSO (which induces pure sparsity) and Elastic Net (which handles collinear grouping via the $L_2$ penalty), I filtered out mathematical artifacts. The resulting feature set represents the statistically robust "core drivers" that survive varying regularization topologies.

## Executive Summary

Based on the empirical results from Schemes A, B, and C, the direct linear prediction of crude oil price movements using only weekly fundamental data is statistically ineffective. However, the feature selection process successfully isolated economically meaningful variables.

### Performance Analysis (The "Null" Result)
- Predictive Failure: Both Scheme A (Next-Week Return) and Scheme B (2-Week Cumulative) resulted in negative $R^2$ scores on the test set (LASSO: -0.0079, Elastic Net: -0.138). This indicates that a simple historical mean would have been a better predictor than the complex linear models.

- Classification Failure: Scheme C (Binary Direction) achieved an accuracy of only 47%, which is slightly worse than a random coin toss.

- The "Efficiency" Reality: This confirms that weekly EIA data (Inventories, Production) is likely "priced in" by the market long before the official release, or that the linear relationship is overwhelmed by high-frequency daily noise.
### Feature Selection Insights (The "Signal")

Based on the comparative performance across Schemes A, B, and C, Scheme A utilizing LASSO Regression is the optimal model choice. While all linear models struggled to predict the magnitude of weekly price changes (as evidenced by negative $R^2$ values), LASSO demonstrated superior capability in noise reduction and feature identification within a high-dimensional dataset ($p \gg n$).

- **Model Performance**: Scheme A (LASSO) achieved the lowest Test MSE (19.65) and the highest relative $R^2$ (-0.0079), outperforming the cumulative trend prediction (Scheme B) and the binary classification model (Scheme C), which failed to select any features and achieved only 47% accuracy.
- **Dimensionality Reduction**: Unlike Elastic Net, which retained an excessive 135 features (suggesting overfitting), LASSO successfully imposed sparsity, narrowing 406 input variables down to just 4 core features.
- **Feature Interpretation**: The four selected variables possess strong economic logic, serving as proxies for Supply Chain Latency and Refining Demand. Specifically, PADD3 Refinery Net Production (Gasoline & Residual Fuel) and Alaska Transit Stocks represent physical constraints in the US oil market.
- **Strategic Application**: Consequently, these 4 features should not be used as direct trading signals (due to the negative $R^2$), but rather as Regime Indicators or State Variables to adjust the risk exposure of a higher-frequency daily model.
