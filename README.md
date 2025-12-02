# WARNING 
The notebook is run locally on Arthur's computer, so if you would like to reproduce the process, please change the file paths to the EIA files

# Required Packages

Before running this notebook, ensure you have the following packages installed:

```bash
%pip install yfinance pandas numpy xlrd matplotlib seaborn requests
```

**Package List:**
- `yfinance` - Download financial market data from Yahoo Finance
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `xlrd` - Reading Excel files (.xls format)
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `requests` - HTTP library for downloading files


---

## Overview
This notebook implements a data pipeline for predicting crude oil market movements using machine learning. We collect and engineer features from multiple market indicators including:
- **Energy commodities**: Crude oil (WTI & Brent), gasoline, heating oil, natural gas
- **Macro indicators**: US Dollar Index (DXY), US 10-Year Treasury, Oil Volatility Index
- **Cross-market signals**: Gold, copper, S&P 500, emerging markets
- **Sector-specific**: Energy stocks, transportation, oil services, credit risk

## Data Range & Considerations
- **Start Date**: 2021-01-01 (Post-COVID recovery period)
- **Frequency**: Daily
- **Rationale**: Avoids COVID-19 market disruption while capturing recent market dynamics

## Some Questions I was wondering about
- How far back should training data extend? Post-2020 data avoids pandemic anomalies but reduces sample size.
- Does adding more variables improve signal or introduce noise? Need to evaluate feature importance.

---

1. ✓ **Data Collection** - Downloaded 18 financial instruments from Yahoo Finance (2021-01-01 onwards)
2. ✓ **Feature Engineering** - Created 5 ratio features (Crack Spread, Gold/Oil, Copper/Oil, Transport/Oil, Services/Oil)
3. ✓ **Weekly Data Integration** - Merged 7 Excel files (15 sheets) of weekly fundamental data with daily resampling
4. ✓ **Data Quality Validation** - Comprehensive 9-step quality assessment
5. ✓ **Weekly Alignment Verification** - Validated forward-fill logic for weekly-to-daily conversion
6. ✓ **Data Cleaning** - Removed features with >50% missing values, zero variance, and infinite values
7. ✓ **Export** - Saved cleaned dataset to CSV

---

**Final Dataset:**
- Date Range: 2021-01-01 onwards
- Frequency: Daily (with forward-filled weekly fundamentals)
- Features: Price data + Engineered ratios + Weekly fundamentals
- Quality: Clean, validated, ready for modeling

---

**Next Steps:**
- Feature selection / dimensionality reduction
- Train/test split
- Model development (LSTM, Random Forest, etc.)
- Backtesting and validation

