import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

# Load dataset
nyse = woo.dataWoo('nyse')

# Create return variable
nyse['ret'] = nyse['return']

# Create lagged return
nyse['ret_lag1'] = nyse['ret'].shift(1)

# Drop missing values created by lag
nyse = nyse.dropna()

# Linear regression model
reg = smf.ols(formula='ret ~ ret_lag1', data=nyse)
results = reg.fit()

# Squared residuals
nyse['resid_sq'] = results.resid ** 2

# Lag of squared residuals
nyse['resid_sq_lag1'] = nyse['resid_sq'].shift(1)

# Drop missing values again
nyse = nyse.dropna()

# ARCH(1) regression
ARCHreg = smf.ols(formula='resid_sq ~ resid_sq_lag1', data=nyse)
results_ARCH = ARCHreg.fit()

# Print regression table
table = pd.DataFrame({
    'b': round(results_ARCH.params, 4),
    'se': round(results_ARCH.bse, 4),
    't': round(results_ARCH.tvalues, 4),
    'pval': round(results_ARCH.pvalues, 4)
})

print(f'table:\n{table}\n')