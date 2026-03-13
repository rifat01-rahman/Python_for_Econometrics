import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load dataset
prminwge = woo.dataWoo('prminwge')

# Number of observations
T = len(prminwge)

# Create time trend variable
prminwge['time'] = prminwge['year'] - 1949

# Set time-series index (yearly starting 1950)
prminwge.index = pd.date_range(start='1950', periods=T, freq='YE').year

# OLS regression model
reg = smf.ols(
    formula='np.log(prepop) ~ np.log(mincov) + np.log(prgnp) + np.log(usgnp) + time',
    data=prminwge
)

# Results with regular standard errors
results_regu = reg.fit()

# Create regression table
table_regu = pd.DataFrame({
    'b': round(results_regu.params, 4),
    'se': round(results_regu.bse, 4),
    't': round(results_regu.tvalues, 4),
    'pval': round(results_regu.pvalues, 4)
})

print(f'table_regu:\n{table_regu}\n')

# Results with HAC (Newey–West) standard errors
results_hac = reg.fit(cov_type='HAC', cov_kwds={'maxlags': 2})

# Create regression table
table_hac = pd.DataFrame({
    'b': round(results_hac.params, 4),
    'se': round(results_hac.bse, 4),
    't': round(results_hac.tvalues, 4),
    'pval': round(results_hac.pvalues, 4)
})

print(f'table_hac:\n{table_hac}\n')