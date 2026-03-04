# -----------------------------------------------------------
# Import necessary libraries
# -----------------------------------------------------------
import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt

# -----------------------------------------------------------
# Load dataset (from Wooldridge data package)
# -----------------------------------------------------------
smoke = woo.dataWoo('smoke')

# -----------------------------------------------------------
# STEP 1: Estimate OLS Model
# cigs = number of cigarettes smoked per day
# income, cigpric = income and cigarette price (logged)
# educ, age, age^2, restaurn = controls
# -----------------------------------------------------------
reg_ols = smf.ols(
    formula='cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke
)

results_ols = reg_ols.fit()

# Create results table
table_ols = pd.DataFrame({
    'b': round(results_ols.params, 4),
    'se': round(results_ols.bse, 4),
    't': round(results_ols.tvalues, 4),
    'pval': round(results_ols.pvalues, 4)
})

print(f'OLS Results:\n{table_ols}\n')


# -----------------------------------------------------------
# STEP 2: Breusch-Pagan Test for Heteroskedasticity
# H0: Homoskedasticity
# H1: Heteroskedasticity
# -----------------------------------------------------------
y, X = pt.dmatrices(
    'cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke,
    return_type='dataframe'
)

result_bp = sm.stats.diagnostic.het_breuschpagan(results_ols.resid, X)

bp_statistic = result_bp[0]
bp_pval = result_bp[1]

print(f'Breusch-Pagan Statistic: {bp_statistic}')
print(f'Breusch-Pagan p-value: {bp_pval}\n')


# -----------------------------------------------------------
# STEP 3: FGLS – Estimate the Variance Function
# Idea:
# If Var(u|X) = σ²h(X)
# then log(u^2) ≈ Xγ + error
# -----------------------------------------------------------

# Create log of squared residuals
smoke['logu2'] = np.log(results_ols.resid ** 2)

# Regress log(u^2) on explanatory variables
reg_fgls = smf.ols(
    formula='logu2 ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke
)

results_fgls = reg_fgls.fit()

table_fgls = pd.DataFrame({
    'b': round(results_fgls.params, 4),
    'se': round(results_fgls.bse, 4),
    't': round(results_fgls.tvalues, 4),
    'pval': round(results_fgls.pvalues, 4)
})

print(f'Variance Function (FGLS first step):\n{table_fgls}\n')


# -----------------------------------------------------------
# STEP 4: Construct Weights for WLS
# Estimated variance:
# Var(u|X) ≈ exp(fitted logu2)
# So weight = 1 / Var(u|X)
# -----------------------------------------------------------
wls_weight = 1 / np.exp(results_fgls.fittedvalues)


# -----------------------------------------------------------
# STEP 5: Estimate Weighted Least Squares (FGLS final step)
# -----------------------------------------------------------
reg_wls = smf.wls(
    formula='cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    weights=wls_weight,
    data=smoke
)

results_wls = reg_wls.fit()

table_wls = pd.DataFrame({
    'b': round(results_wls.params, 4),
    'se': round(results_wls.bse, 4),
    't': round(results_wls.tvalues, 4),
    'pval': round(results_wls.pvalues, 4)
})

print(f'FGLS (WLS) Results:\n{table_wls}\n')