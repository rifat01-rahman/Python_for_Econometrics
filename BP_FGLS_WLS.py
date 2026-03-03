import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt

# Load data
smoke = woo.data('smoke')

# --------------------------
# 1️⃣ OLS
# --------------------------
reg_ols = smf.ols(
    formula='cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke
)

results_ols = reg_ols.fit()

table_ols = pd.DataFrame({
    'b': round(results_ols.params, 4),
    'se': round(results_ols.bse, 4),
    't': round(results_ols.tvalues, 4),
    'pval': round(results_ols.pvalues, 4)
})

print(f'table_ols:\n{table_ols}\n')


# --------------------------
# 2️⃣ Breusch–Pagan Test
# --------------------------
y, X = pt.dmatrices(
    'cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke,
    return_type='dataframe'
)

result_bp = sm.stats.diagnostic.het_breuschpagan(results_ols.resid, X)

bp_statistic = result_bp[0]
bp_pval = result_bp[1]

print(f'bp_statistic: {bp_statistic}')
print(f'bp_pval: {bp_pval}\n')


# --------------------------
# 3️⃣ Variance Function Estimation (FGLS Step 1)
# --------------------------
smoke['logu2'] = np.log(results_ols.resid**2)

reg_var = smf.ols(
    formula='logu2 ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke
)

results_var = reg_var.fit()

table_var = pd.DataFrame({
    'b': round(results_var.params, 4),
    'se': round(results_var.bse, 4),
    't': round(results_var.tvalues, 4),
    'pval': round(results_var.pvalues, 4)
})

print(f'table_variance_model:\n{table_var}\n')


# --------------------------
# 4️⃣ FGLS (WLS Step)
# --------------------------
wls_weight = 1 / np.exp(results_var.fittedvalues)

reg_wls = smf.wls(
    formula='cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn',
    data=smoke,
    weights=wls_weight
)

results_wls = reg_wls.fit()

table_wls = pd.DataFrame({
    'b': round(results_wls.params, 4),
    'se': round(results_wls.bse, 4),
    't': round(results_wls.tvalues, 4),
    'pval': round(results_wls.pvalues, 4)
})

print(f'table_wls:\n{table_wls}\n')