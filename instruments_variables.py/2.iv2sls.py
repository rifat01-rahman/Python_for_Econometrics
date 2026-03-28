import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv
import statsmodels.formula.api as smf

# Load data
mroz = woo.dataWoo('mroz')

# Restrict to non-missing wage observations
mroz = mroz.dropna(subset=['lwage'])

# ---------------------------
# 1st Stage (Reduced Form)
# ---------------------------
reg_redf = smf.ols(
    formula='educ ~ exper + I(exper**2) + motheduc + fatheduc',
    data=mroz
)
results_redf = reg_redf.fit()

# Fitted values of educ
mroz['educ_fitted'] = results_redf.fittedvalues

# Print regression table
table_redf = pd.DataFrame({
    'b': round(results_redf.params, 4),
    'se': round(results_redf.bse, 4),
    't': round(results_redf.tvalues, 4),
    'pval': round(results_redf.pvalues, 4)
})
print("First Stage Results:\n", table_redf, "\n")


# ---------------------------
# 2nd Stage (Manual 2SLS)
# ---------------------------
reg_secstg = smf.ols(
    formula='np.log(wage) ~ educ_fitted + exper + I(exper**2)',
    data=mroz
)
results_secstg = reg_secstg.fit()

table_secstg = pd.DataFrame({
    'b': round(results_secstg.params, 4),
    'se': round(results_secstg.bse, 4),  # NOTE: not valid SE for 2SLS!
    't': round(results_secstg.tvalues, 4),
    'pval': round(results_secstg.pvalues, 4)
})
print("Second Stage (Manual) Results:\n", table_secstg, "\n")


# ---------------------------
# Proper IV (2SLS)
# ---------------------------
reg_iv = iv.IV2SLS.from_formula(
    formula='np.log(wage) ~ 1 + exper + I(exper**2) + [educ ~ motheduc + fatheduc]',
    data=mroz
)

results_iv = reg_iv.fit(cov_type='robust')  # better than 'unadjusted'

table_iv = pd.DataFrame({
    'b': round(results_iv.params, 4),
    'se': round(results_iv.std_errors, 4),
    't': round(results_iv.tstats, 4),
    'pval': round(results_iv.pvalues, 4)
})
print("IV (2SLS) Results:\n", table_iv)