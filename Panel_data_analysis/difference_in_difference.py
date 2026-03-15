import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

# Load dataset
kielmc = woo.dataWoo('kielmc')

# Separate regressions for 1978
y78 = (kielmc['year'] == 1978)
reg78 = smf.ols(formula='rprice ~ nearinc', data=kielmc, subset=y78)
results78 = reg78.fit()

# Separate regressions for 1981
y81 = (kielmc['year'] == 1981)
reg81 = smf.ols(formula='rprice ~ nearinc', data=kielmc, subset=y81)
results81 = reg81.fit()

# Joint regression including interaction term
reg_joint = smf.ols(formula='rprice ~ nearinc * C(year)', data=kielmc)
results_joint = reg_joint.fit()

# Regression table for 1978
table_78 = pd.DataFrame({
    'b': round(results78.params, 4),
    'se': round(results78.bse, 4),
    't': round(results78.tvalues, 4),
    'pval': round(results78.pvalues, 4)
})
print(f'table_78:\n{table_78}\n')

# Regression table for 1981
table_81 = pd.DataFrame({
    'b': round(results81.params, 4),
    'se': round(results81.bse, 4),
    't': round(results81.tvalues, 4),
    'pval': round(results81.pvalues, 4)
})
print(f'table_81:\n{table_81}\n')

# Joint regression table
table_joint = pd.DataFrame({
    'b': round(results_joint.params, 4),
    'se': round(results_joint.bse, 4),
    't': round(results_joint.tvalues, 4),
    'pval': round(results_joint.pvalues, 4)
})
print(f'table_joint:\n{table_joint}\n')