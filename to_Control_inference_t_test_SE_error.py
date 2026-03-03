import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
# reg.fit(cov_type=’nonrobust’) or reg.fit() for the default homoscedasticity-based standard errors.
# reg.fit(cov_type=’HC0’) for the classical version of White’s robust variance-covariance matrix presented by Wooldridge
# reg.fit(cov_type=’HC1’) for a version of White’s robust variance-covariance matrix corrected by degrees of freedom.
# reg.fit(cov_type=’HC2’) for a version with a small sample correction. This is the default behavior of Stata.
# reg.fit(cov_type=’HC3’) for the refined version of White’s robust variance-covariance matrix.
# load dataset
gpa3 = woo.data('gpa3')

# define regression model (only spring semester)
reg = smf.ols(
    formula='cumgpa ~ sat + hsperc + tothrs + female + black + white',
    data=gpa3,
    subset=(gpa3['spring'] == 1)
)

# default OLS (homoskedastic SE)
results_default = reg.fit()
table_default = pd.DataFrame({
    'b': round(results_default.params, 5),
    'se': round(results_default.bse, 5),
    't': round(results_default.tvalues, 5),
    'pval': round(results_default.pvalues, 5)
})
print(f'table_default:\n{table_default}\n')

# White robust SE (HC0)
results_white = reg.fit(cov_type='HC0')
table_white = pd.DataFrame({
    'b': round(results_white.params, 5),
    'se': round(results_white.bse, 5),
    't': round(results_white.tvalues, 5),
    'pval': round(results_white.pvalues, 5)
})
print(f'table_white:\n{table_white}\n')

# Refined White SE (HC3)
results_refined = reg.fit(cov_type='HC3')
table_refined = pd.DataFrame({
    'b': round(results_refined.params, 5),
    'se': round(results_refined.bse, 5),
    't': round(results_refined.tvalues, 5),
    'pval': round(results_refined.pvalues, 5)
})
print(f'table_refined:\n{table_refined}\n')