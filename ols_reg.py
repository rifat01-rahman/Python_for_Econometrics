import wooldridge as woo
import statsmodels.formula.api as smf

# load dataset
wage1 = woo.data("wage1")

# OLS regression
reg = smf.ols(formula="wage ~ educ", data=wage1)
results = reg.fit()

# coefficients
b = results.params
print(f"b:\n{b}\n")