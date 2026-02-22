import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

# load dataset
wage1 = woo.data("wage1")

# log-wage regression
reg = smf.ols(formula="np.log(wage) ~ educ", data=wage1)
results = reg.fit()

print(results.summary())