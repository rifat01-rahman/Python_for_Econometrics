import statsmodels.api as sm
import patsy as pt
import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

# load data
barium = woo.dataWoo('barium')

# OLS regression
model = smf.ols(
    formula='np.log(chnimp) ~ np.log(chempi) + np.log(gas) + np.log(rtwex) + befile6 + affile6 + afdec6',
    data=barium
)

results = model.fit()


# create matrices
y, X = pt.dmatrices(
    'np.log(chnimp) ~ np.log(chempi) + np.log(gas) + np.log(rtwex) + befile6 + affile6 + afdec6',
    data=barium,
    return_type='dataframe'
)

# Breusch-Pagan test
bp_test = sm.stats.diagnostic.het_breuschpagan(results.resid, X)

bp_stat = bp_test[0]
bp_pval = bp_test[1]

print("BP statistic:", bp_stat)
print("BP p-value:", bp_pval)