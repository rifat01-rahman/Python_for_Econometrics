import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
barium = woo.dataWoo('barium')

# Create time index
T = len(barium)
barium.index = pd.date_range(start='1978-02', periods=T, freq='ME')

# Extract month
barium['month'] = barium.index.month

# Regression
reg = smf.ols(
    formula='np.log(chnimp) ~ np.log(chempi) + np.log(gas) + np.log(rtwex) + \
             befile6 + affile6 + afdec6 + C(month)',
    data=barium
)

results = reg.fit()
print(results.summary())