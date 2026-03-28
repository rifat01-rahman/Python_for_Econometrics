import wooldridge as woo
import pandas as pd
import linearmodels.iv as iv

# Load data
jtrain = woo.dataWoo('jtrain')

# Drop missing
jtrain = jtrain.dropna(subset=['lscrap'])

# Select variables
jtrain = jtrain[['lscrap', 'hrsemp', 'grant', 'year', 'fcode']]

# Keep only 1987 and 1988
jtrain_87_88 = jtrain.loc[
    (jtrain['year'] == 1987) | (jtrain['year'] == 1988),
    :
]

# Set panel index
jtrain_87_88 = jtrain_87_88.set_index(['fcode', 'year'])

# Sort properly (IMPORTANT)
jtrain_87_88 = jtrain_87_88.sort_index()

# First differences
jtrain_87_88['lscrap_diff1'] = jtrain_87_88.groupby(level=0)['lscrap'].diff()
jtrain_87_88['hrsemp_diff1'] = jtrain_87_88.groupby(level=0)['hrsemp'].diff()
jtrain_87_88['grant_diff1'] = jtrain_87_88.groupby(level=0)['grant'].diff()

# Drop NA after differencing
jtrain_87_88 = jtrain_87_88.dropna()

# -------------------------
# IV Regression (FD-IV)
# -------------------------
reg_iv = iv.IV2SLS.from_formula(
    'lscrap_diff1 ~ 1 + [hrsemp_diff1 ~ grant_diff1]',
    data=jtrain_87_88
)

results_iv = reg_iv.fit(cov_type='robust')  # better than unadjusted

# Output table
table_iv = pd.DataFrame({
    'b': round(results_iv.params, 4),
    'se': round(results_iv.std_errors, 4),
    't': round(results_iv.tstats, 4),
    'pval': round(results_iv.pvalues, 4)
})

print("FD-IV Results:\n", table_iv)