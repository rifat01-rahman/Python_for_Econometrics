import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import linearmodels.panel as plm

# load dataset
crime2 = woo.dataWoo('crime2')

# create time dummy
crime2['t'] = (crime2['year'] == 87).astype(int)

# create entity id
id_tmp = np.linspace(1, 46, num=46)
crime2['id'] = np.sort(np.concatenate([id_tmp, id_tmp]))

# compute first differences
crime2 = crime2.sort_values(['id', 'year'])

crime2['crmrte_diff1'] = crime2.groupby('id')['crmrte'].diff()
crime2['unem_diff1'] = crime2.groupby('id')['unem'].diff()

# preview variables
var_selection = ['id', 't', 'crimes', 'unem', 'crmrte_diff1', 'unem_diff1']
print(f"crime2[var_selection].head():\n{crime2[var_selection].head()}\n")

# -------------------------
# First Difference using statsmodels
# -------------------------

reg_sm = smf.ols(
    formula='crmrte_diff1 ~ unem_diff1',
    data=crime2
)

results_sm = reg_sm.fit()

table_sm = pd.DataFrame({
    'b': round(results_sm.params, 4),
    'se': round(results_sm.bse, 4),
    't': round(results_sm.tvalues, 4),
    'pval': round(results_sm.pvalues, 4)
})

print(f"table_sm:\n{table_sm}\n")


# -------------------------
# First Difference using linearmodels
# -------------------------

crime2 = crime2.set_index(['id', 'year'])

reg_plm = plm.FirstDifferenceOLS.from_formula(
    formula='crmrte ~ t + unem',
    data=crime2
)

results_plm = reg_plm.fit()

table_plm = pd.DataFrame({
    'b': round(results_plm.params, 4),
    'se': round(results_plm.std_errors, 4),
    't': round(results_plm.tstats, 4),
    'pval': round(results_plm.pvalues, 4)
})

print(f"table_plm:\n{table_plm}\n")