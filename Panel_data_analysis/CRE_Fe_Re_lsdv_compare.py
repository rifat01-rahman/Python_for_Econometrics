import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import linearmodels as plm

# load data
wagepan = woo.dataWoo('wagepan').copy()

# create helper variables
wagepan['t'] = wagepan['year']
wagepan['entity'] = wagepan['nr']

# include group-specific means for time-varying regressors
wagepan['married_b'] = wagepan.groupby('nr')['married'].transform('mean')
wagepan['union_b'] = wagepan.groupby('nr')['union'].transform('mean')

# set panel index
wagepan = wagepan.set_index(['nr', 'year'])

# FE via within estimator
reg_we = plm.PanelOLS.from_formula(
    formula='lwage ~ married + union + C(t)*educ + educ + EntityEffects',
    drop_absorbed=True,
    data=wagepan
)
results_we = reg_we.fit()

# FE via dummy variables
reg_dum = smf.ols(
    formula='lwage ~ married + union + C(t)*educ + educ + C(entity)',
    data=wagepan.reset_index()
)
results_dum = reg_dum.fit()

# CRE / Mundlak
reg_cre = plm.RandomEffects.from_formula(
    formula='lwage ~ married + union + C(t)*educ + educ + married_b + union_b',
    data=wagepan
)
results_cre = reg_cre.fit()

# Naive RE
reg_re = plm.RandomEffects.from_formula(
    formula='lwage ~ married + educ + union + C(t)*educ',
    data=wagepan
)
results_re = reg_re.fit()

# choose coefficients to compare
var_selection = ['married', 'union', 'C(t)[T.1982]:educ']

# comparison table
table = pd.DataFrame({
    'b_we': results_we.params[var_selection].round(4),
    'b_dum': results_dum.params[var_selection].round(4),
    'b_cre': results_cre.params[var_selection].round(4),
    'b_re': results_re.params[var_selection].round(4),
})

print(f'table:\n{table}\n')