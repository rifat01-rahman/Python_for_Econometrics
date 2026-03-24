import wooldridge as woo
import pandas as pd
import linearmodels.panel as plm

# load data
wagepan = woo.dataWoo('wagepan')

# create explicit panel identifiers
wagepan['entity'] = wagepan['nr']
wagepan['t'] = wagepan['year']

# include group-specific means (between means) for CRE
wagepan['married_b'] = wagepan.groupby('entity')['married'].transform('mean')
wagepan['union_b'] = wagepan.groupby('entity')['union'].transform('mean')

# set panel index: entity, time
wagepan = wagepan.set_index(['entity', 't'])

# estimate Correlated Random Effects (CRE) model
reg = plm.RandomEffects.from_formula(
    formula='lwage ~ married + union + educ + black + hisp + married_b + union_b',
    data=wagepan
)

results = reg.fit()

# print regression table
table = pd.DataFrame({
    'b': round(results.params, 4),
    'se': round(results.std_errors, 4),
    't': round(results.tstats, 4),
    'pval': round(results.pvalues, 4)
})

print(f"table:\n{table}\n")