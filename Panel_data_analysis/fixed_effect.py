import wooldridge as woo
import pandas as pd
import linearmodels.panel as plm

# load data
wagepan = woo.dataWoo('wagepan')

# set panel index (entity = nr, time = year)
wagepan = wagepan.set_index(['nr', 'year'], drop=False)

# Fixed Effects (FE) model
reg = plm.PanelOLS.from_formula(
    formula='lwage ~ married + union + C(year)*educ + EntityEffects',
    data=wagepan,
    drop_absorbed=True
)

results = reg.fit()

# regression table
table = pd.DataFrame({
    'b': round(results.params, 4),
    'se': round(results.std_errors, 4),
    't': round(results.tstats, 4),
    'pval': round(results.pvalues, 4)
})

print(f"table:\n{table}\n")