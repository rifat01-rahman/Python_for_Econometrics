import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

CPS1985 = pd.read_csv(r'C:\Users\Binary Gadget\Downloads\8th Sem\Determinants of Wages Data (CPS 1985).csv')


# rename variable to make outputs more compact
CPS1985['oc'] = CPS1985['occupation']

# frequency tables for categorical variables
freq_gender = pd.crosstab(CPS1985['gender'], columns='count')
print(f'freq_gender:\n{freq_gender}\n')

freq_occupation = pd.crosstab(CPS1985['oc'], columns='count')
print(f'freq_occupation:\n{freq_occupation}\n')

# regression with categorical variables
reg = smf.ols(
    formula='np.log(wage) ~ education + experience + C(gender) + C(oc)',
    data=CPS1985
)
results = reg.fit()

# print regression table
table = pd.DataFrame({
    'b': round(results.params, 4),
    'se': round(results.bse, 4),
    't': round(results.tvalues, 4),
    'pval': round(results.pvalues, 4)
})
print(f'table:\n{table}\n')

# regression with different reference categories
reg_newref = smf.ols(
    formula='np.log(wage) ~ education + experience + '
            'C(gender, Treatment("male")) + '
            'C(oc, Treatment("technical"))',
    data=CPS1985
)
results_newref = reg_newref.fit()

# print results with new reference
table_newref = pd.DataFrame({
    'b': round(results_newref.params, 4),
    'se': round(results_newref.bse, 4),
    't': round(results_newref.tvalues, 4),
    'pval': round(results_newref.pvalues, 4)
})
print(f'table_newref:\n{table_newref}\n')