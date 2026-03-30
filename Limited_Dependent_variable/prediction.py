import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

# Load data
mroz = woo.dataWoo('mroz')

# -----------------------------
# Estimate models
# -----------------------------
reg_lin = smf.ols(
    formula='inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)
results_lin = reg_lin.fit(cov_type='HC3')

reg_logit = smf.logit(
    formula='inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)
results_logit = reg_logit.fit(disp=0)

reg_probit = smf.probit(
    formula='inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)
results_probit = reg_probit.fit(disp=0)

# -----------------------------
# New observations (2 extreme women)
# -----------------------------
X_new = pd.DataFrame({
    'nwifeinc': [100, 0],
    'educ': [5, 17],
    'exper': [0, 30],
    'age': [20, 52],
    'kidslt6': [2, 0],
    'kidsge6': [0, 0]
})

# IMPORTANT: create squared term manually
X_new['exper_sq'] = X_new['exper']**2

# -----------------------------
# Predictions
# -----------------------------
predictions_lin = results_lin.predict(X_new)
predictions_logit = results_logit.predict(X_new)
predictions_probit = results_probit.predict(X_new)

print("LPM Predictions:\n", predictions_lin, "\n")
print("Logit Predictions:\n", predictions_logit, "\n")
print("Probit Predictions:\n", predictions_probit)