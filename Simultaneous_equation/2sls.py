import wooldridge as woo
import numpy as np
import pandas as pd
import linearmodels.iv as iv

# Load data
mroz = woo.dataWoo('mroz')

# Restrict to non-missing wage observations
mroz = mroz.dropna(subset=['lwage'])

# ---------------------------
# Equation 1 (hours equation)
# ---------------------------
reg_iv1 = iv.IV2SLS.from_formula(
    'hours ~ 1 + educ + age + kidslt6 + nwifeinc + [np.log(wage) ~ exper + I(exper**2)]',
    data=mroz
)

results_iv1 = reg_iv1.fit(cov_type='robust')

# ---------------------------
# Equation 2 (wage equation)
# ---------------------------
reg_iv2 = iv.IV2SLS.from_formula(
    'np.log(wage) ~ 1 + educ + exper + I(exper**2) + [hours ~ age + kidslt6 + nwifeinc]',
    data=mroz
)

results_iv2 = reg_iv2.fit(cov_type='robust')

# ---------------------------
# Print results
# ---------------------------
table_iv1 = pd.DataFrame({
    'b': round(results_iv1.params, 4),
    'se': round(results_iv1.std_errors, 4),
    't': round(results_iv1.tstats, 4),
    'pval': round(results_iv1.pvalues, 4)
})
print("IV Equation 1 (Hours):\n", table_iv1, "\n")

table_iv2 = pd.DataFrame({
    'b': round(results_iv2.params, 4),
    'se': round(results_iv2.std_errors, 4),
    't': round(results_iv2.tstats, 4),
    'pval': round(results_iv2.pvalues, 4)
})
print("IV Equation 2 (Wage):\n", table_iv2, "\n")

# ---------------------------
# Residual Correlation
# ---------------------------
cor_u1u2 = np.corrcoef(results_iv1.resids, results_iv2.resids)[0, 1]
print("Correlation between residuals:", round(cor_u1u2, 4))

# So, Correlation of residulas tells us that is there 2SLS is sufficient for the analysis or We should go for 3SLS techniques. When the value is close to Zero (0), then we can say 2SLS is ok, otherwise go for 3SLS.