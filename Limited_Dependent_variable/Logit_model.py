import wooldridge as woo
import statsmodels.formula.api as smf
import pandas as pd

# Load data
mroz = woo.dataWoo('mroz')

# Logit model
reg_logit = smf.logit(
    formula='inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)

results_logit = reg_logit.fit(disp=0)

# -------------------------
# Summary
# -------------------------
print("Logit Results:\n", results_logit.summary(), "\n")

# Log-likelihood
print("Log-likelihood:", round(results_logit.llf, 4))

# McFadden pseudo R²
print("Pseudo R²:", round(results_logit.prsquared, 4), "\n")

# -------------------------
# Marginal Effects
# -------------------------
mfx = results_logit.get_margeff(at='mean')  # evaluate at mean values

print("Marginal Effects (at means):\n", mfx.summary())

# Optional: convert to table
mfx_table = pd.DataFrame({
    'dy/dx': mfx.margeff,
    'se': mfx.margeff_se,
    'z': mfx.tvalues,
    'pval': mfx.pvalues
})

print("\nMarginal Effects Table:\n", mfx_table)