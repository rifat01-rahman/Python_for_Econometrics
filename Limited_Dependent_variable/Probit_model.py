import wooldridge as woo
import statsmodels.formula.api as smf
import pandas as pd

# Load data
mroz = woo.dataWoo('mroz')

# -------------------------
# Probit Model
# -------------------------
reg_probit = smf.probit(
    formula='inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)

results_probit = reg_probit.fit(disp=0)

# -------------------------
# Summary
# -------------------------
print("Probit Results:\n", results_probit.summary(), "\n")

# Log-likelihood
print("Log-likelihood:", round(results_probit.llf, 4))

# McFadden pseudo R²
print("Pseudo R²:", round(results_probit.prsquared, 4), "\n")

# -------------------------
# Marginal Effects
# -------------------------
mfx = results_probit.get_margeff(at='mean')  # at mean values

print("Marginal Effects (at means):\n", mfx.summary())

# Optional table
mfx_table = pd.DataFrame({
    'dy/dx': mfx.margeff,
    'se': mfx.margeff_se,
    'z': mfx.tvalues,
    'pval': mfx.pvalues
})

print("\nMarginal Effects Table:\n", mfx_table)