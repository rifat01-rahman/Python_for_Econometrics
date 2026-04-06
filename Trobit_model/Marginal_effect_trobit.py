import wooldridge as woo
import numpy as np
import patsy as pt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.base.model as smclass

# -----------------------------
# Load data
# -----------------------------
mroz = woo.dataWoo('mroz')

# Design matrices
y, X = pt.dmatrices(
    'hours ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz,
    return_type='dataframe'
)

# Convert y to 1D array (important!)
y = np.asarray(y).flatten()

# -----------------------------
# Starting values (OLS)
# -----------------------------
reg_ols = smf.ols(
    formula='hours ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)
results_ols = reg_ols.fit()

sigma_start = np.log(np.mean(results_ols.resid**2))
params_start = np.concatenate((results_ols.params.values, [sigma_start]))

# -----------------------------
# Tobit Model
# -----------------------------
class Tobit(smclass.GenericLikelihoodModel):

    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        p = X.shape[1]

        beta = params[:p]
        sigma = np.exp(params[p])  # ensure sigma > 0

        y_hat = np.dot(X, beta)

        # Split observations
        y_eq = (y == 0)   # censored
        y_g = (y > 0)     # uncensored

        ll = np.zeros(len(y))

        # Censored part
        ll[y_eq] = np.log(stats.norm.cdf(-y_hat[y_eq] / sigma))

        # Uncensored part
        ll[y_g] = (
            np.log(stats.norm.pdf((y[y_g] - y_hat[y_g]) / sigma))
            - np.log(sigma)
        )

        return -ll  # negative log-likelihood


# -----------------------------
# Estimate model
# -----------------------------
reg_tobit = Tobit(endog=y, exog=X)
results_tobit = reg_tobit.fit(start_params=params_start, maxiter=10000, disp=0)

print(results_tobit.summary())

# -----------------------------
# Marginal Effects (Tobit)
# -----------------------------

# Extract parameters
params = results_tobit.params
p = X.shape[1]

beta = params[:p]
sigma = np.exp(params[p])

# Linear index
xb = np.dot(X, beta)
z = xb / sigma

# PDF and CDF
phi = stats.norm.pdf(z)
Phi = stats.norm.cdf(z)

# -----------------------------
# (1) APE: Effect on E(y|x)
# -----------------------------
ME_Ey = beta * np.mean(Phi)

# -----------------------------
# (2) APE: Effect on P(y>0)
# -----------------------------
ME_prob = (beta / sigma) * np.mean(phi)

# -----------------------------
# (3) APE: Effect on E(y | y>0, x)
# -----------------------------
lambda_ = phi / Phi
ME_cond = beta * np.mean(1 - lambda_ * (z + lambda_))

# -----------------------------
# Put into table
# -----------------------------
import pandas as pd

var_names = results_ols.model.exog_names

table_me = pd.DataFrame({
    'Variable': var_names,
    'ME_E[y|x]': np.round(ME_Ey, 4),
    'ME_P(y>0)': np.round(ME_prob, 4),
    'ME_E[y|y>0]': np.round(ME_cond, 4)
})

print("\nMarginal Effects (APE):\n")
print(table_me)