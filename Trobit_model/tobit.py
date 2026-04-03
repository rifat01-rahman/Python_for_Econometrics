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