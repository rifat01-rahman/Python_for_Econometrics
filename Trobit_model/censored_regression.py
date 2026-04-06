import wooldridge as woo
import numpy as np
import patsy as pt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.base.model as smclass

# -----------------------------
# Load data
# -----------------------------
recid = woo.dataWoo('recid')

# Censoring indicator (True = censored)
censored = recid['cens'] != 0

# Design matrices
y, X = pt.dmatrices(
    'ldurat ~ workprg + priors + tserved + felon + alcohol + drugs + black + married + educ + age',
    data=recid,
    return_type='dataframe'
)

# Convert y to 1D
y = np.asarray(y).flatten()

# -----------------------------
# Starting values (OLS)
# -----------------------------
reg_ols = smf.ols(
    formula='ldurat ~ workprg + priors + tserved + felon + alcohol + drugs + black + married + educ + age',
    data=recid
)
results_ols = reg_ols.fit()

sigma_start = np.log(np.mean(results_ols.resid**2))
params_start = np.concatenate((results_ols.params.values, [sigma_start]))

# -----------------------------
# Censored Regression Model
# -----------------------------
class CensReg(smclass.GenericLikelihoodModel):

    def __init__(self, endog, exog, cens):
        self.cens = cens
        super().__init__(endog, exog, missing='none')

    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        cens = self.cens

        p = X.shape[1]
        beta = params[:p]
        sigma = np.exp(params[p])

        y_hat = np.dot(X, beta)

        ll = np.zeros(len(y))

        # -----------------------------
        # Uncensored observations
        # -----------------------------
        ll[~cens] = (
            np.log(stats.norm.pdf((y[~cens] - y_hat[~cens]) / sigma))
            - np.log(sigma)
        )

        # -----------------------------
        # Right-censored observations
        # -----------------------------
        ll[cens] = np.log(
            1 - stats.norm.cdf((y[cens] - y_hat[cens]) / sigma)
        )

        return -ll


# -----------------------------
# Estimation
# -----------------------------
reg_censReg = CensReg(endog=y, exog=X, cens=censored)

results_censReg = reg_censReg.fit(
    start_params=params_start,
    maxiter=10000,
    method='BFGS',
    disp=0
)

print(results_censReg.summary())