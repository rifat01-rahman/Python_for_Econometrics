import wooldridge as woo

import pandas as pd 

import numpy as np 

import statsmodels.api as sm

import statsmodels.formula.api as smf
 
# load datset 

barium = woo.dataWoo ('barium')

# Number of observations

T = len (barium)

# Create time index, where time starts at 1978 

barium.index = pd.date_range (start= "1978-02", periods = T, freq = "ME")

# Estimate the model with OLS 

model = smf.ols (formula= 'np.log (chnimp) ~ np.log (chempi) + np.log (gas) + np.log (rtwex) + '
                 'befile6 + affile6 + afdec6', 
                 data = barium)

results = model.fit ()

# AUTOMATIC Breusch–Godfrey test (up to 3 lags)

bg_test = sm. stats.diagnostic.acorr_breusch_godfrey (results, nlags = 3 )

fstat = bg_test [2]

fpvalue = bg_test [3] 

print (f"Automatic Breusch_godfrey test: F-statistics = {fstat:.4f}, p-value = {fpvalue:.4f}")

# Manula process for Serial Correlation 

barium ['resid']= results.resid

# Create lagged residuals up to 3 lags 

barium ['resid_lag1'] = barium ['resid'].shift (1)
barium ['resid_lag2'] = barium ['resid'].shift (2)
barium ['resid_lag3'] = barium ['resid'].shift (3)

# run ols regression 

bg_manual = smf.ols (formula = 'resid ~ resid_lag1 + resid_lag2 + resid_lag3 + '
                     'np.log (chempi) + np.log (gas) + np.log (rtwex) + ' \
                     'befile6 + affile6 + afdec6',
             data= barium
)


results_manual = bg_manual.fit () 

# test joint significance of lagged residuals 

hypothesis = 'resid_lag1 = resid_lag2 = resid_lag3 = 0'

f_test_manual = results_manual.f_test (hypothesis)

fstat_manual = float (f_test_manual. statistic)

fpvalue_manual = float (f_test_manual. pvalue)

print (f"Manual Breusch-Godfrey test : F-statitics = {fstat_manual :.4f}, p-value = {fpvalue_manual:.4f}")
