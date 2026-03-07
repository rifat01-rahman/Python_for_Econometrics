# -----------------------------------------------------------
# Import libraries
# -----------------------------------------------------------
import wooldridge as woo
import pandas as pd

# -----------------------------------------------------------
# Load the dataset
# -----------------------------------------------------------
lawsch85 = woo.data('lawsch85')

# -----------------------------------------------------------
# Extract LSAT variable (Law School Admission Test score)
# -----------------------------------------------------------
lsat_pd = lawsch85['LSAT']

# -----------------------------------------------------------
# Create a boolean indicator showing whether LSAT is missing
# True  = missing value
# False = observed value
# -----------------------------------------------------------
missLSAT = lsat_pd.isna()

# -----------------------------------------------------------
# Preview LSAT values and missing indicator for schools 120–129
# -----------------------------------------------------------
preview = pd.DataFrame({
    'lsat_pd': lsat_pd[119:129],
    'missLSAT': missLSAT[119:129]
})

print(f'preview:\n{preview}\n')

# -----------------------------------------------------------
# Frequency table: how many LSAT values are missing
# -----------------------------------------------------------
freq_missLSAT = pd.crosstab(missLSAT, columns='count')

print(f'freq_missLSAT:\n{freq_missLSAT}\n')

# -----------------------------------------------------------
# Detect missing values for ALL variables in the dataset
# -----------------------------------------------------------
miss_all = lawsch85.isna()

# Count number of missing values per variable
colsums = miss_all.sum(axis=0)

print(f'colsums:\n{colsums}\n')

# -----------------------------------------------------------
# Identify rows with COMPLETE data (no missing values)
# -----------------------------------------------------------
complete_cases = (miss_all.sum(axis=1) == 0)

# Frequency table for complete vs incomplete observations
freq_complete_cases = pd.crosstab(complete_cases, columns='count')

print(f'freq_complete_cases:\n{freq_complete_cases}\n')