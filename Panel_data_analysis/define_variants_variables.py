import wooldridge as woo

# load data
wagepan = woo.dataWoo('wagepan')

# -------------------------
# Panel dimensions
# -------------------------
N = wagepan.shape[0]
T = wagepan['year'].nunique()
n = wagepan['nr'].nunique()

print(f"N: {N}\n")
print(f"T: {T}\n")
print(f"n: {n}\n")

# -------------------------
# (I) Check time-invariant variables (within individuals)
# -------------------------
isv_nr = (wagepan.groupby('nr').var(numeric_only=True) == 0)

# variables with zero variance for ALL individuals
noVar_nr = isv_nr.all(axis=0)

print(f"isv_nr.columns[noVar_nr]:\n{isv_nr.columns[noVar_nr]}\n")

# -------------------------
# (II) Check cross-section invariant variables (within time)
# -------------------------
isv_t = (wagepan.groupby('year').var(numeric_only=True) == 0)

# variables with zero variance for ALL years
noVar_t = isv_t.all(axis=0)

print(f"isv_t.columns[noVar_t]:\n{isv_t.columns[noVar_t]}\n")