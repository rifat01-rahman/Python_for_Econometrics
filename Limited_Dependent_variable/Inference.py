import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

# -----------------------------
# STEP 1: Load Data
# -----------------------------
mroz = woo.dataWoo('mroz')

# -----------------------------
# STEP 2: Estimate FULL Probit Model
# -----------------------------
reg_probit = smf.probit(
    formula='inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6',
    data=mroz
)

results_probit = reg_probit.fit(disp=0)

print("FULL PROBIT MODEL:\n")
print(results_probit.summary())
print("\n")

# -----------------------------
# TEST 1: Overall Significance (LR Test)
# H0: All slope coefficients = 0
# -----------------------------
llr1_manual = 2 * (results_probit.llf - results_probit.llnull)

print("TEST 1: Overall Model Significance")
print("LR Statistic (manual):", round(llr1_manual, 4))
print("LR Statistic (built-in):", round(results_probit.llr, 4))
print("p-value:", round(results_probit.llr_pvalue, 4))
print("\n")

# -----------------------------
# TEST 2: Wald Test
# H0: exper = exper^2 = age = 0
# -----------------------------
hypotheses = ['exper=0', 'I(exper ** 2)=0', 'age=0']

waldstat = results_probit.wald_test(hypotheses)

print("TEST 2: Wald Test (Joint Significance)")
print("Test Statistic:", float(waldstat.statistic))
print("p-value:", float(waldstat.pvalue))
print("\n")

# -----------------------------
# TEST 3: LR Test (Restricted vs Full)
# H0: exper = exper^2 = age = 0
# -----------------------------
reg_probit_restr = smf.probit(
    formula='inlf ~ nwifeinc + educ + kidslt6 + kidsge6',
    data=mroz
)

results_probit_restr = reg_probit_restr.fit(disp=0)

llr2_manual = 2 * (results_probit.llf - results_probit_restr.llf)
pval2_manual = 1 - stats.chi2.cdf(llr2_manual, 3)

print("TEST 3: LR Test (Restricted vs Full)")
print("LR Statistic:", round(llr2_manual, 4))
print("p-value:", round(pval2_manual, 4))