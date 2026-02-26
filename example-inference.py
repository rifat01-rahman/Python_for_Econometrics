import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

gpa1 = woo.data("gpa1")

# store and display results:
reg = smf.ols(formula="colGPA ~ hsGPA + ACT + skipped", data=gpa1)
results = reg.fit()

print(f"results.summary():\n{results.summary()}\n")

# manually confirm the formulas, i.e. extract coefficients and SE:
b = results.params
se = results.bse

# reproduce t statistic:
tstat = b / se
print(f"tstat:\n{tstat}\n")

# reproduce p value:
df = int(results.df_resid)
pval = 2 * stats.t.cdf(-abs(tstat), df)

print(f"pval:\n{pval}\n")