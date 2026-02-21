import wooldridge as woo
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# load dataset
ceosal1 = woo.data("ceosal1")

# OLS regression
reg = smf.ols(formula="salary ~ roe", data=ceosal1)
results = reg.fit()

# scatterplot and fitted values
plt.plot(ceosal1["roe"], ceosal1["salary"],
         color="grey", marker="o", linestyle="")

plt.plot(ceosal1["roe"], results.fittedvalues,
         color="black", linestyle="-")

plt.ylabel("salary")
plt.xlabel("roe")

plt.savefig("Example-2-3-3.pdf")
plt.show()