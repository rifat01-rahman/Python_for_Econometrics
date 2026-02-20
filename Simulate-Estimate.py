import numpy as np
import scipy.stats as stats

# set the random seed:
np.random.seed(123456)

# set sample size:
n = 100

# draw a sample given the population parameters:
sample1 = stats.norm.rvs(10, 2, size=n)

# estimate the population mean with the sample average:
estimate1 = np.mean(sample1)
print(f'estimate1: {estimate1}\n')

# draw a different sample and estimate again:
sample2 = stats.norm.rvs(10, 2, size=n)
estimate2 = np.mean(sample2)
print(f'estimate2: {estimate2}\n')

# draw a third sample and estimate again:
sample3 = stats.norm.rvs(10, 2, size=n)
estimate3 = np.mean(sample3)
print(f'estimate3: {estimate3}\n')

# Here, We have used the techniques of Monte Carlo Simulation to draw three different samples. Where we have already decided that the whole population is normally distributed, and when we randomly selsct the sample after that calculate the mean. It shows that mean are approximalely equal. It's all a proof of Central limit theorem. Estimate1 = 9.57360, Estimate2 = 10.245, Estimate3 = 9.960