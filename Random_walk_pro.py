import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# set the random seed
np.random.seed(1234567)

# initialize x-axis
x_range = np.linspace(0, 50, num=51)

# set axis limits
plt.ylim([-18, 18])
plt.xlim([0, 50])

# loop over 30 simulations
for r in range(0, 30):

    # i.i.d. standard normal shocks
    e = stats.norm.rvs(0, 1, size=51)

    # set first shock to zero (y0 = 0)
    e[0] = 0

    # random walk: cumulative sum
    y = np.cumsum(e)

    # plot the random walk
    plt.plot(x_range, y, color='lightgrey', linestyle='-')

# add horizontal reference line
plt.axhline(linewidth=2, linestyle='--', color='black')

# labels
plt.ylabel('y')
plt.xlabel('time')

# save figure
plt.savefig('Simulate-RandomWalk.pdf')

# show plot
plt.show()