from scipy.stats import rice
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from cycler import cycler

matplotlib.rc('text', usetex=True)
fig, ax = plt.subplots(1, 1)
b = 10
mean, var, skew, kurt = rice.stats(b, moments='mvsk')
ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
                  cycler('lw', [1, 2, 3, 4]))

for i in np.arange(0, 3, 0.75):
    x = np.linspace(rice.ppf(0.01, i),
              rice.ppf(0.99, i), 100)
    ax.plot(x, rice.pdf(x, i), lw=2, alpha=0.6, label=r"$\nu =$ "+str(i))

plt.legend()
plt.xlabel(r"$\nu$")
plt.ylabel(r"$p(x | \nu)$")

plt.show()
