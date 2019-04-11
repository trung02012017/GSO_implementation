import sympy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

x = [0, 1, 2]
# y = [sympy.Float('1e-20'), sympy.Float('1e-100'), sympy.Float('1e-700')]
y = [sympy.Float('1.23e+100'), sympy.Float('1.23e-200'), sympy.Float('1.23e-700')]

def log_formatter(x):
    return "$10^{{{:d}}}$".format(int(x))

formatter = FuncFormatter(log_formatter)
fig, ax = plt.subplots()

y2 = list(map(lambda x:sympy.log(x, 10), y))
print(y2)
ax.plot(x, y2)
ax.yaxis.set_major_formatter(formatter)
ax.grid();
plt.show()