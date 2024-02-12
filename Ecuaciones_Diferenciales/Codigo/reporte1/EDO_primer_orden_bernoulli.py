import numpy as np
import matplotlib.pyplot as plt
from sympy import *


x = symbols('x')
y = symbols('y', cls=Function)

eq = Eq(y(x).diff(x), (1/x) * y(x) + y(x)**2)

an_sol = dsolve(eq, ics={y(1): 1})

print('ODE class: ', classify_ode(eq)[0])

pprint(an_sol)

x_begin=1.

x_end=10.

x_nsamples=101

x_space = np.linspace(x_begin, x_end, x_nsamples)

lmbd_sol = lambdify(x, an_sol.rhs)

x_an_sol = lmbd_sol(x_space)

plt.figure()
plt.plot(x_space, x_an_sol, linewidth=1, label='solución analítica')
plt.title('Ecuación diferencial de Bernoulli de primer orden')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
