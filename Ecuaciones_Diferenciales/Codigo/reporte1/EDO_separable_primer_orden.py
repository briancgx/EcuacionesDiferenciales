import numpy as np
import matplotlib.pyplot as plt
from sympy import *

""" 
define la variable independiente 
"""
x = symbols('x')
""" 
define la variable dependiente 
"""
y = symbols('y', cls=Function)
""" 
define la ecuación diferencial 
"""
eq = Eq(y(x).diff(x), -2 * x * y(x))
""" 
determina la solución analítica
de la ecuación diferencial
"""
an_sol = dsolve(eq, ics={y(0): 1})
""" 
imprime qué clase de ecuación diferencial es 
"""
print('ODE class: ', classify_ode(eq)[0])
""" 
escribe la solución analítica 
"""
pprint(an_sol)
""" 
define el punto inicial 
"""
x_begin=0.
""" 
define el punto final para resolver numéricamente 
"""
x_end=10.
""" 
define el número de puntos en la solución 
"""
x_nsamples=101
""" 
haz un arregle de puntos donde evaluar 
la solución de la ecuación diferencial 
"""
x_space = np.linspace(x_begin, x_end, x_nsamples)
"""
crea un objeto que evalúa la solución en los puntos de integración
"""
lmbd_sol = lambdify(x, an_sol.rhs)
"""
evalua la solución analítica en los puntos de integración
"""
x_an_sol = lmbd_sol(x_space)
"""
haz una gráfica de la solución
"""
plt.figure()
plt.plot(x_space, x_an_sol, linewidth=1, label='solución analítica')
plt.title('Ecuación diferencial separable de primer orden')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
