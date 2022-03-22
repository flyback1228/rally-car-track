import casadi
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('output/speed.csv', delimiter=',')
v = data[0,1:]
a = (v[1:]-v[0:-1])/0.2

opti = casadi.Opti()
k = opti.variable(2)
estimate_a = opti.variable(len(a))

estimate_a = k[0] - k[1]*v[0:-1]*v[0:-1]

opti.minimize(casadi.norm_2(estimate_a-a))

opti.solver("ipopt",{},{}) # set numerical backend
sol = opti.solve()   # actual solve

print(sol.value(k))

t = np.linspace(0,0.2*len(a),len(a),endpoint=False)
plt.figure()
plt.plot(t,a)
plt.plot(t,sol.value(estimate_a))
plt.show()

