from cmath import pi
import casadi
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('output/state.csv', delimiter=',')
real_phi = casadi.DM(data[1:-1,3])
ref_phi = casadi.DM(data[2:,3])
diff = ref_phi-real_phi

v0 = casadi.DM(data[1:-1,4])
dt = casadi.DM(data[2:,0])
#estimate = data[2:,4:]
steer = 2.0/180*casadi.pi



opti = casadi.Opti()
l = opti.variable()
#estimate_phi = opti.variable(len(real_phi))
 
#estimate_phi = real_phi + v0/l * casadi.tan(steer)*dt
#estimate = k[0] - k[1]*v[0:-1]*v[0:-1]

opti.minimize(casadi.norm_2(v0*l * casadi.tan(steer)*dt-diff))

opti.solver("ipopt",{},{}) # set numerical backend
sol = opti.solve()   # actual solve

print(1.0/float(sol.value(l)))

#t = np.linspace(0,0.2*len(a),len(a),endpoint=False)
#plt.figure()
#plt.plot(t,a)
#plt.plot(t,sol.value(estimate_a))
#plt.show()

 