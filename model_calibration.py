from cProfile import label
import casadi
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

def standard_car_calib():
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

def nwh_car_calib(table_name,start_row,end_row):
    
    con = sqlite3.connect('output/sql_data.db')
    cur = con.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    raw_data = cur.fetchall() 
    data = np.array(raw_data[start_row:end_row])
    t = data[:,0]
    x=data[:,4]
    y=data[:,5]
    psi=data[:,6]
    v=data[:,7]
    steer=data[:,10]
    throttle = data[:,11]
    brake = data[:,12]
    a = (v[1:]-v[:-1])/(t[1:]-t[:-1])*1000
    throttle = throttle[:-1]
    brake = brake[:-1]
    
    print(np.dot(np.sign(a),np.sign(throttle)))
    print(np.dot(np.sign(a[1:]),np.sign(throttle[:-1])))
    
    
    opti = casadi.Opti()
    k1 = opti.variable()
    k2 = opti.variable()
    k3 = opti.variable()
    opti.minimize(casadi.norm_2(k1*throttle[:-1] - k2*brake[:-1] -k3*v[1:-1] -a[1:]))

    opti.solver("ipopt",{},{}) # set numerical backend
    sol = opti.solve()   # actual solve
    print(sol.value(k1))
    print(sol.value(k2))
    print(sol.value(k3))
    plt.plot(a[1:],label='a')
    plt.plot(float(sol.value(k1))*throttle[:-1] - float(sol.value(k2))*brake[:-1] -float(sol.value(k3))*v[1:-1],label='simulation')
    plt.legend()
    plt.show()

    
if __name__=='__main__':
    nwh_car_calib('_04_15_2022_17_55_58',12,150)
    

 