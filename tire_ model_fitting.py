from plistlib import FMT_XML
import casadi
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(-1,1,100)
lamb = np.linspace(-1,1,100)




fx_unity = 2.5*lamb
index = np.where(lamb<-0.4)
fx_unity[index] = -1.25*lamb[index]-1.5

index = np.where(lamb>0.4)
fx_unity[index] = -1.25*lamb[index]+1.5

index = np.where(lamb<-0.8)
fx_unity[index] = -0.5

index = np.where(lamb>0.8)
fx_unity[index] = 0.5

fy_unity = 5*alpha
index = np.where(alpha<-0.2)
fy_unity[index] = -0.25/0.3*alpha[index]-7.0/6.0

index = np.where(alpha>0.2)
fy_unity[index] = -0.25/0.3*alpha[index]+7.0/6.0

index = np.where(alpha<-0.5)
fy_unity[index] = -0.75

index = np.where(alpha>0.5)
fy_unity[index] = 0.75


v_long = []
v_lat=[]

for i in range(1,1000):
    B = i/100.0
    for j in range(1,1000):
        C = j/100.0
        a = np.sin(C*np.arctan(B*lamb))-fx_unity
        b = np.sin(C*np.arctan(B*alpha))-fy_unity
        v_long.append([i,j,np.dot(a,a)])
        v_lat.append([i,j,np.dot(b,b)])

value_long = np.array(v_long)
value_lat = np.array(v_lat)
#print(value.shape)

index_long =np.argmin(value_long[:,2])
print("long:")
print(value_long[index_long])
B_long = value_long[index_long,0]/100.0
C_long = value_long[index_long,1]/100.0

index_lat =np.argmin(value_lat[:,2])
print("lat:")
print(value_lat[index_lat])
B_lat = value_lat[index_lat,0]/100.0
C_lat = value_lat[index_lat,1]/100.0

fig = plt.figure()
plt.plot(lamb,fx_unity)
plt.plot(lamb,np.sin(C_long*np.arctan(B_long*lamb)))

plt.plot(lamb,fy_unity)
plt.plot(lamb,np.sin(C_lat*np.arctan(B_lat*alpha)))

plt.show()

#long:
#[111.         340.           1.60423715]
#lat:
#[397.         184.           0.77698341]

"""
fig = plt.figure()
plt.plot(lamb,sol_fx)
plt.plot(lamb,fx_unity)

fy_unity = 5*alpha
index = np.where(alpha<-0.2)
fy_unity[index] = -0.25/0.3*alpha[index]-7.0/6.0

index = np.where(alpha>0.2)
fy_unity[index] = -0.25/0.3*alpha[index]+7.0/6.0

index = np.where(alpha<-0.5)
fy_unity[index] = -0.75

index = np.where(alpha>0.5)
fy_unity[index] = 0.75

plt.show()
"""