import imp
import casadi
import numpy as np
from track import SymbolicTrack
from poly_track import PolynomialTrack
import matplotlib.pyplot as plt
from scipy import interpolate
from poly_track import PolyPath
from simulation_twowheeldrivebycicle_nwh_no_long_update_v6 import convertXYtoSN,convertSNtoXY

#check_pos = [-282,41]
check_pos =[-286.3,24.7]
track = SymbolicTrack('tracks/temp_nwh.csv',5)
tau0,n0 = track.convertXYtoTN(check_pos)
s0 = float(track.getSFromT(tau0))

ds = 100
s0_array = np.arange(s0-2,s0 + ds,min(ds/200,0.2))

tau_array = casadi.reshape(track.getTFromS(s0_array),1,len(s0_array))   
pos = track.pt_t(tau_array)    
pos = np.array(pos).reshape(len(s0_array),2)

#polynomial fitting    
x_axis = s0_array - s0
sum_array =[]
for order in range(3,20):
    coeff = np.polyfit(x_axis,pos,order)    
    x_poly = np.polyval(coeff[:,0],x_axis)
    y_poly = np.polyval(coeff[:,1],x_axis)
    poly_val = np.vstack([x_poly,y_poly]).T
    norm_array = np.linalg.norm(poly_val-pos,axis=-1)
    sum_norm = np.sum(norm_array)
    sum_array.append(sum_norm)
    if sum_norm/200<0.1:
        break

ref_x = np.polyval(coeff[:,0],x_axis)
ref_y = np.polyval(coeff[:,1],x_axis)

best_order = np.argmin(sum_array)+3    
print(f"ds: {ds}, order: {best_order}, sum norm {sum_array[best_order-3]}")   
coeff = np.polyfit(x_axis,pos,best_order)


s_new,n_new = convertXYtoSN(track,coeff,check_pos,s0)
print([s_new,n_new])

x_new,y_new = convertSNtoXY(coeff,[s_new,n_new])
print([x_new,y_new])

fig,ax = plt.subplots()
track.plot(ax)
plt.plot(ref_x,ref_y)
plt.plot(check_pos[0],check_pos[1],'*g')
plt.plot(x_new,y_new,'*r')
plt.show()