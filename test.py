import casadi
import numpy as np
from track import SymbolicTrack
from poly_track import PolynomialTrack
import matplotlib.pyplot as plt
from scipy import interpolate
from poly_track import PolyPath

waypoints = np.genfromtxt('tracks/temp_nwh.csv', delimiter=',')
n = len(waypoints)

waypoints = np.vstack([waypoints,waypoints[0,:]])

t = np.arange(0, n+1)
resolution = 100
print(t.shape)
print(waypoints.shape)
circle = interpolate.interp1d(t,waypoints,kind='cubic',axis=0,fill_value = 'extrapolate')
ts = np.linspace(0, n+1, (n+1)*resolution,endpoint=False)
print(ts)
center_line = circle(ts)
d = center_line[1:,:] - center_line[0:-1,:]
ds = np.linalg.norm(d,ord=2,axis=1)        
ds = np.insert(ds,0,0.0)
s = np.cumsum(ds)

my_track = SymbolicTrack('tracks/temp_nwh.csv',5)

t = np.linspace(2,6,100).reshape(1,100)
#pos = my_track.convertParameterToPos(t,np.zeros(100),500);
pos = my_track.pt_t(t)
pos = np.array(pos).reshape(100,2)
print(pos.shape)

p = np.polyfit(t.reshape(100,),pos,5)
print(p)



polyval_x = np.polyval(p[:,0],t.reshape(100,))
polyval_y = np.polyval(p[:,1],t.reshape(100,))

plt.plot(polyval_x,polyval_y,'-*r')

polytrack = PolynomialTrack('tracks/temp_nwh.csv',5)
#plt.plot(my_track.center_line[:,0],my_track.center_line[:,1])
plt.plot(polytrack.center_line[:,0],polytrack.center_line[:,1])
plt.show()