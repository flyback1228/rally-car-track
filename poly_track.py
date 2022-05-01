import numpy as np
import casadi
import matplotlib.pyplot as plt
from scipy import interpolate

class PolynomialTrack:
    def __init__(self,filename,track_width) -> None:
        resolution = 100
        self.waypoints = np.genfromtxt(filename, delimiter=',')
        circle_pts = np.vstack([self.waypoints,self.waypoints[0,:]])
        n = len(self.waypoints)
        t = np.arange(0, n+1)
        
        self.circle = interpolate.interp1d(t,circle_pts,kind='cubic',axis = 0, fill_value='extrapolate')
        ts = np.linspace(0, n+1, (n+1)*resolution,endpoint=False)
        self.center_line = self.circle(ts)
        d = self.center_line[1:,:] - self.center_line[0:-1,:]
        ds = np.linalg.norm(d,ord=2,axis=1)        
        ds = np.insert(ds,0,0.0)
        s = np.cumsum(ds)
        pass
    
class PolyPath:
    def __init__(self,orders,coeff) -> None:
        assert(len(coeff)==orders+1)
        s = casadi.MX.sym('s')
        self.pos = coeff[0,:]*casadi.power(s,orders)
        for i in range(1,orders):
            self.pos += coeff[i,:] * casadi.power(s,orders-i)
        self.pos += coeff[-1,:]
        
        jac = casadi.jacobian(self.pos,s)
        hes = casadi.jacobian(jac,s)
        kappa = (jac[0]*hes[1]-jac[1]*hes[0])/casadi.power(casadi.norm_2(jac),3)
        phi = casadi.arctan2(jac[1],jac[0])
        self.kappa = casadi.Function('kappa',[s],[kappa])
        self.phi = casadi.Function('phi',[s],[phi])
        
        pass
    
    