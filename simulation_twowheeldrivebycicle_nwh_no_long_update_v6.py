from warnings import catch_warnings
from weakref import ref
import casadi
from numpy import true_divide
import yaml
from track import *
from dynamics_models import BicycleDynamicsModelTwoWheelDriveWithBrakeNWHFxInput
from tire_model import SimplePacTireMode
import sqlite3
from datetime import datetime
import vehicle_animation
from poly_track import PolyPath
import sympy as sym
import math

track_width = 5
track = SymbolicTrack('tracks/temp_nwh.csv',track_width)

#define tire mode

with open('params/nwh_tire.yaml') as file:
    front_tire_params = yaml.safe_load(file)

front_tire_model = SimplePacTireMode(front_tire_params)
print(front_tire_model.getLambdaAtMaxForce())
with open('params/nwh_tire.yaml') as file:
    rear_tire_params = yaml.safe_load(file)
rear_tire_model = SimplePacTireMode(rear_tire_params)
print(rear_tire_model.getLambdaAtMaxForce())
"""
with open('params/simple_tire.yaml') as file:
    front_tire_params = yaml.load(file)
front_tire_model = SimpleTire(front_tire_params)

with open('params/simple_tire.yaml') as file:
    rear_tire_params = yaml.load(file)
rear_tire_model = SimpleTire(rear_tire_params)
"""

#define model
with open('params/racecar_nwh.yaml') as file:
    params = yaml.safe_load(file)
model = BicycleDynamicsModelTwoWheelDriveWithBrakeNWHFxInput(params,track,front_tire_model,rear_tire_model)

print(params)
lf = params['lf']
lr = params['lr']
mass = params['m']
Iz = params['Iz']

#parameters
v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min']  # minimum steering angle [rad]
delta_max = params['delta_max'] 

delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
delta_dot_max = params['delta_dot_max']

vehicle_length = (params['lf']+params['lr'])*1.2
vehicle_width = params['width']


#initial boundary
tau0 = 18
s0 = track.getSFromT(tau0)
init_s0 = float(s0)
v0 = 5
phi0 =track.getPhiFromT(tau0)
#x = [t,n,phi,vx,vy,omega,steer]
pt = track.pt_t(tau0)
#pt = track.convertParameterToPos(tau0,0,1)
y0 = casadi.vertcat(pt[0],pt[1],phi0,v0,0,0,0)

N = 120
nx = 7
nu = 3
T =4.0
dt = T/N
#ds = T*v_max
t = np.linspace(0,1,N+1)


#data save to sql
con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = 'unity'+now.strftime("_%m_%d_%Y_%H_%M_%S")
cur.execute("CREATE TABLE {} (computation_time real,phi real, vx real, vy real, steer real,omega real,front_wheel_alpha real,rear_wheel_alpha real,front_fx real, rear_fx real,front_fy real, rear_fy real,s real,ds real,poly_order real,ptx text,pty text,error text)".format(table_name))

#casadi options
option = {}
option['max_iter']=3000
option['tol'] = 1e-6
option['print_level']=0
#option['linear_solver']='ma27'
#casadi options
except_option = {}
except_option['max_iter']=10000
except_option['tol'] = 1e-5
except_option['print_level']=0
except_option['linear_solver']='ma27'

casadi_option={}
#casadi_option['print_time']=False

first_run = True



Fz = casadi.DM.ones(2)*params['m']/2*9.81

Fx_f_max = front_tire_model.getMaxLongitudinalForce(Fz[0])
Fx_r_max = rear_tire_model.getMaxLongitudinalForce(Fz[1])

Fy_f_max = front_tire_model.getMaxLateralForce(Fz[0])
Fy_r_max = rear_tire_model.getMaxLateralForce(Fz[1])

sol_x = None
sol_u = None

cd = 5000
cm1 =100
cm0 = 0

guess_X = casadi.DM.zeros(nx,N+1)
guess_U = casadi.DM.zeros(nu,N)

def forwardModel(X,U):
    phi = X[2]
    vx = X[3]
    vy = X[4,:]
    omega = X[5]
    delta = X[6]
    
    # control input
    delta_dot = U[0]
    front_throttle = U[1]
    rear_throttle=U[2]

    #kappa_sym_array = f_kappa(s_sym_array[0:-1])
    #dphi_c_sym_array = phi_sym_array[0:-1]  - f_phi(s_sym_array[0:-1])
    
    fx_f = cd*front_throttle - cm0-cm1*vx
    fx_r = cd*rear_throttle - cm0-cm1*vx
    #tangent_vec_sym_array = track.getTangentVec(tau_sym_array[0:-1]) 

    #tangent_vec_norm = (tangent_vec_sym_array[0,:]*tangent_vec_sym_array[0,:]+tangent_vec_sym_array[1,:]*tangent_vec_sym_array[1,:])**0.5

    alpha_f = -casadi.atan2(omega*lf+vy, vx) + delta
    alpha_r = casadi.atan2(omega*lr-vy,vx)

    fy_f = front_tire_model.getLateralForce(alpha_f,Fz[0])
    fy_r = rear_tire_model.getLateralForce(alpha_r,Fz[1])
    
    #front_rate = casadi.fmax((fy_f_sym_array*fy_f_sym_array+fx_f_sym_array*fx_f_sym_array)/(Fy_f_max*Fy_f_max),1)
    #rear_rate = casadi.fmax((fy_r_sym_array*fy_r_sym_array+fx_r_sym_array*fx_r_sym_array)/(Fy_r_max*Fy_r_max),1)

    #fy_f_sym_array=fy_f_sym_array/(front_rate**0.5)
    #fy_r_sym_array=fy_r_sym_array/(rear_rate**0.5)
    #fx_f_sym_array=fx_f_sym_array/(front_rate**0.5)
    #fx_r_sym_array=fx_r_sym_array/(rear_rate**0.5)
    return casadi.veccat(
        vx*casadi.cos(phi)-vy*casadi.sin(phi), #x_dot
        vx*casadi.sin(phi)+vy*casadi.cos(phi),  # n_dot
        omega, #phi_dot
        1/mass * (fy_r + fx_f*casadi.sin(delta) + fy_f*casadi.cos(delta) - mass*vx*omega),  #vydot    
        1/mass * (fx_r + fx_f*casadi.cos(delta) - fy_f*casadi.sin(delta) + mass*vy*omega),  #vxdot    
        1/Iz * (fy_f*lf*casadi.cos(delta) + fx_f*lf*casadi.sin(delta) - fy_r*lr), #omegadot
        delta_dot
    )

def convertXYtoTN(coeff,pos,s0):    
    
    x1 = float(pos[0])
    y1 = float(pos[1])
    ref_t,ref_n = track.convertXYtoTN([x1,y1])
    assert(ref_t!=-1)
    ref_s = float(track.getSFromT(ref_t))-s0
    #print(track.getPhiFromT(ref_t))
    
    s = sym.Symbol('s')
    order = len(coeff)-1
    
    pos_mx = coeff[0,:]*sym.Pow(s,order)
    
    for i in range(1,order):
        pos_mx += coeff[i,:] * sym.Pow(s,order-i)
    pos_mx += coeff[-1,:]
    dx = sym.diff(pos_mx[0])
    dy = sym.diff(pos_mx[1])
    
    f = -dx*(pos_mx[0] - x1) - dy*(pos_mx[1] - y1)
    #print(f)
    new_coeffs = sym.Poly(f).coeffs();
    #print(new_coeffs)
    raw_roots = np.roots(new_coeffs)
    print(raw_roots)
    abs_root = np.abs(raw_roots - float(ref_s))
    idx = np.argmin(abs_root)
    root = raw_roots[idx]
    assert(np.imag(root)==0)
    root =np.real(root)
    
    x_org = pos_mx[0].evalf(subs={s:root})
    y_org = pos_mx[1].evalf(subs={s:root})
    dx_val = dx.evalf(subs={s:root})
    dy_val = dy.evalf(subs={s:root})
     
    new_x = x1-x_org
    new_y = y1-y_org
    
    theta = math.atan2(dy_val,dx_val)
    M = np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])    
    new_pos = np.matmul(M,np.array([new_x,new_y,1]))
    #print(new_pos)
    
    return [root,new_pos[1]]


def optimize(y0):  
    #define ocp 
    global guess_X,guess_U,sol_x,sol_u
    error = ''
    #s0 = float(X0[0])
    ds = 200
    
    tau0,n0 = track.convertXYtoTN([float(y0[0]),float(y0[1])])
    s0 = float(track.getSFromT(tau0))
    
     #find the polymonial fitting range, from -2 to s0+ds, ds is estimated from last step 
    if sol_x is None:
        ds = 100
    else:
        ds = float(sol_x[0,-1]) + cd / 50*dt     
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
        if sum_norm/200<0.5:
            break
        
    best_order = np.argmin(sum_array)+3    
    print(f"ds: {ds}, order: {best_order}, sum norm {sum_array[best_order-3]}")   
    coeff = np.polyfit(x_axis,pos,best_order)
    #weight = np.linspace(2,0.5,len(x_axis))
    
    s_new,n_new = convertXYtoTN(coeff,y0[0:2],s0)
    X0 = casadi.veccat(float(s_new),float(n_new),float(y0[2]),float(y0[3]),float(y0[4]),float(y0[5]),float(y0[6]))
    
    #construct the symbolic polynomial    
    s = casadi.MX.sym('s')
    pos_mx = coeff[0,:]*casadi.power(s,best_order)
    for i in range(1,best_order):
        pos_mx += coeff[i,:] * casadi.power(s,best_order-i)
    pos_mx += coeff[-1,:]
    
    jac = casadi.jacobian(pos_mx,s)
    hes = casadi.jacobian(jac,s)
    kappa_mx = (jac[0]*hes[1]-jac[1]*hes[0])/casadi.power(casadi.norm_2(jac),3)
    phi_mx = casadi.arctan2(jac[1],jac[0])
    f_kappa = casadi.Function('kappa',[s],[kappa_mx])
    f_phi = casadi.Function('phi',[s],[phi_mx])
    
    #plt.plot(track.center_line[:,0],track.center_line[:,1])
    #polyval_x = np.polyval(coeff[:,0],x_axis)
    #polyval_y = np.polyval(coeff[:,1],x_axis)
    
    #plt.plot(polyval_x,polyval_y,'-*r')
    #plt.show()
    
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    X_dot = opti.variable(nx,N)
    U = opti.variable(nu,N)
    
    s_sym_array = X[0,:]
    n_sym_array = X[1,:]
    phi_sym_array = X[2,:]
    vx_sym_array = X[3,:]
    vy_sym_array = X[4,:]
    omega_sym_array = X[5,:]
    delta_sym_array = X[6,:]
    
    # control input
    delta_dot_sym_array = U[0,:]
    front_throttle_array = U[1,:]
    rear_throttle_array=U[2,:]

    kappa_sym_array = f_kappa(s_sym_array[0:-1])
    dphi_c_sym_array = phi_sym_array[0:-1]  - f_phi(s_sym_array[0:-1])
    
    fx_f_sym_array = cd*front_throttle_array - cm0-cm1*vx_sym_array[0:-1]
    fx_r_sym_array = cd*rear_throttle_array - cm0-cm1*vx_sym_array[0:-1]
    #tangent_vec_sym_array = track.getTangentVec(tau_sym_array[0:-1]) 

    #tangent_vec_norm = (tangent_vec_sym_array[0,:]*tangent_vec_sym_array[0,:]+tangent_vec_sym_array[1,:]*tangent_vec_sym_array[1,:])**0.5

    alpha_f = -casadi.atan2(omega_sym_array[0:-1]*lf+vy_sym_array[0:-1], vx_sym_array[0:-1]) + delta_sym_array[0:-1]
    alpha_r = casadi.atan2(omega_sym_array[0:-1]*lr-vy_sym_array[0:-1],vx_sym_array[0:-1])

    fy_f_sym_array = front_tire_model.getLateralForce(alpha_f,Fz[0])
    fy_r_sym_array = rear_tire_model.getLateralForce(alpha_r,Fz[1])
    
    front_rate = casadi.fmax((fy_f_sym_array*fy_f_sym_array+fx_f_sym_array*fx_f_sym_array)/(Fy_f_max*Fy_f_max),1)
    rear_rate = casadi.fmax((fy_r_sym_array*fy_r_sym_array+fx_r_sym_array*fx_r_sym_array)/(Fy_r_max*Fy_r_max),1)

    #fy_f_sym_array=fy_f_sym_array/(front_rate**0.5)
    #fy_r_sym_array=fy_r_sym_array/(rear_rate**0.5)
    #fx_f_sym_array=fx_f_sym_array/(front_rate**0.5)
    #fx_r_sym_array=fx_r_sym_array/(rear_rate**0.5)

    



    X_dot[0,:] = (vx_sym_array[0:-1]*casadi.cos(dphi_c_sym_array)-vy_sym_array[0:-1]*casadi.sin(dphi_c_sym_array))/(1-n_sym_array[0:-1]*kappa_sym_array) #s_dot
    X_dot[1,:] = vx_sym_array[0:-1]*casadi.sin(dphi_c_sym_array)+vy_sym_array[0:-1]*casadi.cos(dphi_c_sym_array)/(1-n_sym_array[0:-1]*kappa_sym_array)  # n_dot
    X_dot[2,:] = omega_sym_array[0:-1] #phi_dot

    X_dot[4,:] = 1/mass * (fy_r_sym_array + fx_f_sym_array*casadi.sin(delta_sym_array[0:-1]) + fy_f_sym_array*casadi.cos(delta_sym_array[0:-1]) - mass*vx_sym_array[0:-1]*omega_sym_array[0:-1])  #vydot    
    X_dot[3,:] = 1/mass * (fx_r_sym_array + fx_f_sym_array*casadi.cos(delta_sym_array[0:-1]) - fy_f_sym_array*casadi.sin(delta_sym_array[0:-1]) + mass*vy_sym_array[0:-1]*omega_sym_array[0:-1])  #vxdot    
    X_dot[5,:] = 1/Iz * (fy_f_sym_array*lf*casadi.cos(delta_sym_array[0:-1]) + fx_f_sym_array*lf*casadi.sin(delta_sym_array[0:-1]) - fy_r_sym_array*lr) #omegadot
    X_dot[6,:] = delta_dot_sym_array
    
       #objective
           
    #opti.minimize(-50*(s_sym_array[-1]))
        #+ casadi.dot(delta_dot_sym_array,delta_dot_sym_array)*0.00001/N 
        #+ n_sym_array[-1]*n_sym_array[-1]*0.00001/N)
    n_obj = (casadi.atan(5*(n_sym_array**2-(track_width/2)**2))+casadi.pi/2)*10
    opti.minimize(-1000*(s_sym_array[-1]) + casadi.dot(n_obj,n_obj))
    
    #initial condition
    opti.subject_to(X[:,0] == X0)


    #dynamics
    opti.subject_to(X[:,1:]==X[:,0:-1] + dt*X_dot)

    #state bound
    #opti.subject_to(opti.bounded(0.0,vx_sym_array,v_max))
    #opti.subject_to(opti.bounded(-track_width/2,n_sym_array[0:-1],track_width/2))
    #opti.subject_to(opti.bounded(-track_width/4,n_sym_array[int(-N/3):],track_width/4))
    opti.subject_to(opti.bounded(delta_min,delta_sym_array,delta_max))
     
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot_sym_array,delta_dot_max))
    opti.subject_to(opti.bounded(-2,front_throttle_array,1) ) 
    opti.subject_to(opti.bounded(-2,rear_throttle_array,1) ) 

    #opti.subject_to(fy_f_sym_array*fy_f_sym_array + fx_f_sym_array*fx_f_sym_array<Fy_f_max*Fy_f_max)  
    #opti.subject_to(fy_r_sym_array*fy_r_sym_array + fx_r_sym_array*fx_r_sym_array<Fy_r_max*Fy_r_max)
    
    
    #initial guess:
      
    if sol_x is None:
        for i in range(N):
            guess_X[:,i]=X0
    else:        
        guess_X[:,0]=X0
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=sol_u[:,-1]  
        yt = y0  
        for i in range(1,N+1):            
            yt = yt+dt*forwardModel(yt,guess_U[:,i-1])
            t_guess,n_guess = convertXYtoTN(coeff,yt[0:2],s0)
            guess_X[:,i] = casadi.vertcat(float(t_guess),float(n_guess),float(yt[2]),float(yt[3]),float(yt[4]),float(yt[5]),float(yt[6]))
        
    opti.set_initial(X,guess_X)
    opti.set_initial(U,guess_U)
    print(guess_X)
    
    opti.solver("ipopt",{},option) # set numerical backend

    
    try:
        sol = opti.solve()   # actual solve        
               
    
    except Exception as e:
        print(e) 
        exit()
        

    computation_time=sol.stats()['t_proc_total']

    sol_s = sol.value(s_sym_array)
    sol_n = sol.value(n_sym_array)
    sol_phi = sol.value(phi_sym_array)
    sol_vx = sol.value(vx_sym_array)
    sol_vy = sol.value(vy_sym_array)
    
    
    sol_omega = sol.value(omega_sym_array)
    
    sol_front_fx = sol.value(fx_f_sym_array)
    sol_rear_fx = sol.value(fx_r_sym_array)

    sol_front_fy = sol.value(fy_f_sym_array)
    sol_rear_fy = sol.value(fy_r_sym_array)
    
    sol_steer = sol.value(delta_sym_array)

    #print(sol_vx)
    #print(sol_front_fx)
    #print(sol_n)
    #print(sol_steer)
    #print(sol_steer_dot)
    
    sol_x = sol.value(X)
    sol_u = sol.value(U)    
    
    plane_x = casadi.DM.zeros(nx,N+1)    
    plane_x[:,0] = y0
    for i in range(1,N+1):
        plane_x[:,i] = plane_x[:,i-1] + dt* forwardModel(plane_x[:,i-1],sol_u[:,i-1])
        
    '''
    guess_X[:,0:-1]=sol_x[:,1:]
    guess_X[:,-1]=sol_x[:,-1]
    guess_X[0,:]=guess_X[0,:]-guess_X[0,0]
    
    guess_U[:,0:-1]=sol_u[:,1:]
    guess_U[:,-1]=sol_u[:,-1]   
    '''
    
    sql_ptx = ','.join("{:0.2f}".format(float(plane_x[0,i])) for i in range(N+1))
    sql_pty = ','.join("{:0.2f}".format(float(plane_x[1,i])) for i in range(N+1))
    
    #trajectory_history.append(pts)
    #pos_history.append(pts[0,:])
    sql_phi = float(sol_phi[0])
    
    sql_vx=float(sol_vx[0])
    sql_vy=float(sol_vy[0])
    
    #sql_front_wheel_omega=float(sol_front_wheel_omega[0])*params['front_wheel_radius']
    #sql_rear_wheel_omega=float(sol_rear_wheel_omega[0])*params['rear_wheel_radius']
    
    sql_front_fx=float(sol_front_fx[0])
    sql_rear_fx=float(sol_rear_fx[0])
    sql_front_fy=float(sol_front_fy[0])
    sql_rear_fy=float(sol_rear_fy[0])
    
    #sql_front_wheel_lamb = float((-1+ params['front_wheel_radius']*sol_front_wheel_omega/sol_vx)[0])
    #sql_rear_wheel_lamb = float((-1+ params['rear_wheel_radius']*sol_rear_wheel_omega/sol_vx)[0])
    
    sql_front_wheel_alpha = float((-casadi.atan2(sol_omega*params['lf'] + sol_vy, sol_vx) + sol_steer)[0])
    sql_rear_wheel_alpha = float((casadi.atan2(sol_omega*params['lr'] - sol_vy,sol_vx))[0])

    sql_steer=float(sol_steer[0])       
    sql_omega=float(sol_omega[0])
    
    sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
    cur.execute(sql_query,
        (computation_time,sql_phi,sql_vx,sql_vy,sql_steer,sql_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_fx,sql_rear_fx,sql_front_fy,sql_rear_fy,s0,ds,float(best_order),sql_ptx,sql_pty,error))
    con.commit()    

    return plane_x[:,1]

    
if __name__=='__main__':
    total_time = 60
    #total_frame = int(total_time*N/T)
    i = 0
    #for i in range(total_frame):
    
    while s0<init_s0+track.max_s:
        y0 =optimize(y0)
        print(f"finished: {float((s0-init_s0)/track.max_s)*100:.2f}%")
        print(y0)
        i+=1
    
    #cur.execute("SELECT * FROM {}".format(table_name))
    #data = cur.fetchall()    
    #vehicle_animation.plot(track,data,params,T/N)
