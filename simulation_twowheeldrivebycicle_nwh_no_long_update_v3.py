import casadi
import yaml
from track import *
from dynamics_models import BicycleDynamicsModelTwoWheelDriveWithBrakeNWHFxInput
from tire_model import SimplePacTireMode
import sqlite3
from datetime import datetime
import vehicle_animation

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
init_tau = tau0
v0 = 1
phi0 =track.getPhiFromT(tau0)
#x = [t,n,phi,vx,vy,omega,steer]
X0 = casadi.DM([tau0,0,phi0,v0,0,0,0])

N = 20
nx = 7
nu = 3
#T = 2.0
#dt = T/N
#ds = T*v_max
t = np.linspace(0,1,N+1)


#data save to sql
con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = 'unity'+now.strftime("_%m_%d_%Y_%H_%M_%S")
cur.execute("CREATE TABLE {} (computation_time real,phi real, vx real, vy real, steer real,omega real,front_wheel_alpha real,rear_wheel_alpha real,front_fx real, rear_fx real,front_fy real, rear_fy real,ptx text,pty text,error text)".format(table_name))

#casadi options
option = {}
option['max_iter']=10000
option['tol'] = 1e-5
option['print_level']=0
option['linear_solver']='ma57'

one_step_option = {}
one_step_option['max_iter']=1000
one_step_option['print_level']=0

casadi_option={}
#casadi_option['print_time']=False

first_run = True

guess_X = casadi.DM.zeros(nx,N+1)
for i in range(N):
    guess_X[:,i]=X0
guess_U = casadi.DM.zeros(nu,N)

Fz = casadi.DM.ones(2)*params['m']/2*9.81

Fx_f_max = front_tire_model.getMaxLongitudinalForce(Fz[0])
Fx_r_max = rear_tire_model.getMaxLongitudinalForce(Fz[1])

Fy_f_max = front_tire_model.getMaxLateralForce(Fz[0])
Fy_r_max = rear_tire_model.getMaxLateralForce(Fz[1])

sol_x = None
sol_u = None


def optimize(X0,forward_N):  
    #define ocp 
    global first_run,guess_X,guess_U,tau0,sol_x,sol_u
    error = ''
    tau0 = float(X0[0])
    v0 = float(X0[3])
    ds = v0*2+5
    s0 = track.getSFromT(tau0)
    tau_T = track.getTFromS(ds+s0)
    
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    X_dot = opti.variable(nx,N)
    U = opti.variable(nu,N)
    
    T = opti.variable()
    dt = T/N
    
    
    #X_dot = opti.variable(nx,N)
    #fy_f_sym_array = opti.variable(N)
    #fy_r_sym_array = opti.variable(N)
    
    tau_sym_array = X[0,:]
    n_sym_array = X[1,:]
    phi_sym_array = X[2,:]
    vx_sym_array = X[3,:]
    vy_sym_array = X[4,:]
    omega_sym_array = X[5,:]
    delta_sym_array = X[6,:]
    
    # control input
    delta_dot_sym_array = U[0,:]
    fx_f_sym_array = U[1,:]
    fx_r_sym_array=U[2,:]

    kappa_sym_array = track.f_kappa(tau_sym_array[0:-1])
    dphi_c_sym_array = phi_sym_array[0:-1]  - track.getPhiSym(tau_sym_array[0:-1])
    tangent_vec_sym_array = track.getTangentVec(tau_sym_array[0:-1]) 

    tangent_vec_norm = (tangent_vec_sym_array[0,:]*tangent_vec_sym_array[0,:]+tangent_vec_sym_array[1,:]*tangent_vec_sym_array[1,:])**0.5

    alpha_f = -casadi.atan2(omega_sym_array[0:-1]*lf+vy_sym_array[0:-1], vx_sym_array[0:-1]) + delta_sym_array[0:-1]
    alpha_r = casadi.atan2(omega_sym_array[0:-1]*lr-vy_sym_array[0:-1],vx_sym_array[0:-1])

    fy_f_sym_array = front_tire_model.getLateralForce(alpha_f,Fz[0])
    fy_r_sym_array = rear_tire_model.getLateralForce(alpha_r,Fz[1])
    
    front_rate = casadi.fmax((fy_f_sym_array*fy_f_sym_array+fx_f_sym_array*fx_f_sym_array)/(Fy_f_max*Fy_f_max),1)
    rear_rate = casadi.fmax((fy_r_sym_array*fy_r_sym_array+fx_r_sym_array*fx_r_sym_array)/(Fy_r_max*Fy_r_max),1)

    fy_f_sym_array=fy_f_sym_array/(front_rate**0.5)
    fy_r_sym_array=fy_r_sym_array/(rear_rate**0.5)
    fx_f_sym_array=fx_f_sym_array/(front_rate**0.5)
    fx_r_sym_array=fx_r_sym_array/(rear_rate**0.5)

    #fy_f_sym_array = casadi.fmin(fy_f_temp,casadi.sign(fy_f_temp)*(Fy_f_max*Fy_f_max - fx_f_sym_array*fx_f_sym_array)**0.5)
    #fy_r_sym_array = casadi.fmin(fy_r_temp,casadi.sign(fy_r_temp)*(Fy_r_max*Fy_r_max - fx_r_sym_array*fx_r_sym_array)**0.5)


    X_dot[0,:] = (vx_sym_array[0:-1]*casadi.cos(dphi_c_sym_array)-vy_sym_array[0:-1]*casadi.sin(dphi_c_sym_array))/(tangent_vec_norm*(1-n_sym_array[0:-1]*kappa_sym_array)) #t_dot
    X_dot[1,:] = vx_sym_array[0:-1]*casadi.sin(dphi_c_sym_array)+vy_sym_array[0:-1]*casadi.cos(dphi_c_sym_array) # n_dot
    X_dot[2,:] = omega_sym_array[0:-1] #phi_dot

    X_dot[4,:] = 1/mass * (fy_r_sym_array + fx_f_sym_array*casadi.sin(delta_sym_array[0:-1]) + fy_f_sym_array*casadi.cos(delta_sym_array[0:-1]) - mass*vx_sym_array[0:-1]*omega_sym_array[0:-1])  #vydot    
    X_dot[3,:] = 1/mass * (fx_r_sym_array + fx_f_sym_array*casadi.cos(delta_sym_array[0:-1]) - fy_f_sym_array*casadi.sin(delta_sym_array[0:-1]) + mass*vy_sym_array[0:-1]*omega_sym_array[0:-1])  #vxdot    
    X_dot[5,:] = 1/Iz * (fy_f_sym_array*lf*casadi.cos(delta_sym_array[0:-1]) + fx_f_sym_array*lf*casadi.sin(delta_sym_array[0:-1]) - fy_r_sym_array*lr) #omegadot
    X_dot[6,:] = delta_dot_sym_array
    
       #objective
           
    opti.minimize(T+casadi.dot(vy_sym_array,vy_sym_array)*0.0001)
    
    opti.subject_to(T>0)
    #initial condition
    opti.subject_to(X[:,0] == X0)
    opti.subject_to(casadi.fabs(tau_sym_array[-1] - tau_T)<0.1)

    #dynamics
    opti.subject_to(X[:,1:]==X[:,0:-1] + dt*X_dot)

    #state bound
    opti.subject_to(opti.bounded(0.0,vx_sym_array,v_max))
    opti.subject_to(opti.bounded(-track_width/2,n_sym_array,track_width/2))
    #opti.subject_to(opti.bounded(-track_width/4,n_sym_array[-1],track_width/4))
    opti.subject_to(opti.bounded(delta_min,delta_sym_array,delta_max))
     
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot_sym_array,delta_dot_max))

    #opti.subject_to(casadi.fabs(fx_f_sym_array)<Fx_f_max)  
    #opti.subject_to(casadi.fabs(fx_r_sym_array)<Fx_r_max) 

    #opti.subject_to(fy_f_sym_array*fy_f_sym_array + fx_f_sym_array*fx_f_sym_array<Fy_f_max*Fy_f_max)  
    #opti.subject_to(fy_r_sym_array*fy_r_sym_array + fx_r_sym_array*fx_r_sym_array<Fy_r_max*Fy_r_max)
    
    #initial guess:
    opti.set_initial(X,guess_X)
    opti.set_initial(U,guess_U)
    
    opti.solver("ipopt",{},option) # set numerical backend
    
    try:
        sol = opti.solve()   # actual solve        
        #post processor
        first_run = False
        computation_time=sol.stats()['t_proc_total']

        sol_tau = sol.value(tau_sym_array)
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
        sol_steer_dot = sol.value(delta_dot_sym_array)

        #print(sol_vx)
        #print(sol_front_fx)
        #print(sol_n)
        #print(sol_steer)
        #print(sol_steer_dot)
        
        sol_x = sol.value(X)
        sol_u = sol.value(U)
        
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=sol_u[:,-1]
        guess_X[:,0:-1] = sol_x[:,1:]  
        #guess_X[:,-1] = sol_x[:,-1]+dt*model.update(sol_x[:,-1],sol_u[:,-1])
        print("T = {}".format(float(sol.value(T))))
       
    
    except Exception as e:
        print(e)
        if(first_run):
            exit()
        computation_time = -1   
        error = e.args[0]
        #if(e.args[0].find('Infeasible_Problem_Detected')!=-1):            
        #    first_run = True
        
        sol_x[:,0:-1]=sol_x[:,1:]
        sol_u[:,0:-1]=sol_u[:,1:]

        sol_tau = sol_x[0,:]
        sol_n = sol_x[1,:]
        sol_phi = sol_x[2,:]
        sol_vx = sol_x[3,:]
        sol_vy = sol_x[4,:]
        sol_omega = sol_x[5,:]
        sol_steer = sol_x[6,:]
        sol_front_fx = sol_u[1,:]
        sol_rear_fx = sol_u[2,:] 
        sol_front_fy = [0]
        sol_rear_fy = [0]                     

        guess_U = casadi.DM.zeros(nu,N)  
        guess_X[:,0] = sol_x[:,1]   
        for i in range(N):
            guess_X[:,i] = sol_x[:,1]
        
       
    pts = track.convertParameterToPos(sol_tau,sol_n,N+1)
    sql_ptx = ','.join("{:0.2f}".format(pts[i,0]) for i in range(int(len(pts))))
    sql_pty = ','.join("{:0.2f}".format(pts[i,1]) for i in range(int(len(pts))))
    
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
    
    sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
    cur.execute(sql_query,
        (computation_time,sql_phi,sql_vx,sql_vy,sql_steer,sql_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_fx,sql_rear_fx,sql_front_fy,sql_rear_fy,sql_ptx,sql_pty,error))
    con.commit()
    
    #X = [tau,n,phi,v]
    #temp_Xt = [float(sol_tau[forward_N]),float(sol_n[forward_N]),float(sol_phi[forward_N]),float(sol_vx[forward_N]),float(sol_vy[forward_N]),float(sol_omega[forward_N])]
    temp_Xt = sol_x[:,forward_N]
    return temp_Xt

    
if __name__=='__main__':
    total_time = 60
    #total_frame = int(total_time*N/T)
    i = 0
    #for i in range(total_frame):
    while tau0<init_tau+track.max_t:
        X0 =optimize(X0,1)
        print(f"finished: {(tau0-init_tau)/track.max_t*100:.2f}%")
        print(X0)
        i+=1
    
    #cur.execute("SELECT * FROM {}".format(table_name))
    #data = cur.fetchall()    
    #vehicle_animation.plot(track,data,params,T/N)
