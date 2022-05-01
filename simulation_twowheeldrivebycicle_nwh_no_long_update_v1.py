import casadi
import yaml
from track import *
from tire_model import SimplePacTireMode
import sqlite3
from datetime import datetime
from dynamics_models import BicycleDynamicsModelTwoWheelDriveWithBrakeNWHFxInput

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


#define model
with open('params/racecar_nwh.yaml') as file:
    params = yaml.safe_load(file)
model = BicycleDynamicsModelTwoWheelDriveWithBrakeNWHFxInput(params,track,front_tire_model,rear_tire_model)


Fz = casadi.DM.ones(2)*params['m']*9.81/2
Fx_f_max = front_tire_model.getMaxLongitudinalForce(Fz[0])
Fx_r_max = rear_tire_model.getMaxLongitudinalForce(Fz[1])

#parameters
lf = params['lf']
lr = params['lr']
mass = params['m']
Iz = params['Iz']
vehicle_width = params['width']

v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min']  # minimum steering angle [rad]
delta_max = params['delta_max'] 

delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
delta_dot_max = params['delta_dot_max']

vehicle_length = (params['lf']+params['lr'])*1.2


#initial boundary
tau0 = 1
init_tau = tau0
v0 = 2
phi0 =track.getPhiFromT(tau0)
#x = [t,n,phi,vx,vy,omega,steer]
X0 = casadi.DM([tau0,0,phi0,v0,0,0,0])

#data save to sql
con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = 'unity'+now.strftime("_%m_%d_%Y_%H_%M_%S")
cur.execute("CREATE TABLE {} (computation_time real,phi real, vx real, vy real, steer real,omega real,front_wheel_alpha real,rear_wheel_alpha real,front_brake real, rear_brake real,ptx text,pty text,error text)".format(table_name))


N = 40
T = 2
dt = T/N
ds = T*v_max

nx = 7
nu = 3

#casadi options
option = {}
option['max_iter']=3000
option['tol'] = 1e-3
option['print_level']=0

one_step_option = {}
one_step_option['max_iter']=1000
one_step_option['print_level']=0

casadi_option={}
#casadi_option['print_time']=False


#computation_time=[]
first_run = True

guess_X = casadi.DM.zeros(nx,N+1)
#guess_X[:,0]=X0
for i in range(N):
    guess_X[:,i]=X0
guess_U = casadi.DM.zeros(nu,N)




def optimize(X0,forward_N):  
    #define ocp 
    global first_run,guess_X,guess_U,tau0
    error = ''
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    #X_dot = opti.variable(nx,N)
    
    #X = [tau,n,phi,v]
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
    fx_r_sym_array =U[2,:]

    #target boundary
    tau0 = float(X0[0])  
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s        
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t 
    
    
    #objective
    #opti.minimize(-10*tau[-1])
    #opti.minimize(casadi.dot(delta_dot,delta_dot)*0.01 + casadi.dot(n,n)*0.01-10*tau[-1])
    opti.minimize(-100*(tau_sym_array[-1]-tau_t) + casadi.dot(delta_dot_sym_array,delta_dot_sym_array)*0.0001)# +casadi.dot(d,d)*0.00001 )
    #initial condition
    opti.subject_to(X[:,0] == X0)

    for k in range(N):
        #x_next = X[:,k] + dt*model.update(X[:,k],U[:,k])
        #X_dot[:,k] = model.update(X[:,k],U[:,k])               
        #x_next = X[:,k] + dt*X_dot[:,k]          
        kappa_sym = track.f_kappa(tau_sym_array[k])
        dphi_c_sym = phi_sym_array[k]  - track.getPhiSym(tau_sym_array[k])
        tangent_vec_sym = track.getTangentVec(tau_sym_array[k]) 

        alpha_f = -casadi.atan2(omega_sym_array[k]*lf+vy_sym_array[k], vx_sym_array[k]) + delta_sym_array[k]
        alpha_r = casadi.atan2(omega_sym_array[k]*lr-vy_sym_array[k],vx_sym_array[k])

        fy_f_sym = front_tire_model.getLateralForce(alpha_f,Fz[0])
        fy_r_sym = rear_tire_model.getLateralForce(alpha_r,Fz[1])

        t_dot = (vx_sym_array[k]*casadi.cos(dphi_c_sym)-vy_sym_array[k]*casadi.sin(dphi_c_sym))/(casadi.norm_2(tangent_vec_sym)*(1-n_sym_array[k]*kappa_sym))
        n_dot = vx_sym_array[k]*casadi.sin(dphi_c_sym)+vy_sym_array[k]*casadi.cos(dphi_c_sym)        


        vy_dot = 1/mass * (fy_r_sym + fx_f_sym_array[k]*casadi.sin(delta_sym_array[k]) + fy_f_sym*casadi.cos(delta_sym_array[k]) - mass*vx_sym_array[k]*omega_sym_array[k])  #vydot    
        vx_dot = 1/mass * (fx_r_sym_array[k] + fx_f_sym_array[k]*casadi.cos(delta_sym_array[k]) - fy_f_sym*casadi.sin(delta_sym_array[k]) + mass*vy_sym_array[k]*omega_sym_array[k])  #vxdot    
        omega_dot = 1/Iz * (fy_f_sym*lf*casadi.cos(delta_sym_array[k]) + fx_f_sym_array[k]*lf*casadi.sin(delta_sym_array[k]) - fy_r_sym*lr) #omegadot

        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        x_next = X[:,k] + dt*casadi.veccat(
            t_dot,n_dot,omega_sym_array[k],
            vx_dot,
            vy_dot,
            omega_dot,
            delta_dot_sym_array[k]
        )
        #dynamic function constraints
        opti.subject_to(fy_f_sym*fy_f_sym + fx_f_sym_array[k]*fx_f_sym_array[k]<Fx_f_max*Fx_f_max)  
        opti.subject_to(fy_r_sym*fy_r_sym + fx_r_sym_array[k]*fx_r_sym_array[k]<Fx_r_max*Fx_r_max)  
        opti.subject_to(X[:,k+1]==x_next)  
        
    

    #state bound
    opti.subject_to(opti.bounded(0,vx_sym_array,v_max))
    opti.subject_to(opti.bounded(-track_width/2,n_sym_array,track_width/2))
    opti.subject_to(opti.bounded(delta_min,delta_sym_array,delta_max))
    
    #opti.subject_to(wheel_omega>0)
        
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot_sym_array,delta_dot_max))
    #opti.subject_to(opti.bounded(-Fx_f_max,fx_f_sym_array,Fx_f_max))   
    #opti.subject_to(opti.bounded(-Fx_r_max,fx_r_sym_array,Fx_r_max))   
    #opti.subject_to(d*front_brake*rear_brake==0)
    
    #if not first_run:
    opti.set_initial(X,guess_X)
    opti.set_initial(U,guess_U)
    
    opti.solver("ipopt",{},option) # set numerical backend
    
    try:
        sol = opti.solve()   # actual solve        
        #post processor
        first_run = False
        sol_tau = sol.value(tau_sym_array)
        sol_n = sol.value(n_sym_array)
        sol_phi = sol.value(phi_sym_array)
        sol_vx = sol.value(vx_sym_array)
        sol_vy = sol.value(vy_sym_array)
        sol_omega = sol.value(omega_sym_array)
        #sol_front_wheel_omega = sol.value(front_wheel_omega)
        #sol_rear_wheel_omega = sol.value(rear_wheel_omega)
        sol_front_fx = sol.value(fx_f_sym_array)
        sol_rear_fx = sol.value(fx_r_sym_array)
        
        #sol_d = sol.value(d)
        sol_steer = sol.value(delta_sym_array)
        
        sol_x = sol.value(X)
        sol_u = sol.value(U)
        
        #tau_history.append(float(sol_tau[0]))
        #n_history.append(float(sol_n[0]))

        
        guess_U = casadi.DM.zeros(nu,N)  
        guess_X[:,0] = sol_x[:,1]   
        for i in range(N):
            guess_X[:,i] = sol_x[:,1]   
        
        """
        xguess,uguess = oneStep(sol_x[:,-1])        
        guess_X[:,0:-1] = sol_x[:,1:]
        guess_X[:,-1] = xguess  
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=uguess
        """
        
        
        #x_last,u_last = oneStep(sol_x[:,-1])
        #guess_U[:,0:-1]=sol_u[:,1:]
        #guess_U[:,-1]=sol_u[:,-1]
        #guess_X[:,0:-1] = sol_x[:,1:]  
        #guess_X[:,-1] = sol_x[:,-1]+dt*model.update(sol_x[:,-1],sol_u[:,-1])
         
        
        #print(sol_u)

        #fig, ax = plt.subplots()
        #track.plot(fig)

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
        
        
        #sql_front_wheel_lamb = float((-1+ params['front_wheel_radius']*sol_front_wheel_omega/sol_vx)[0])
        #sql_rear_wheel_lamb = float((-1+ params['rear_wheel_radius']*sol_rear_wheel_omega/sol_vx)[0])
        
        sql_front_wheel_alpha = float((-casadi.atan2(sol_omega*params['lf'] + sol_vy, sol_vx) + sol_steer)[0])
        sql_rear_wheel_alpha = float((casadi.atan2(sol_omega*params['lr'] - sol_vy,sol_vx))[0])
  
               
        sql_steer=float(sol_steer[0])
        #sql_d=float(sol_d[0])        
        sql_omega=float(sol_omega[0])
        computation_time=sol.stats()['t_proc_total']
        #cur.execute("CREATE TABLE {} (,ptx text,pty text)".format(table_name))

        sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
        cur.execute(sql_query,
            (computation_time,sql_phi,sql_vx,sql_vy,sql_steer,sql_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_fx,sql_rear_fx,sql_ptx,sql_pty,error))
        con.commit()
        temp_Xt = sol_x[:,forward_N]
        return temp_Xt
    
    except Exception as e:
        print(e)
        if(first_run):
            exit()
        computation_time = -1   
        error = e.args[0]
        if(e.args[0].find('Infeasible_Problem_Detected')!=-1):
            sol_tau = guess_X[0,:]
            sol_n = guess_X[1,:]
            sol_phi = guess_X[2,:]
            sol_vx = guess_X[3,:]
            sol_vy = guess_X[4,:]
            sol_omega = guess_X[5,:]
            sol_steer = guess_X[6,:]
            #sol_front_wheel_omega = guess_X[7,:]
            #sol_rear_wheel_omega = guess_X[8,:]

            
            #sol_d = guess_U[1,:]
            
            sol_front_fx = guess_U[1,:]
            sol_rear_fx = guess_U[2,:]
            
            sol_x = guess_X
            sol_u = guess_U
            
            #xguess,uguess = oneStep(sol_x[:,-1])    
            
            guess_X = casadi.DM.ones(nx,N+1)
            guess_U = casadi.DM.zeros(nu,N)  
            guess_X[:,0] = sol_x[:,1]   
            for i in range(N):
                guess_X[:,i] = sol_x[:,1]   
            first_run = True
        
        else:
            #post processor
            sol_tau = opti.debug.value(tau_sym_array)
            sol_n = opti.debug.value(n_sym_array)
            sol_phi = opti.debug.value(phi_sym_array)
            sol_vx = opti.debug.value(vx_sym_array)
            sol_vy = opti.debug.value(vy_sym_array)
            sol_omega = opti.debug.value(omega_sym_array)
            #sol_front_wheel_omega = opti.debug.value(front_wheel_omega)
            #sol_rear_wheel_omega = opti.debug.value(rear_wheel_omega)
            sol_front_fx = opti.debug.value(fx_f_sym_array)
            sol_rear_fx = opti.debug.value(fx_r_sym_array)

            sol_steer = opti.debug.value(delta_sym_array)
            
            #sol_d = opti.debug.value(d)
            sol_x = opti.debug.value(X)
            sol_u = opti.debug.value(U)
            

            #xguess,uguess = oneStep(sol_x[:,-1])    
            #guess_X[:,0:-1] = sol_x[:,1:]
            #guess_X[:,-1] = xguess 
            #guess_U[:,0:-1]=sol_u[:,1:]
            #guess_U[:,-1]=uguess
       
        
          
        
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=sol_u[:,-1]
        guess_X[:,0:-1] = sol_x[:,1:]  
        guess_X[:,-1] = sol_x[:,-1]+dt*model.update(sol_x[:,-1],sol_u[:,-1])    
        
       
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
        
        #sql_front_wheel_lamb = float((-1+ params['front_wheel_radius']*sol_front_wheel_omega/sol_vx)[0])
        #sql_rear_wheel_lamb = float((-1+ params['rear_wheel_radius']*sol_rear_wheel_omega/sol_vx)[0])
        
        sql_front_wheel_alpha = float((-casadi.atan2(sol_omega*params['lf'] + sol_vy, sol_vx) + sol_steer)[0])
        sql_rear_wheel_alpha = float((casadi.atan2(sol_omega*params['lr'] - sol_vy,sol_vx))[0])   

        sql_steer=float(sol_steer[0])
        #sql_d=float(sol_d[0])        
        sql_omega=float(sol_omega[0])
        
        sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
        cur.execute(sql_query,
            (computation_time,sql_phi,sql_vx,sql_vy,sql_steer,sql_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_fx,sql_rear_fx,sql_ptx,sql_pty,error))
        con.commit()
        
        #X = [tau,n,phi,v]
        #temp_Xt = [float(sol_tau[forward_N]),float(sol_n[forward_N]),float(sol_phi[forward_N]),float(sol_vx[forward_N]),float(sol_vy[forward_N]),float(sol_omega[forward_N])]
        temp_Xt = sol_x[:,forward_N]
        return temp_Xt

    
if __name__=='__main__':
    total_time = 60
    total_frame = int(total_time*N/T)
    i = 0
    #for i in range(total_frame):
    while tau0<init_tau+track.max_t:
        X0 =optimize(X0,1)
        print(f"finished: {(tau0-init_tau)/track.max_t*100:.2f}%")
        print(X0)
        i+=1

