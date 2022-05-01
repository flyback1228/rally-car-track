import casadi
from matplotlib.pyplot import figure
from numpy.core.fromnumeric import repeat, trace
from track import *
from dynamics_models import *
from tire_model import SimplePacTireMode
from tire_model import SimpleTire
from matplotlib import animation
import sqlite3
from datetime import datetime
import vehicle_animation

track_width = 6 
track = SymbolicTrack('tracks/temp.csv',track_width)

#define tire mode

with open('params/racecar_simple_tire_front.yaml') as file:
    front_tire_params = yaml.load(file)
front_tire_model = SimplePacTireMode(front_tire_params)

with open('params/racecar_simple_tire_rear.yaml') as file:
    rear_tire_params = yaml.load(file)
rear_tire_model = SimplePacTireMode(rear_tire_params)

"""
with open('params/simple_tire.yaml') as file:
    front_tire_params = yaml.load(file)
front_tire_model = SimpleTire(front_tire_params)

with open('params/simple_tire.yaml') as file:
    rear_tire_params = yaml.load(file)
rear_tire_model = SimpleTire(rear_tire_params)
"""

#define model
with open('params/racecar.yaml') as file:
    params = yaml.load(file)
model = BicycleDynamicsModelTwoWheelDriveWithBrake(params,track,front_tire_model,rear_tire_model)

#parameters
d_min = params['d_min']
d_max = params['d_max']
v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min']  # minimum steering angle [rad]
delta_max = params['delta_max'] 

#d_dot_min = params['d_dot_min']
#d_dot_max = params['d_dot_max']
delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
delta_dot_max = params['delta_dot_max']

vehicle_length = (params['lf']+params['lr'])*1.2
vehicle_width = params['width']


#track_length = track.s_value[-1]
#track_length_tau = track.max_t

#initial boundary
tau0 = 4
v0 = 10
phi0 =track.getPhiFromT(tau0)
#x = [t,n,phi,vx,vy,omega,steer,front_wheel_speed,rear_wheel_speed]
X0 = casadi.DM([tau0,0,phi0,v0,0,0,0,v0/params['wheel_radius'],v0/params['wheel_radius']])

#data save to sql
con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = 'global'+now.strftime("_%m_%d_%Y_%H_%M_%S")
cur.execute("CREATE TABLE {} (computation_time real,phi real, vx real, vy real, steer real,d real,omega real,front_wheel_omega real,rear_wheel_omega real,front_wheel_alpha real,rear_wheel_alpha real,front_wheel_lambda real,rear_wheel_lambda real,front_brake real, rear_brake real,ptx text,pty text,error text)".format(table_name))
#ocp params0.01
N = 30
nx = model.nx
nu = model.nu
T = 3.0
dt = T/N
ds = T*v_max
t = np.linspace(0,1,N+1)
#casadi options
option = {}
option['max_iter']=2000
option['tol'] = 1e-4
option['print_level']=0

one_step_option = {}
one_step_option['max_iter']=1000
one_step_option['print_level']=0

casadi_option={}
casadi_option['print_time']=False

"""
phi_history = []
trajectory_history = []
pos_history = []
vx_history = []
vy_history = []
steer_history =[]
d_history=[]
omega_history=[]
"""

#computation_time=[]
first_run = True

guess_X = casadi.DM.zeros(nx,N+1)
#guess_X[:,0]=X0
for i in range(N):
    guess_X[:,i]=X0
guess_U = casadi.DM.zeros(nu,N)



def oneStep(X0):    
    opti = casadi.Opti()
    X = opti.variable(nx)
    U = opti.variable(nu)
    
    #print(X0)
    tau0 = float(X0[0])    
    tau_t = tau0+0.1
    
    #objective
    #opti.minimize(casadi.sum2(ref_tau-tau)+10*(ref_tau[-1]-tau[-1]))
    opti.minimize(tau0-X[0])    
    k1 = model.update(X0,U)
    k2 = model.update(X0+dt/2*k1,U)
    k3 = model.update(X0+dt/2*k2,U)
    k4 = model.update(X0+dt*k3,U)
    #dynamic function constraints
    opti.subject_to(X == X0 + dt/6*(k1+2*k2+2*k3+k4) ) 
    
    #opti.subject_to(wheel_omega>0)
    #state bound
    opti.subject_to(opti.bounded(v_min,X[3],v_max*1.1))
    opti.subject_to(opti.bounded(-track_width/2,X[1],track_width/2))
    opti.subject_to(opti.bounded(delta_min,X[6],delta_max))    
    opti.subject_to(opti.bounded(0,X[7],v_max/params['wheel_radius']))
    opti.subject_to(opti.bounded(0,X[8],v_max/params['wheel_radius']))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,U[0],delta_dot_max))
    opti.subject_to(opti.bounded(d_min,U[1],d_max))   
    opti.subject_to(opti.bounded(0,U[2],1))   
    opti.subject_to(opti.bounded(0,U[3],1))   
    
    opti.solver("ipopt",casadi_option,one_step_option) # set numerical backend
    try:
        sol = opti.solve()   # actual solve    
        return sol.value(X),sol.value(U)
    except:
        return X0,casadi.DM([0,0,0,0])
 

def optimize(X0,forward_N):  
    #define ocp 
    global first_run,guess_X,guess_U
    error = ''
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    #X_dot = opti.variable(nx,N)
    
    #X = [tau,n,phi,v]
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    vx = X[3,:]
    vy = X[4,:]
    omega = X[5,:]
    delta = X[6,:]
    front_wheel_omega = X[7,:]
    rear_wheel_omega = X[8,:]
    
    # control input
    delta_dot = U[0,:]
    d = U[1,:]
    front_brake=U[2,:]
    rear_brake=U[3,:]
    #target boundary
    tau0 = float(X0[0])  
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s        
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t 
    
    ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T    
    vec_t = track.getTangentVec(tau[-1])/casadi.norm_2(track.getTangentVec(tau[-1]))
    phi_vec = casadi.veccat(casadi.cos(phi[-1]),casadi.sin(phi[-1]))
    
    #objective
    #opti.minimize(-10*tau[-1])
    #opti.minimize(casadi.dot(delta_dot,delta_dot)*0.01 + casadi.dot(n,n)*0.01-10*tau[-1])
    opti.minimize(-0.1*(tau[-1]-ref_tau[-1]) + casadi.dot(delta_dot,delta_dot)*0.001 + casadi.dot(front_brake,front_brake)*0.0001 +casadi.dot(rear_brake,rear_brake)*0.0001 +casadi.dot(d,d)*0.0001 )
    
    for k in range(N):
        #x_next = X[:,k] + dt*model.update(X[:,k],U[:,k])
        #X_dot[:,k] = model.update(X[:,k],U[:,k])               
        #x_next = X[:,k] + dt*X_dot[:,k]          
        k1 = model.update(X[:,k],U[:,k])
        k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        k4 = model.update(X[:,k]+dt*k3,U[:,k])
        x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(0,vx,v_max*1.1))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))
    #opti.subject_to(opti.bounded(-track_width/3,n[-1],track_width/3))
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(0,front_wheel_omega,v_max/params['wheel_radius']))
    opti.subject_to(opti.bounded(0,rear_wheel_omega,v_max/params['wheel_radius']))
    #opti.subject_to(wheel_omega>0)
        
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot,delta_dot_max))
    opti.subject_to(opti.bounded(d_min,d,d_max))
    opti.subject_to(opti.bounded(0,front_brake,1))
    opti.subject_to(opti.bounded(0,rear_brake,1))
    
    
    #if not first_run:
    #if not first_run:
    opti.set_initial(X,guess_X)
    opti.set_initial(U,guess_U)
    
    opti.solver("ipopt",{},option) # set numerical backend
    failure = False
    try:
        sol = opti.solve()   # actual solve        
        #post processor
        first_run = False
        sol_tau = sol.value(tau)
        sol_n = sol.value(n)
        sol_phi = sol.value(phi)
        sol_vx = sol.value(vx)
        sol_vy = sol.value(vy)
        sol_omega = sol.value(omega)
        sol_front_wheel_omega = sol.value(front_wheel_omega)
        sol_rear_wheel_omega = sol.value(rear_wheel_omega)
        sol_front_brake = sol.value(front_brake)
        sol_rear_brake = sol.value(rear_brake)
        
        sol_d = sol.value(d)
        sol_steer = sol.value(delta)
        
        sol_x = sol.value(X)
        sol_u = sol.value(U)
        
        #tau_history.append(float(sol_tau[0]))
        #n_history.append(float(sol_n[0]))
        for i in range(N):
            guess_X[:,i] = sol_x[:,1]   
        """
        xguess,uguess = oneStep(sol_x[:,-1])        
        guess_X[:,0:-1] = sol_x[:,1:]
        guess_X[:,-1] = xguess  
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=uguess
        """
        """
        x_last,u_last = oneStep(sol_x[:,-1])
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=sol_u[:,-1]
        guess_X[:,0:-1] = sol_x[:,1:]  
        guess_X[:,-1] = sol_x[:,-1]+dt*model.update(sol_x[:,-1],sol_u[:,-1])
        """
        
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
        
        sql_front_wheel_omega=float(sol_front_wheel_omega[0])*params['wheel_radius']
        sql_rear_wheel_omega=float(sol_rear_wheel_omega[0])*params['wheel_radius']

                
        sql_front_brake=float(sol_front_brake[0])
        sql_rear_brake=float(sol_rear_brake[0])
        
        sql_front_wheel_lamb = float((-1+ params['wheel_radius']*sol_front_wheel_omega/sol_vx)[0])
        sql_rear_wheel_lamb = float((-1+ params['wheel_radius']*sol_rear_wheel_omega/sol_vx)[0])
        
        sql_front_wheel_alpha = float((-casadi.atan2(sol_omega*params['lf'] + sol_vy, sol_vx) + sol_steer)[0])
        sql_rear_wheel_alpha = float((casadi.atan2(sol_omega*params['lr'] - sol_vy,sol_vx))[0])
  
               
        sql_steer=float(sol_steer[0])
        sql_d=float(sol_d[0])        
        sql_omega=float(sol_omega[0])
        computation_time=sol.stats()['t_proc_total']
        #cur.execute("CREATE TABLE {} (,ptx text,pty text)".format(table_name))

        sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
        cur.execute(sql_query,
            (computation_time,sql_phi,sql_vx,sql_vy,sql_steer,sql_d,sql_omega,sql_front_wheel_omega,sql_rear_wheel_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_wheel_lamb,sql_rear_wheel_lamb,sql_front_brake,sql_rear_brake,sql_ptx,sql_pty,error))
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
            sol_front_wheel_omega = guess_X[7,:]
            sol_rear_wheel_omega = guess_X[8,:]

            
            sol_d = guess_U[1,:]
            
            sol_front_brake = guess_U[2,:]
            sol_rear_brake = guess_U[3,:]
            
            sol_x = guess_X
            #sol_u = guess_U
            
            #xguess,uguess = oneStep(sol_x[:,-1])    
            
            guess_X = casadi.DM.ones(nx,N+1)
            guess_U = casadi.DM.zeros(nu,N)  
            guess_X[:,0] = sol_x[:,1]   
            for i in range(N):
                guess_X[:,i] = sol_x[:,1]   
            first_run = True
        
        else:
            #post processor
            sol_tau = opti.debug.value(tau)
            sol_n = opti.debug.value(n)
            sol_phi = opti.debug.value(phi)
            sol_vx = opti.debug.value(vx)
            sol_vy = opti.debug.value(vy)
            sol_omega = opti.debug.value(omega)
            sol_front_wheel_omega = opti.debug.value(front_wheel_omega)
            sol_rear_wheel_omega = opti.debug.value(rear_wheel_omega)
            sol_front_brake = opti.debug.value(front_brake)
            sol_rear_brake = opti.debug.value(rear_brake)
            sol_steer = opti.debug.value(delta)
            
            sol_d = opti.debug.value(d)
            sol_x = opti.debug.value(X)
            #sol_u = opti.debug.value(U)
            
            for i in range(N):
                guess_X[:,i] = sol_x[:,1]   
            """
            xguess,uguess = oneStep(sol_x[:,-1])    
            guess_X[:,0:-1] = sol_x[:,1:]
            guess_X[:,-1] = xguess 
            guess_U[:,0:-1]=sol_u[:,1:]
            guess_U[:,-1]=uguess
            """
        
          
        """
        guess_U[:,0:-1]=sol_u[:,1:]
        guess_U[:,-1]=sol_u[:,-1]
        guess_X[:,0:-1] = sol_x[:,1:]  
        guess_X[:,-1] = sol_x[:,-1]+dt*model.update(sol_x[:,-1],sol_u[:,-1])    
        """
       
        pts = track.convertParameterToPos(sol_tau,sol_n,N+1)
        sql_ptx = ','.join("{:0.2f}".format(pts[i,0]) for i in range(int(len(pts))))
        sql_pty = ','.join("{:0.2f}".format(pts[i,1]) for i in range(int(len(pts))))
        
        #trajectory_history.append(pts)
        #pos_history.append(pts[0,:])
        sql_phi = float(sol_phi[0])
        
        sql_vx=float(sol_vx[0])
        sql_vy=float(sol_vy[0])
        
        sql_front_wheel_omega=float(sol_front_wheel_omega[0])*params['wheel_radius']
        sql_rear_wheel_omega=float(sol_rear_wheel_omega[0])*params['wheel_radius']
        
        sql_front_brake=float(sol_front_brake[0])
        sql_rear_brake=float(sol_rear_brake[0])
        
        sql_front_wheel_lamb = float((-1+ params['wheel_radius']*sol_front_wheel_omega/sol_vx)[0])
        sql_rear_wheel_lamb = float((-1+ params['wheel_radius']*sol_rear_wheel_omega/sol_vx)[0])
        
        sql_front_wheel_alpha = float((-casadi.atan2(sol_omega*params['lf'] + sol_vy, sol_vx) + sol_steer)[0])
        sql_rear_wheel_alpha = float((casadi.atan2(sol_omega*params['lr'] - sol_vy,sol_vx))[0])   
               
        sql_steer=float(sol_steer[0])
        sql_d=float(sol_d[0])        
        sql_omega=float(sol_omega[0])
        
        sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
        cur.execute(sql_query,
            (computation_time,sql_phi,sql_vx,sql_vy,sql_steer,sql_d,sql_omega,sql_front_wheel_omega,sql_rear_wheel_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_wheel_lamb,sql_rear_wheel_lamb,sql_front_brake,sql_rear_brake,sql_ptx,sql_pty,error))
        con.commit()
        
        #X = [tau,n,phi,v]
        #temp_Xt = [float(sol_tau[forward_N]),float(sol_n[forward_N]),float(sol_phi[forward_N]),float(sol_vx[forward_N]),float(sol_vy[forward_N]),float(sol_omega[forward_N])]
        temp_Xt = sol_x[:,forward_N]
        return temp_Xt

    
if __name__=='__main__':
    total_time = 60
    total_frame = int(total_time*N/T)
    for i in range(total_frame):
        X0 =optimize(X0,1)
        print(f"finished: {i+1} of {total_frame}")
    
    cur.execute("SELECT * FROM {}".format(table_name))
    data = cur.fetchall()    
    vehicle_animation.plot(track,data,params,T/N)
