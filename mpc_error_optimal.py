import casadi
from matplotlib.pyplot import figure
from track import *
from dynamics_models import *
from tire_model import *

def testBicycleKinecticModel():
    
    #define track
    track_width = 0.4    
    track = SymbolicTrack('tracks/simpleoval.csv',track_width)

    #define model
    with open('params/bicyle.yaml') as file:
        params = yaml.load(file)
    model = BicycleKineticModelByParametricArc(params,track)
    
    #parameters
    d_min = params['d_min']
    d_max = params['d_max']
    v_min = params['v_min']
    v_max = params['v_max']
    delta_min = params['delta_min']  # minimum steering angle [rad]
    delta_max = params['delta_max']

    #track_length = track.s_value[-1]
    track_length_tau = track.max_t

    #initial boundary
    tau0 = 3.0
    phi0 =track.getPhiFromT(tau0)
    X0 = casadi.DM([tau0,0,phi0,0])
    
    #ocp params
    N = 50
    nx = model.nx
    nu = model.nu

    #define ocp 
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    T = opti.variable()

    #X = [tau,n,phi,v]
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    v = X[3,:]

    # control input
    delta = U[0,:]
    d = U[1,:]

    #objective
    opti.minimize(T)
    dt = T/N
    
    #target boundary
    dtau = 4.0
    tau_t = (tau0+dtau)%track_length_tau
    phi_t = track.getPhiFromT(tau_t)

    for k in range(N):
        x_next = X[:,k] + dt*model.update(X[:,k],U[:,k])
        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        #x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(v_min,v,v_max))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))

    #target condition
    opti.subject_to(tau[-1]==tau_t) #position
    opti.subject_to(n[-1]==0) #position
    opti.subject_to(opti.bounded(phi_t-np.pi/6,phi[-1],phi_t+np.pi/6))#orientation
    
    #input bound
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(d_min,d,d_max))
    opti.subject_to(T>=0) # Time must be positive

    #solve
    option = {}
    option['max_iter']=30000
    opti.solver("ipopt",{},option) # set numerical backend
    sol = opti.solve()   # actual solve

    #post processor
    sol_tau = sol.value(tau)
    sol_n = sol.value(n)
    sol_t = sol.value(T)
    sol_phi = sol.value(phi)
    print('optimal time: '+str(sol_t))
    fig = plt.figure()
    track.plot(fig)

    pts = track.convertParameterToPos(sol_tau,sol_n)
    plt.plot(pts[:,0],pts[:,1])


    markersize = [80]
    #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
    #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
    sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))

    figure()
    #plt.plot(sol_tau,sol.value(delta),'-r')
    plt.plot(sol_tau,sol_phi,'--b')
    plt.plot(sol_tau,sol_phi_c,'-g')
    figure()
    plt.plot(sol.value(d),label="d")
    figure()
    plt.plot(sol.value(v),label="v")
    print(sol_phi)
    print(sol.value(delta))
    plt.show()
    

def testBicycleDynamicsModel():
    #x = [t,n,phi,vx,vy,omega]
    #u = [delta,d]
    
    #define track
    track_width = 0.12
    track = SymbolicTrack('tracks/slider.csv',track_width)

    #define tire mode
    with open('params/simple_tire_front.yaml') as file:
        front_tire_params = yaml.load(file)
    front_tire_model = SimpleElectricalDrivenTire(front_tire_params,False)

    with open('params/simple_tire_rear.yaml') as file:
        rear_tire_params = yaml.load(file)
    rear_tire_model = SimpleElectricalDrivenTire(rear_tire_params,True)
    

    #define model
    with open('params/bicyle.yaml') as file:
        params = yaml.load(file)
    model = BicycleDynamicsModelByParametricArc(params,track,front_tire_model,rear_tire_model)
    
    #parameters
    d_min = params['d_min']
    d_max = params['d_max']
    v_min = params['v_min']
    v_max = params['v_max']
    delta_min = params['delta_min']  # minimum steering angle [rad]
    delta_max = params['delta_max']

    #track_length = track.s_value[-1]
    track_length_tau = track.max_t

    #initial boundary
    tau0 = 38
    phi0 =track.getPhiFromT(tau0)
    X0 = casadi.DM([tau0,0,phi0,0.01,0,0])
    
    #ocp params
    N = 50
    nx = model.nx
    nu = model.nu

    #define ocp 
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    T = 1

    #X = [tau,n,phi,v]
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    vx = X[3,:]
    vy = X[4,:]
    omega = X[5,:]

    # control input
    delta = U[0,:]
    d = U[1,:]

    
    
    dt = T/N
    ds = T*v_max
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s
    
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t
    
    #target boundary
    
    #dtau = 2
    #tau_t = tau0+dtau
    
    
    vec_t = track.dsdt(tau_t,0)/casadi.norm_2(track.dsdt(tau_t,0))
    
    print(vec_t)
    phi_t = track.getPhiFromT(tau_t)
    final_align_deviate_angle = casadi.pi/10

    t = np.linspace(0,1,N+1)
    ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T
    
    #objective
    opti.minimize(casadi.sum2(ref_tau-tau)+10*(ref_tau[-1]-tau[-1]))
    #opti.minimize(ref_tau[-1]-tau[-1])
    for k in range(N):
        x_next = X[:,k] + dt*model.update(X[:,k],U[:,k])        
        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        #x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(v_min,vx,v_max))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))
    
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(d_min,d,d_max))
    
    option = {}
    option['max_iter']=30000
    option['tol'] = 1e-6
    opti.solver("ipopt",{},option) # set numerical backend
    sol = opti.solve()   # actual solve

    print(sol.stats()['t_proc_total'])
    print(sol.stats()['success'])
    #post processor
    sol_tau = sol.value(tau)
    sol_n = sol.value(n)
    sol_t = sol.value(T)
    sol_phi = sol.value(phi)
    print('optimal time: '+str(sol_t))
    fig = plt.figure()
    track.plot(fig)

    pts = track.convertParameterToPos(sol_tau,sol_n)
    plt.plot(pts[:,0],pts[:,1])


    markersize = [80]
    #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
    #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
    sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
    
    #print(casadi.cos(sol_phi[-1]))
    #print(casadi.sin(sol_phi[-1]))
    #print(np.cos(sol_phi_c[-1]))
    #print(np.sin(sol_phi_c[-1]))
    
    #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
    #print(casadi.cos(final_align_deviate_angle))
    
    
    figure()
    #plt.plot(sol_tau,sol.value(delta),'-r')
    plt.plot(t,sol_tau,'--b',label="tau")
    plt.legend()
    figure()
    plt.plot(t,sol_phi,'--b',label="vehicle phi")
    plt.plot(t,sol_phi_c,'-g',label="track phi")
    plt.legend()
    figure()
    plt.plot(t[0:-1],sol.value(d),label="throttle")
    plt.legend()
    figure()
    plt.plot(t,sol.value(vx),label="vx")
    plt.plot(t,sol.value(vy),label="vy")
    plt.legend()
    #print(sol_phi)
    #print(sol.value(delta))
    plt.show()

    
def testRacecarDynamicsModel():
    #x = [t,n,phi,vx,vy,omega,steer,throttle,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
    #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
    
    #define track
    track_width = 0.12   
    track = SymbolicTrack('tracks/slider.csv',track_width)

    #define tire mode
    with open('params/racecar_simple_tire_front.yaml') as file:
        front_tire_params = yaml.load(file)
    front_tire_model = SimplePacTireMode(front_tire_params)

    with open('params/racecar_simple_tire_rear.yaml') as file:
        rear_tire_params = yaml.load(file)
    rear_tire_model = SimplePacTireMode(rear_tire_params)
    

    #define model
    with open('params/racecar.yaml') as file:
        params = yaml.load(file)
    model = RacecarDynamicsModel(params,track,front_tire_model,rear_tire_model)
    
    #parameters
    d_min = params['d_min']
    d_max = params['d_max']
    v_min = params['v_min']
    v_max = params['v_max']
    delta_min = params['delta_min']  # minimum steering angle [rad]
    delta_max = params['delta_max'] 
   
    delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
    delta_dot_max = params['delta_dot_max']

   
    
    #track_length = track.s_value[-1]
    track_length_tau = track.max_t

    #initial boundary
    tau0 = 18
    phi0 =track.getPhiFromT(tau0)
    X0 = casadi.DM([tau0,0,phi0,0.5,0,0,0,0.5/params['wheel_radius'],0.5/params['wheel_radius'],0.5/params['wheel_radius'],0.5/params['wheel_radius']])
    
    #ocp params
    N = 100
    nx = model.nx
    nu = model.nu
    
    T = 1
    dt = T/N
    #dtau = 2
    ds = T*v_max
    t = np.linspace(0,1,N+1)

    option = {}
    option['max_iter']=30000
    option['tol'] = 1e-4
    option['alpha_red_factor']=0.6
    #option['       ']='yes'
    #option['perturb_dec_fact']=0.3
    #option['print_level']=0

    vehicle_width = 0.03
    vehicle_length = 0.04

    #define ocp 
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    X_dot = opti.variable(nx,N)
    
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    vx = X[3,:]
    vy = X[4,:]
    omega = X[5,:]
    delta = X[6,:]
    wheel_omega = X[7:,:]
    
    

    # control input
    delta_dot = U[0,:]
    d = U[1,:]
    brake = U[2:,:]
    
    
    #target boundary    
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s        
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t 
    ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T
    
    #objective
    #opti.minimize(casadi.dot(delta_dot,delta_dot)*0.1 + casadi.dot(d_dot,d_dot)*0.001 -10*(tau[-1]-ref_tau[-1]))
    #opti.minimize(-(tau[-1]-ref_tau[-1]))
    opti.minimize(casadi.dot(delta_dot,delta_dot)*0.01 -200*(tau[-1]-ref_tau[-1]) + 30*n[-1]*n[-1] + 0.1*casadi.dot(brake[0,:],brake[0,:]) +0.1*casadi.dot(brake[1,:],brake[1,:])+ 0.1*casadi.dot(brake[2,:],brake[2,:]) +0.1*casadi.dot(brake[3,:],brake[3,:]) )
    for k in range(N):
        X_dot[:,k] = model.update(X[:,k],U[:,k])        
        x_next = X[:,k] + dt*X_dot[:,k]       
        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        #x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(v_min,vx,v_max*1.1))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(0,wheel_omega,v_max/params['wheel_radius']))
    #opti.subject_to(wheel_omega[0,:] + wheel_omega[1,:] == wheel_omega[2,:] + wheel_omega[3,:])
    
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot,delta_dot_max))    
    opti.subject_to(opti.bounded(casadi.DM.zeros(4),brake,casadi.DM.ones(4)*0.5))  
    opti.subject_to(opti.bounded(d_min,d,d_max))  
    opti.subject_to(casadi.dot(brake[0,:],brake[1,:])==0)
    opti.subject_to(casadi.dot(brake[2,:],brake[3,:])==0)
    
    #solve
    #option = {}
    #option['max_iter']=3000
    #option['tol'] = 1e-6
    opti.solver("ipopt",{},option) # set numerical backend
    try:
        sol = opti.solve()
    except:
        #print('wheel omega:')
        #print(opti.debug.value(front_wheel_omega))
        #print('omega:')
        #print(opti.debug.value(omega))
        #print('vy:')
        #print(opti.debug.value(vy))
            
        sol_tau = opti.debug.value(tau)
        sol_n = opti.debug.value(n)        
        sol_phi = opti.debug.value(phi)
        #print('optimal time: '+str(sol_t))
        fig = plt.figure()
        track.plot(fig)

        pts = track.convertParameterToPos(sol_tau,sol_n)
        plt.plot(pts[:,0],pts[:,1])


        
        #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
        #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
        sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
        
        #print(casadi.cos(sol_phi[-1]))
        #print(casadi.sin(sol_phi[-1]))
        #print(np.cos(sol_phi_c[-1]))
        #print(np.sin(sol_phi_c[-1]))
        
        #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
        #print(casadi.cos(final_align_deviate_angle))

        figure()
        #plt.plot(sol_tau,sol.value(delta),'-r')
        plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
        plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
        plt.legend()
        figure()
        plt.plot(opti.debug.value(d),label="throttle")
        plt.legend()
        
        vy_value = opti.debug.value(vy)
        vx_value = opti.debug.value(vx)
        omega_value = opti.debug.value(omega)
        wheel_omega_value = opti.debug.value(wheel_omega)        
        
        figure()
        plt.plot(vx_value,label="vx")
        plt.plot(vy_value,label="vy") 
        plt.plot(wheel_omega_value[0,:]*params['wheel_radius'],label="front left wheel speed")
        plt.plot(wheel_omega_value[1,:]*params['wheel_radius'],label="front right wheel speed")
        plt.plot(wheel_omega_value[2,:]*params['wheel_radius'],label="rear left wheel speed")
        plt.plot(wheel_omega_value[3,:]*params['wheel_radius'],label="rear right wheel speed")
        plt.legend()
        
        x_dot_value = opti.debug.value(X_dot)
        figure()
        plt.plot(x_dot_value[7,:],label="front left wheel_omega_dot")
        plt.plot(x_dot_value[8,:],label="front right wheel_omega_dot")
        plt.plot(x_dot_value[9,:],label="rear left wheel_omega_dot")
        plt.plot(x_dot_value[10,:],label="rear right wheel_omega_dot")
        plt.legend()
        
        brake_value  = opti.debug.value(brake)
        figure()
        plt.plot(brake_value[0,:],label="front left brake")
        plt.plot(brake_value[1,:],label="front right brake")
        plt.plot(brake_value[2,:],label="rear left brake")
        plt.plot(brake_value[3,:],label="rear right brake")
        plt.legend()
        #print(sol.value(wheel_omega_dot))
        #print(opti.debug.value(X_dot))
        
        speed_at_wheel_fl = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
        speed_at_wheel_fr = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
        speed_at_wheel_rl = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
        speed_at_wheel_rr = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
        
        #slipping ratio & angle
        #need work
        delta_value = opti.debug.value(delta)
        alpha_fl = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value-omega_value*params['width']/2+0.01) + delta_value
        alpha_fr = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value+omega_value*params['width']/2+0.01) + delta_value
        alpha_rl = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value-omega_value*params['width']/2+0.01)
        alpha_rr = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value+omega_value*params['width']/2+0.01)       
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
        lamb_fl = -1 + params['wheel_radius']*wheel_omega_value[0,:]/speed_at_wheel_fl
        lamb_fr = -1 + params['wheel_radius']*wheel_omega_value[1,:]/speed_at_wheel_fr
        lamb_rl = -1 + params['wheel_radius']*wheel_omega_value[2,:]/speed_at_wheel_rl
        lamb_rr = -1 + params['wheel_radius']*wheel_omega_value[3,:]/speed_at_wheel_rr
        
              #lamb = casadi.fmax(lamb,-1)
        #lamb = casadi.fmin(lamb,1)
        figure()
        plt.plot(lamb_fl,label="front left wheel slip ratio")
        plt.plot(lamb_fr,label="front right wheel slip ratio")
        plt.plot(lamb_rl,label="rear left wheel slip ratio")
        plt.plot(lamb_rr,label="rear right wheel slip ratio")
        
        plt.plot(alpha_fl,label="front left wheel slip angle")
        plt.plot(alpha_fr,label="front right wheel slip angle")
        plt.plot(alpha_rl,label="rear left wheel slip angle")
        plt.plot(alpha_rr,label="rear right wheel slip angle")
        plt.legend()
        plt.show()
    
    
    #print(sol.stats())
    #post processor
    sol_tau = sol.value(tau)
    sol_n = sol.value(n)        
    sol_phi = sol.value(phi)
    #print('optimal time: '+str(sol_t))
    plt.subplots()
    ax = plt.subplot(1,1,1)
    track.plot(ax)

    pts = track.convertParameterToPos(sol_tau,sol_n)
    plt.plot(pts[:,0],pts[:,1])


    
    #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
    #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
    sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
    
    #print(casadi.cos(sol_phi[-1]))
    #print(casadi.sin(sol_phi[-1]))
    #print(np.cos(sol_phi_c[-1]))
    #print(np.sin(sol_phi_c[-1]))
    
    #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
    #print(casadi.cos(final_align_deviate_angle))

    figure()
    #plt.plot(sol_tau,sol.value(delta),'-r')
    plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
    plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
    plt.legend()
    figure()
    plt.plot(sol.value(d),label="throttle")
    plt.legend()
    
    vy_value = sol.value(vy)
    vx_value = sol.value(vx)
    omega_value = sol.value(omega)
    wheel_omega_value = sol.value(wheel_omega)        
    
    figure()
    plt.plot(vx_value,label="vx")
    plt.plot(vy_value,label="vy") 
    plt.plot(wheel_omega_value[0,:]*params['wheel_radius'],label="front left wheel speed")
    plt.plot(wheel_omega_value[1,:]*params['wheel_radius'],label="front right wheel speed")
    plt.plot(wheel_omega_value[2,:]*params['wheel_radius'],label="rear left wheel speed")
    plt.plot(wheel_omega_value[3,:]*params['wheel_radius'],label="rear right wheel speed")
    plt.legend()
    
    x_dot_value = sol.value(X_dot)
    figure()
    plt.plot(x_dot_value[7,:],label="front left wheel_omega_dot")
    plt.plot(x_dot_value[8,:],label="front right wheel_omega_dot")
    plt.plot(x_dot_value[9,:],label="rear left wheel_omega_dot")
    plt.plot(x_dot_value[10,:],label="rear right wheel_omega_dot")
    plt.legend()
    
    brake_value  = sol.value(brake)
    figure()
    plt.plot(brake_value[0,:],label="front left brake")
    plt.plot(brake_value[1,:],label="front right brake")
    plt.plot(brake_value[2,:],label="rear left brake")
    plt.plot(brake_value[3,:],label="rear right brake")
    plt.legend()
    #print(sol.value(wheel_omega_dot))
    #print(sol.value(X_dot))
    
    speed_at_wheel_fl = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
    speed_at_wheel_fr = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
    speed_at_wheel_rl = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
    speed_at_wheel_rr = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
    
    #slipping ratio & angle
    #need work
    delta_value = sol.value(delta)
    alpha_fl = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value-omega_value*params['width']/2+0.01) + delta_value
    alpha_fr = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value+omega_value*params['width']/2+0.01) + delta_value
    alpha_rl = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value-omega_value*params['width']/2+0.01)
    alpha_rr = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value+omega_value*params['width']/2+0.01)       
    #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
    lamb_fl = -1 + params['wheel_radius']*wheel_omega_value[0,:]/speed_at_wheel_fl
    lamb_fr = -1 + params['wheel_radius']*wheel_omega_value[1,:]/speed_at_wheel_fr
    lamb_rl = -1 + params['wheel_radius']*wheel_omega_value[2,:]/speed_at_wheel_rl
    lamb_rr = -1 + params['wheel_radius']*wheel_omega_value[3,:]/speed_at_wheel_rr
    
            #lamb = casadi.fmax(lamb,-1)
    #lamb = casadi.fmin(lamb,1)
    figure()
    plt.plot(lamb_fl,label="front left wheel slip ratio")
    plt.plot(lamb_fr,label="front right wheel slip ratio")
    plt.plot(lamb_rl,label="rear left wheel slip ratio")
    plt.plot(lamb_rr,label="rear right wheel slip ratio")
    
    plt.plot(alpha_fl,label="front left wheel slip angle")
    plt.plot(alpha_fr,label="front right wheel slip angle")
    plt.plot(alpha_rl,label="rear left wheel slip angle")
    plt.plot(alpha_rr,label="rear right wheel slip angle")
    plt.legend()
    plt.show()
     
    
def testRacecarDynamicsAlternativeModel():
    #x = [t,n,phi,vx,vy,omega,steer,throttle,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
    #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
    
    #define track
    track_width = 0.12   
    track = SymbolicTrack('tracks/slider.csv',track_width)

    #define tire mode
    with open('params/racecar_simple_tire_front.yaml') as file:
        front_tire_params = yaml.load(file)
    front_tire_model = SimplePacTireMode(front_tire_params)

    with open('params/racecar_simple_tire_rear.yaml') as file:
        rear_tire_params = yaml.load(file)
    rear_tire_model = SimplePacTireMode(rear_tire_params)
    

    #define model
    with open('params/racecar.yaml') as file:
        params = yaml.load(file)
    model = RacecarDynamicsModel(params,track,front_tire_model,rear_tire_model)
    
    #parameters
    d_min = params['d_min']
    d_max = params['d_max']
    v_min = params['v_min']
    v_max = params['v_max']
    delta_min = params['delta_min']  # minimum steering angle [rad]
    delta_max = params['delta_max'] 
   
    delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
    delta_dot_max = params['delta_dot_max']

   
    
    #track_length = track.s_value[-1]
    track_length_tau = track.max_t

    #initial boundary
    tau0 = 18
    phi0 =track.getPhiFromT(tau0)
    X0 = casadi.DM([tau0,0,phi0,0.5,0,0,0,0.5/params['wheel_radius'],0.5/params['wheel_radius'],0.5/params['wheel_radius'],0.5/params['wheel_radius']])
    
    #ocp params
    N = 100
    nx = model.nx
    nu = model.nu
    
    T = 1
    dt = T/N
    #dtau = 2
    ds = T*v_max
    t = np.linspace(0,1,N+1)

    option = {}
    option['max_iter']=30000
    option['tol'] = 1e-4
    option['alpha_red_factor']=0.8
    
    #option['print_level']=0

    vehicle_width = 0.03
    vehicle_length = 0.04

    #define ocp 
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    X_dot = opti.variable(nx,N)
    
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    vx = X[3,:]
    vy = X[4,:]
    omega = X[5,:]
    delta = X[6,:]
    wheel_omega = X[7:,:]
    
    

    # control input
    delta_dot = U[0,:]
    d = U[1,:]
    brake = U[2:,:]
    
    
    #target boundary    
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s        
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t 
    ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T
    
    #objective
    #opti.minimize(casadi.dot(delta_dot,delta_dot)*0.1 + casadi.dot(d_dot,d_dot)*0.001 -10*(tau[-1]-ref_tau[-1]))
    #opti.minimize(-(tau[-1]-ref_tau[-1]))
    #opti.minimize(casadi.dot(delta_dot,delta_dot)*0.01 -200*(tau[-1]-ref_tau[-1]) + 30*n[-1]*n[-1] + 0.1*casadi.dot(brake[0,:],brake[0,:]) +0.1*casadi.dot(brake[1,:],brake[1,:]))
    opti.minimize(casadi.dot(delta_dot,delta_dot)*0.1 -150*(tau[-1]-ref_tau[-1]) + 20*n[-1]*n[-1] + 0.05*casadi.dot(brake[0,:],brake[0,:]) +0.05*casadi.dot(brake[1,:],brake[1,:]) )
    for k in range(N):
        X_dot[:,k] = model.update(X[:,k],U[:,k])        
        x_next = X[:,k] + dt*X_dot[:,k]       
        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        #x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(v_min,vx,v_max))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(0,wheel_omega,1.2*v_max/params['wheel_radius']))
    opti.subject_to(wheel_omega[0,:] + wheel_omega[1,:] == wheel_omega[2,:] + wheel_omega[3,:])
    
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot,delta_dot_max))    
    opti.subject_to(opti.bounded(-0.5,brake,0.5)) 
    opti.subject_to(opti.bounded(d_min,d,d_max))  

    
    #solve
    #option = {}
    #option['max_iter']=3000
    #option['tol'] = 1e-6
    opti.solver("ipopt",{},option) # set numerical backend
    try:
        sol = opti.solve()
    except:
        #print('wheel omega:')
        #print(opti.debug.value(front_wheel_omega))
        #print('omega:')
        #print(opti.debug.value(omega))
        #print('vy:')
        #print(opti.debug.value(vy))
            
        sol_tau = opti.debug.value(tau)
        sol_n = opti.debug.value(n)        
        sol_phi = opti.debug.value(phi)
        #print('optimal time: '+str(sol_t))
        fig = plt.figure()
        track.plot(fig)

        pts = track.convertParameterToPos(sol_tau,sol_n)
        plt.plot(pts[:,0],pts[:,1])


        
        #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
        #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
        sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
        
        #print(casadi.cos(sol_phi[-1]))
        #print(casadi.sin(sol_phi[-1]))
        #print(np.cos(sol_phi_c[-1]))
        #print(np.sin(sol_phi_c[-1]))
        
        #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
        #print(casadi.cos(final_align_deviate_angle))

        figure()
        #plt.plot(sol_tau,sol.value(delta),'-r')
        plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
        plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
        plt.legend()
        figure()
        plt.plot(opti.debug.value(d),label="throttle")
        plt.legend()
        
        vy_value = opti.debug.value(vy)
        vx_value = opti.debug.value(vx)
        omega_value = opti.debug.value(omega)
        wheel_omega_value = opti.debug.value(wheel_omega)        
        
        figure()
        plt.plot(vx_value,label="vx")
        plt.plot(vy_value,label="vy") 
        plt.plot(wheel_omega_value[0,:]*params['wheel_radius'],label="front left wheel speed")
        plt.plot(wheel_omega_value[1,:]*params['wheel_radius'],label="front right wheel speed")
        plt.plot(wheel_omega_value[2,:]*params['wheel_radius'],label="rear left wheel speed")
        plt.plot(wheel_omega_value[3,:]*params['wheel_radius'],label="rear right wheel speed")
        plt.legend()
        
        x_dot_value = opti.debug.value(X_dot)
        figure()
        plt.plot(x_dot_value[7,:],label="front left wheel_omega_dot")
        plt.plot(x_dot_value[8,:],label="front right wheel_omega_dot")
        plt.plot(x_dot_value[9,:],label="rear left wheel_omega_dot")
        plt.plot(x_dot_value[10,:],label="rear right wheel_omega_dot")
        plt.legend()
        
        brake_value  = opti.debug.value(brake)
        figure()
        plt.plot(brake_value[0,:],label="front brake")
        plt.plot(brake_value[1,:],label="rear brake")
        #plt.plot(brake_value[2,:],label="rear left brake")
        #plt.plot(brake_value[3,:],label="rear right brake")
        plt.legend()
        #print(sol.value(wheel_omega_dot))
        #print(opti.debug.value(X_dot))
        
        speed_at_wheel_fl = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
        speed_at_wheel_fr = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
        speed_at_wheel_rl = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
        speed_at_wheel_rr = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
        
        #slipping ratio & angle
        #need work
        delta_value = opti.debug.value(delta)
        alpha_fl = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value-omega_value*params['width']/2+0.01) + delta_value
        alpha_fr = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value+omega_value*params['width']/2+0.01) + delta_value
        alpha_rl = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value-omega_value*params['width']/2+0.01)
        alpha_rr = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value+omega_value*params['width']/2+0.01)       
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
        lamb_fl = -1 + params['wheel_radius']*wheel_omega_value[0,:]/speed_at_wheel_fl
        lamb_fr = -1 + params['wheel_radius']*wheel_omega_value[1,:]/speed_at_wheel_fr
        lamb_rl = -1 + params['wheel_radius']*wheel_omega_value[2,:]/speed_at_wheel_rl
        lamb_rr = -1 + params['wheel_radius']*wheel_omega_value[3,:]/speed_at_wheel_rr
        
              #lamb = casadi.fmax(lamb,-1)
        #lamb = casadi.fmin(lamb,1)
        figure()
        plt.plot(lamb_fl,label="front left wheel slip ratio")
        plt.plot(lamb_fr,label="front right wheel slip ratio")
        plt.plot(lamb_rl,label="rear left wheel slip ratio")
        plt.plot(lamb_rr,label="rear right wheel slip ratio")
        
        plt.plot(alpha_fl,label="front left wheel slip angle")
        plt.plot(alpha_fr,label="front right wheel slip angle")
        plt.plot(alpha_rl,label="rear left wheel slip angle")
        plt.plot(alpha_rr,label="rear right wheel slip angle")
        plt.legend()
        plt.show()
    
    
    #print(sol.stats())
    #post processor
    sol_tau = sol.value(tau)
    sol_n = sol.value(n)        
    sol_phi = sol.value(phi)
    #print('optimal time: '+str(sol_t))
    fig, ax = plt.subplots()
    track.plot(ax)
    pts = track.convertParameterToPos(sol_tau,sol_n)
    plt.plot(pts[:,0],pts[:,1])
    sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
    
    figure()
    plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
    plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
    plt.legend()
    
    figure()
    plt.plot(sol.value(d),label="throttle")
    plt.legend()
    
    vy_value = sol.value(vy)
    vx_value = sol.value(vx)
    omega_value = sol.value(omega)
    wheel_omega_value = sol.value(wheel_omega)        
    
    figure()
    plt.plot(vx_value,label="vx")
    plt.plot(vy_value,label="vy") 
    plt.plot(wheel_omega_value[0,:]*params['wheel_radius'],label="front left wheel speed")
    plt.plot(wheel_omega_value[1,:]*params['wheel_radius'],label="front right wheel speed")
    plt.plot(wheel_omega_value[2,:]*params['wheel_radius'],label="rear left wheel speed")
    plt.plot(wheel_omega_value[3,:]*params['wheel_radius'],label="rear right wheel speed")
    plt.legend()
    
    x_dot_value = sol.value(X_dot)
    figure()
    plt.plot(x_dot_value[7,:],label="front left wheel_omega_dot")
    plt.plot(x_dot_value[8,:],label="front right wheel_omega_dot")
    plt.plot(x_dot_value[9,:],label="rear left wheel_omega_dot")
    plt.plot(x_dot_value[10,:],label="rear right wheel_omega_dot")
    plt.legend()
    
    brake_value  = sol.value(brake)
    figure()
    plt.plot(brake_value[0,:],label="front brake")
    plt.plot(brake_value[1,:],label="rear brake")
    #plt.plot(brake_value[2,:],label="rear left brake")
    plt.plot(brake_value[3,:],label="rear right brake")
    plt.legend()
    #print(sol.value(wheel_omega_dot))
    #print(sol.value(X_dot))
    
    speed_at_wheel_fl = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
    speed_at_wheel_fr = casadi.fmax((vy_value+omega_value*params['lf'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
    speed_at_wheel_rl = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value-omega_value*params['width']/2)**2,0.001)**0.5
    speed_at_wheel_rr = casadi.fmax((vy_value-omega_value*params['lr'])**2 + (vx_value+omega_value*params['width']/2)**2,0.001)**0.5
    
    #slipping ratio & angle
    #need work
    delta_value = sol.value(delta)
    alpha_fl = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value-omega_value*params['width']/2+0.01) + delta_value
    alpha_fr = -casadi.atan2(omega_value*params['lf'] + vy_value, vx_value+omega_value*params['width']/2+0.01) + delta_value
    alpha_rl = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value-omega_value*params['width']/2+0.01)
    alpha_rr = casadi.atan2(omega_value*params['lr'] - vy_value,vx_value+omega_value*params['width']/2+0.01)       
    #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
    lamb_fl = -1 + params['wheel_radius']*wheel_omega_value[0,:]/speed_at_wheel_fl
    lamb_fr = -1 + params['wheel_radius']*wheel_omega_value[1,:]/speed_at_wheel_fr
    lamb_rl = -1 + params['wheel_radius']*wheel_omega_value[2,:]/speed_at_wheel_rl
    lamb_rr = -1 + params['wheel_radius']*wheel_omega_value[3,:]/speed_at_wheel_rr
    
            #lamb = casadi.fmax(lamb,-1)
    #lamb = casadi.fmin(lamb,1)
    figure()
    plt.plot(lamb_fl,label="front left wheel slip ratio")
    plt.plot(lamb_fr,label="front right wheel slip ratio")
    plt.plot(lamb_rl,label="rear left wheel slip ratio")
    plt.plot(lamb_rr,label="rear right wheel slip ratio")
    
    plt.plot(alpha_fl,label="front left wheel slip angle")
    plt.plot(alpha_fr,label="front right wheel slip angle")
    plt.plot(alpha_rl,label="rear left wheel slip angle")
    plt.plot(alpha_rr,label="rear right wheel slip angle")
    plt.legend()
    plt.show()
     
        
    
def testBicycleTwoWheelDrive():
    #x = [t,n,phi,vx,vy,omega,steer,throttle,wheel_speed]
    #u = [delta,d]
    
    #define track
    track_width = 0.12   
    track = SymbolicTrack('tracks/slider.csv',track_width)

    #define tire mode
    with open('params/racecar_simple_tire_front.yaml') as file:
        front_tire_params = yaml.load(file)
    front_tire_model = SimplePacTireMode(front_tire_params)

    with open('params/racecar_simple_tire_rear.yaml') as file:
        rear_tire_params = yaml.load(file)
    rear_tire_model = SimplePacTireMode(rear_tire_params)
    
    #define tire mode
    #with open('params/simple_tire_front.yaml') as file:
    ##    front_tire_params = yaml.load(file)
    #front_tire_model = SimpleElectricalDrivenTire(front_tire_params,True)

    ##with open('params/simple_tire_rear.yaml') as file:
    #    rear_tire_params = yaml.load(file)
    #rear_tire_model = SimpleElectricalDrivenTire(rear_tire_params,False)
    

    #define model
    with open('params/racecar.yaml') as file:
        params = yaml.load(file)
    model = BicycleDynamicsModelTwoWheelDrive(params,track,front_tire_model,rear_tire_model)
    
    #parameters
    d_min = params['d_min']
    d_max = params['d_max']
    v_min = params['v_min']
    v_max = params['v_max']
    delta_min = params['delta_min']  # minimum steering angle [rad]
    delta_max = params['delta_max'] 

    d_dot_min = params['d_dot_min']
    d_dot_max = params['d_dot_max']
    delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
    delta_dot_max = params['delta_dot_max']

    #track_length = track.s_value[-1]
    track_length_tau = track.max_t

    #initial boundary
    tau0 = 21
    phi0 =track.getPhiFromT(tau0)
    X0 = casadi.DM([tau0,0,phi0,0.2,0,0,0,0,0.2/params['wheel_radius']])
    
    #ocp params
    N = 100
    nx = model.nx
    nu = model.nu
    
    T = 1
    dt = T/N
    #dtau = 2
    ds = T*v_max
    t = np.linspace(0,1,N+1)

    option = {}
    option['max_iter']=10000
    option['tol'] = 1e-1
    #option['nlp_scaling_method']='equilibration-based'
    #option['warm_start_entire_iterate']='yes'
    #option['accept_every_trial_step']='yes'
    option['fast_step_computation']='yes'
    option['alpha_red_factor']=0.3
    #option['mehrotra_algorithm']='yes'
    option['max_refinement_steps']=5
    option['accept_after_max_steps'] = 1
    
    vehicle_width = 0.03
    vehicle_length = 0.04

    #define ocp 
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    X_dot = opti.variable(nx,N)
    
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    vx = X[3,:]
    vy = X[4,:]
    omega = X[5,:]
    delta = X[6,:]
    d = X[7,:]
    wheel_omega = X[8,:]
    
    #wheel_omega_dot = X_dot[8,:]

    # control input
    delta_dot = U[0,:]
    d_dot = U[1,:]    
    
    #target boundary    
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s        
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t 
    ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T
    
    vec_t = track.getTangentVec(tau[-1])/casadi.norm_2(track.getTangentVec(tau[-1]))
    phi_vec = casadi.veccat(casadi.cos(phi[-1]),casadi.sin(phi[-1]))
    
    #objective
    opti.minimize(casadi.dot(delta_dot,delta_dot)*0.1 + casadi.dot(d_dot,d_dot)*0.001 -10*(tau[-1]-ref_tau[-1])- 10*casadi.dot(phi_vec,vec_t))
    #opti.minimize(-(tau[-1]-ref_tau[-1]))
    
    for k in range(N):
        X_dot[:,k] = model.update(X[:,k],U[:,k])        
        x_next = X[:,k] + dt*X_dot[:,k]
        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        #x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(v_min,vx,v_max*1.1))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(d_min,d,d_max))
    opti.subject_to(opti.bounded(0,wheel_omega,v_max/params['wheel_radius']))
    #opti.subject_to(wheel_omega>0)
        
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot,delta_dot_max))
    opti.subject_to(opti.bounded(d_dot_min,d_dot,d_dot_max))
    
    opti.solver("ipopt",{},option) # set numerical backend
    #sol = opti.solve()
    try:
       sol = opti.solve()
    except:
        print('wheel omega:')
        print(opti.debug.value(wheel_omega))
        print('omega:')
        print(opti.debug.value(omega))
        print('vy:')
        print(opti.debug.value(vy))
        pass

    print(sol.stats())
    #post processor
    sol_tau = sol.value(tau)
    sol_n = sol.value(n)
    sol_t = sol.value(T)
    sol_phi = sol.value(phi)
    print('optimal time: '+str(sol_t))
    fig = plt.figure()
    track.plot(fig)

    pts = track.convertParameterToPos(sol_tau,sol_n)
    plt.plot(pts[:,0],pts[:,1])


    markersize = [80]
    #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
    #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
    sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
    
    #print(casadi.cos(sol_phi[-1]))
    #print(casadi.sin(sol_phi[-1]))
    #print(np.cos(sol_phi_c[-1]))
    #print(np.sin(sol_phi_c[-1]))
    
    #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
    #print(casadi.cos(final_align_deviate_angle))

    figure()
    #plt.plot(sol_tau,sol.value(delta),'-r')
    plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
    plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
    plt.legend()
    figure()
    plt.plot(sol.value(d),label="throttle")
    plt.legend()
    figure()
    plt.plot(sol.value(vx),label="vx")
    plt.plot(sol.value(vy),label="vy")
    plt.plot(sol.value(wheel_omega)*params['wheel_radius'],label="wheel speed")
    plt.legend()
    figure()
    plt.plot(sol.value(X_dot)[8,:],label="wheel_omega_dot")
    plt.legend()
    #print(sol.value(wheel_omega_dot))
    print(sol.value(X_dot))
    
    
    lamb1 = -1+ params['wheel_radius']*sol.value(wheel_omega)/casadi.sqrt(casadi.fmax((sol.value(vy)+sol.value(omega)*params['lf'])**2 + sol.value(vx)**2,0.001))
    lamb2 = -1+ params['wheel_radius']*sol.value(wheel_omega)/casadi.sqrt(casadi.fmax((sol.value(vy)-sol.value(omega)*params['lr'])**2 + sol.value(vx)**2,0.001))
    
    alpha1 = -casadi.atan2(sol.value(omega)*params['lf'] + sol.value(vy), sol.value(vx)+0.01) + sol.value(delta)
    alpha2 = casadi.atan2(sol.value(omega)*params['lr'] - sol.value(vy),sol.value(vx)+0.01)   
    #lamb = casadi.fmax(lamb,-1)
    #lamb = casadi.fmin(lamb,1)
    figure()
    plt.plot(lamb1,label="front wheel slip ratio")
    plt.plot(lamb2,label="rear wheel slip ratio")
    plt.plot(alpha1,label="front wheel slip angle")
    plt.plot(alpha2,label="rear wheel slip angle")
    plt.legend()
    
    plt.show()

 
def testBicycleTwoWheelDriveWithBrake():
    #x = [t,n,phi,vx,vy,omega,steer,throttle,wheel_speed]
    #u = [delta,d]
    
    #define track
    track_width = 0.12   
    track = SymbolicTrack('tracks/slider.csv',track_width)

    #define tire mode
    with open('params/racecar_simple_tire_front.yaml') as file:
        front_tire_params = yaml.load(file)
    front_tire_model = SimplePacTireMode(front_tire_params)

    with open('params/racecar_simple_tire_rear.yaml') as file:
        rear_tire_params = yaml.load(file)
    rear_tire_model = SimplePacTireMode(rear_tire_params)

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

    #track_length = track.s_value[-1]
    track_length_tau = track.max_t

    #initial boundary
    tau0 = 24
    phi0 =track.getPhiFromT(tau0)
    X0 = casadi.DM([tau0,0,phi0,1.0,0,0,0,1.0/params['wheel_radius'],1.0/params['wheel_radius']])
    
    #ocp params
    N = 100
    nx = model.nx
    nu = model.nu
    
    T = 1.0
    dt = T/N
    #dtau = 2
    ds = T*v_max
    t = np.linspace(0,1,N+1)

    option = {}

    option['max_iter']=10000
    option['tol'] = 1e-6
    #option['nlp_scaling_method']='equilibration-based'
    #option['warm_start_entire_iterate']='yes'
    #option['accept_every_trial_step']='yes'
    #option['fast_step_computation']='yes'
    option['alpha_red_factor']=0.9
    #option['mehrotra_algorithm']='yes'
    #option['max_refinement_steps']=5
    #option['accept_after_max_steps'] = 100

    

    #define ocp 
    opti = casadi.Opti()
    X = opti.variable(nx,N+1)
    U = opti.variable(nu,N)
    X_dot = opti.variable(nx,N)
    
    tau = X[0,:]
    n = X[1,:]
    phi = X[2,:]
    vx = X[3,:]
    vy = X[4,:]
    omega = X[5,:]
    delta = X[6,:]
    #d = X[7,:]
    front_wheel_omega = X[7,:]
    rear_wheel_omega = X[8,:]
    #wheel_omega_dot = X_dot[8,:]

    # control input
    delta_dot = U[0,:]
    #d_dot = U[1,:]
    d = U[1,:]
    front_brake = U[2,:]
    rear_brake = U[3,:]    
    
    #target boundary    
    s0 = track.getSFromT(tau0%track.max_t)
    st = (s0 + ds)%track.max_s        
    tau_t = float(track.getTFromS(st))
    while tau_t<tau0:
        tau_t = tau_t + track.max_t 
    ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T
    
    vec_t = track.getTangentVec(tau[-1])/casadi.norm_2(track.getTangentVec(tau[-1]))
    phi_vec = casadi.veccat(casadi.cos(phi[-1]),casadi.sin(phi[-1]))
    
    #objective
    opti.minimize(casadi.dot(delta_dot,delta_dot)*0.1 -150*(tau[-1]-ref_tau[-1]) + 10*n[-1]*n[-1] + 0.1*casadi.dot(front_brake,front_brake) +0.1*casadi.dot(rear_brake,rear_brake) )
    #opti.minimize(-(tau[-1]-ref_tau[-1]) )
    
    for k in range(N):
        X_dot[:,k] = model.update(X[:,k],U[:,k])        
        x_next = X[:,k] + dt*X_dot[:,k]
        #k1 = model.update(X[:,k],U[:,k])
        #k2 = model.update(X[:,k]+dt/2*k1,U[:,k])
        #k3 = model.update(X[:,k]+dt/2*k2,U[:,k])
        #k4 = model.update(X[:,k]+dt*k3,U[:,k])
        #x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #dynamic function constraints
        opti.subject_to(X[:,k+1]==x_next)  
        
    #initial condition
    opti.subject_to(X[:,0] == X0)

    #state bound
    opti.subject_to(opti.bounded(v_min,vx,v_max*1.1))
    opti.subject_to(opti.bounded(-track_width/2,n,track_width/2))
    opti.subject_to(opti.bounded(delta_min,delta,delta_max))
    opti.subject_to(opti.bounded(d_min,d,d_max))
    opti.subject_to(opti.bounded(0,front_wheel_omega,v_max/params['wheel_radius']))
    opti.subject_to(opti.bounded(0,rear_wheel_omega,v_max/params['wheel_radius']))
    #opti.subject_to(wheel_omega>0)
        
    #target condition
    #opti.subject_to(opti.bounded(-track_width/4,n[-1],track_width/4)) #position
    #opti.subject_to((casadi.cos(phi[-1])*vec_t[0]+casadi.sin(phi[-1])*vec_t[1])>=casadi.cos(final_align_deviate_angle))
    
    #input bound
    opti.subject_to(opti.bounded(delta_dot_min,delta_dot,delta_dot_max))
    #opti.subject_to(opti.bounded(d_dot_min,d_dot,d_dot_max))
    opti.subject_to(opti.bounded(0,front_brake,0.5))
    opti.subject_to(opti.bounded(0,rear_brake,0.5))
    #opti.subject_to(casadi.dot(front_brake,d)==0)
    #opti.subject_to(casadi.dot(rear_brake,d)==0)
    
    opti.solver("ipopt",{},option) # set numerical backend
    #sol = opti.solve()
    try:
       sol = opti.solve()
    except:
        print('wheel omega:')
        print(opti.debug.value(front_wheel_omega))
        print('omega:')
        print(opti.debug.value(omega))
        print('vy:')
        print(opti.debug.value(vy))
        
        sol_tau = opti.debug.value(tau)
        sol_n = opti.debug.value(n)        
        sol_phi = opti.debug.value(phi)
        #print('optimal time: '+str(sol_t))
        fig = plt.figure()
        track.plot(fig)

        pts = track.convertParameterToPos(sol_tau,sol_n)
        plt.plot(pts[:,0],pts[:,1])


        markersize = [80]
        #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
        #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
        sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
        
        #print(casadi.cos(sol_phi[-1]))
        #print(casadi.sin(sol_phi[-1]))
        #print(np.cos(sol_phi_c[-1]))
        #print(np.sin(sol_phi_c[-1]))
        
        #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
        #print(casadi.cos(final_align_deviate_angle))

        figure()
        #plt.plot(sol_tau,sol.value(delta),'-r')
        plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
        plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
        plt.legend()
        figure()
        plt.plot(opti.debug.value(d),label="throttle")
        plt.legend()
        figure()
        plt.plot(opti.debug.value(vx),label="vx")
        plt.plot(opti.debug.value(vy),label="vy")
        plt.plot(opti.debug.value(front_wheel_omega)*params['wheel_radius'],label="front wheel speed")
        plt.plot(opti.debug.value(rear_wheel_omega)*params['wheel_radius'],label="rear wheel speed")
        plt.legend()
        figure()
        plt.plot(opti.debug.value(X_dot)[7,:],label="front wheel_omega_dot")
        plt.plot(opti.debug.value(X_dot)[8,:],label="rear wheel_omega_dot")
        plt.legend()
        
        figure()
        plt.plot(opti.debug.value(front_brake),label="front_brake")
        plt.plot(opti.debug.value(rear_brake),label="rear_brake")
        plt.legend()
        #print(sol.value(wheel_omega_dot))
        print(opti.debug.value(X_dot))
        
        
        lamb1 = -1+ params['wheel_radius']*opti.debug.value(front_wheel_omega)/casadi.sqrt(casadi.fmax((opti.debug.value(vy)+opti.debug.value(omega)*params['lf'])**2 + opti.debug.value(vx)**2,0.001))
        lamb2 = -1+ params['wheel_radius']*opti.debug.value(rear_wheel_omega)/casadi.sqrt(casadi.fmax((opti.debug.value(vy)-opti.debug.value(omega)*params['lr'])**2 + opti.debug.value(vx)**2,0.001))
        
        alpha1 = -casadi.atan2(opti.debug.value(omega)*params['lf'] + opti.debug.value(vy), opti.debug.value(vx)+0.01) + opti.debug.value(delta)
        alpha2 = casadi.atan2(opti.debug.value(omega)*params['lr'] - opti.debug.value(vy),opti.debug.value(vx)+0.01)   
        #lamb = casadi.fmax(lamb,-1)
        #lamb = casadi.fmin(lamb,1)
        figure()
        plt.plot(lamb1,label="front wheel slip ratio")
        plt.plot(lamb2,label="rear wheel slip ratio")
        plt.plot(alpha1,label="front wheel slip angle")
        plt.plot(alpha2,label="rear wheel slip angle")
        plt.legend()
        
        
        
        plt.show()
        return

    #print(sol.stats())
    #post processor
    sol_tau = sol.value(tau)
    sol_n = sol.value(n)
    sol_t = sol.value(T)
    sol_phi = sol.value(phi)    
    fig = plt.figure()
    track.plot(fig)

    pts = track.convertParameterToPos(sol_tau,sol_n)
    plt.plot(pts[:,0],pts[:,1])


    markersize = [80]
    #plt.scatter([float(p0[1])],[float(p0[2])],markersize)
    #plt.scatter([float(Xt[1])],[float(Xt[2])],markersize)
    sol_phi_c = np.array([track.getPhiFromT(t) for t in sol_tau]).reshape(len(sol_tau))
    
    #print(casadi.cos(sol_phi[-1]))
    #print(casadi.sin(sol_phi[-1]))
    #print(np.cos(sol_phi_c[-1]))
    #print(np.sin(sol_phi_c[-1]))
    
    #print(np.cos(sol_phi[-1])*np.cos(sol_phi_c[-1])+np.sin(sol_phi[-1])*np.sin(sol_phi_c[-1]))
    #print(casadi.cos(final_align_deviate_angle))

    figure()
    #plt.plot(sol_tau,sol.value(delta),'-r')
    plt.plot(sol_tau,sol_phi,'--b',label="vehicle phi")
    plt.plot(sol_tau,sol_phi_c,'-g',label="track phi")
    plt.legend()
    
    figure()
    plt.plot(sol.value(d),label="throttle")
    plt.legend()
    
    figure()
    plt.plot(sol.value(vx),label="vx")
    plt.plot(sol.value(vy),label="vy")
    plt.plot(sol.value(front_wheel_omega)*params['wheel_radius'],label="front wheel speed")
    plt.plot(sol.value(rear_wheel_omega)*params['wheel_radius'],label="rear wheel speed")
    plt.legend()
    
    figure()
    plt.plot(sol.value(X_dot)[7,:],label="front wheel_omega_dot")
    plt.plot(sol.value(X_dot)[8,:],label="rear wheel_omega_dot")
    plt.legend()
    
    figure()
    plt.plot(sol.value(front_brake),label="front_brake")
    plt.plot(sol.value(rear_brake),label="rear_brake")
    plt.legend()
    #print(sol.value(wheel_omega_dot))
    #print(sol.value(X_dot))
    
    
    lamb1 = -1+ params['wheel_radius']*sol.value(front_wheel_omega)/casadi.sqrt(casadi.fmax((sol.value(vy)+sol.value(omega)*params['lf'])**2 + sol.value(vx)**2,0.001))
    lamb2 = -1+ params['wheel_radius']*sol.value(rear_wheel_omega)/casadi.sqrt(casadi.fmax((sol.value(vy)-sol.value(omega)*params['lr'])**2 + sol.value(vx)**2,0.001))
    
    alpha1 = -casadi.atan2(sol.value(omega)*params['lf'] + sol.value(vy), sol.value(vx)+0.01) + sol.value(delta)
    alpha2 = casadi.atan2(sol.value(omega)*params['lr'] - sol.value(vy),sol.value(vx)+0.01)   
    #lamb = casadi.fmax(lamb,-1)
    #lamb = casadi.fmin(lamb,1)
    figure()
    plt.plot(lamb1,label="front wheel slip ratio")
    plt.plot(lamb2,label="rear wheel slip ratio")
    plt.plot(alpha1,label="front wheel slip angle")
    plt.plot(alpha2,label="rear wheel slip angle")
    plt.legend()
    
    
    
    plt.show()



if __name__=='__main__':
    testRacecarDynamicsAlternativeModel()
    #testRacecarDynamicsModel()
    #testBicycleTwoWheelDriveWithBrake()