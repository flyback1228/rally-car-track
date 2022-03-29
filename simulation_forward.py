import casadi
from matplotlib.pyplot import figure
from track import *
from dynamics_models import *
from tire_model import *
    
def testBicycleTwoWheelDriveXYForward():
    #x = [px,py,phi,vx,vy,omega,steer,throttle,wheel_speed]
    #u = [delta,d]
    
    
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
    model = BicycleDynamicsModelTwoWheelDriveXY(params,front_tire_model,rear_tire_model)
    
    #parameters
    d_min = params['d_min']
    d_max = params['d_max']
    v_min = params['v_min']
    v_max = params['v_max']
    delta_min = params['delta_min']  # minimum steering angle [rad]
    delta_max = params['delta_max'] 


    delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
    delta_dot_max = params['delta_dot_max']

    X0 = casadi.DM([0,0,0,0.2,0,0,0.1,0,0.2/params['wheel_radius']])
    u = [0.0,1.1]
    
    
    
    #ocp params
    T = 80
    N = 200*T
    nx = model.nx
    nu = model.nu
    
    
    dt = T/N
    ds = T*v_max
    t = np.linspace(0,1,N+1)
    
    X = casadi.DM.zeros(nx,N+1)
    X_dot = casadi.DM.zeros(nx,N)
    
            
    X[:,0]=X0
    #X[:,0] = [0,0,0,0.2,0,0,0,0,0.2/params['wheel_radius']]
    for k in range(N):
        X_dot[:,k] = model.update(X[:,k],u)
        X[:,k+1] = X[:,k] + dt*X_dot[:,k]
        
    px = X[0,:].T
    py = X[1,:].T
    phi = X[2,:].T
    vx = X[3,:].T
    vy = X[4,:].T
    omega = X[5,:].T
    delta = X[6,:].T
    d = X[7,:].T
    wheel_omega = X[8,:].T   
    
    omega_dot = X_dot[5,:].T

    Fz = casadi.DM.ones(2)*4*params['m']*9.81
       
    
    figure()
    plt.plot(px,py,'--b',label="vehicle trajectory")
    plt.legend()
    
    
    
    figure()
    plt.plot(vx,label="vx")
    plt.plot(vy,label="vy")
    plt.plot(wheel_omega*params['wheel_radius'],label="wheel speed")
    plt.legend()
    
    figure()
    plt.plot(X_dot[8,:].T,label="wheel_omega_dot")
    plt.legend()
        
    lamb1 = -1+ params['wheel_radius']*wheel_omega/casadi.sqrt(casadi.fmax(vy+omega*params['lf']**2 + vx**2,0.001))
    lamb2 = -1+ params['wheel_radius']*wheel_omega/casadi.sqrt(casadi.fmax(vy-omega*params['lr']**2 + vx**2,0.001))
    
    alpha1 = -casadi.atan2(omega*params['lf'] + vy, vx+0.01) + delta
    alpha2 = casadi.atan2(omega*params['lr'] - vy,vx+0.01)   
    
    Ff = front_tire_model.getForce(lamb1,alpha1,Fz[0])
    Fr = rear_tire_model.getForce(lamb2,alpha2,Fz[0])  
    length = int(Ff.size()[0]/2)
    Ff = Ff.reshape((length,2))
    print(Ff.size())
    Fr = Fr.reshape((length,2))   
    Ff_x = Ff[:,0]
    Ff_y= Ff[:,1]
    Fr_x = Fr[:,0]
    Fr_y= Fr[:,1]
    figure()
    
    print(Ff_x.size())
    
    plt.plot(Ff_x,label="Ff_x")
    plt.plot(Ff_y,label="Ff_y")
    plt.plot(Fr_x,label="Fr_x")
    plt.plot(Fr_y,label="Fr_y")
    plt.legend()
    #lamb = casadi.fmax(lamb,-1)
    #lamb = casadi.fmin(lamb,1)
    figure()
    plt.plot(lamb1,label="front wheel slip ratio")
    plt.plot(lamb2,label="rear wheel slip ratio")
    plt.plot(alpha1,label="front wheel slip angle")
    plt.plot(alpha2,label="rear wheel slip angle")
    plt.legend()
    
    figure()
    plt.plot(omega,label="omega")
    plt.plot(omega_dot,label="omega_dot")
    plt.legend()
    
    
    
    plt.show()

if __name__=='__main__':
    #testRacecarDynamicsModel()
    testBicycleTwoWheelDriveXYForward()