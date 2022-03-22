from cmath import pi
import json
import asyncio
import websockets
import csv
from track import SymbolicTrack
import yaml
from dynamics_models import BicycleKineticModelWithSteerDot
import time
import casadi
import numpy as np
import sqlite3
from datetime import datetime

track_width = 2.0
track = SymbolicTrack('tracks/temp.csv',track_width)
with open('params/racecar.yaml') as file:
        params = yaml.load(file)

d_min = params['d_min']
d_max = params['d_max']
v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min'] 
delta_max = params['delta_max']
delta_dot_min = params['delta_dot_min']  # minimum steering angle [rad]
delta_dot_max = params['delta_dot_max']
model = BicycleKineticModelWithSteerDot(params,track)

real_states=[]
real_inputs=[]
estimated_states=[]
predicated_states=[]

steering = 0
throttle = 0

N = 20
T = 2.0
dt = T/N
nx = model.nx
nu = model.nu
estimate_dt = dt+0.05

x_guess = casadi.DM.zeros(nx,N+1)
u_guess = casadi.DM.zeros(nu,N)

last_mpc_x = casadi.DM.zeros(nx,N+1)
last_mpc_u = casadi.DM.zeros(nu,N)

#solve options

option = {} 
option['print_level']=0

con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = now.strftime("_%m_%d_%Y_%H_%M_%S")

cur.execute("CREATE TABLE {} (unity_sent_time real, python_receved_time real, python_sent_time real, unity_received_type real,real_x real,real_y real,real_psi real,real_v real,received_steer real, received_throttle real, sent_steer real, sent_throttle real)".format(table_name))

def _check_mode(msg):
    return  "Auto" if msg and msg[:2] == '42' else "Manual"

def _parse_telemetry(msg):
    msg_json = msg[2:]
    parsed = json.loads(msg_json)
    
    msg_type = parsed[0]
    assert msg_type == 'telemetry' or msg_type == 'waypoints', "Invalid message type {}".format(msg_type) 
    
    if(msg_type == 'waypoints'):
        global track,model
        values = parsed[1] 
        x = values['ptsx']
        y = values['ptsy']
        with open('tracks/temp.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(x)):
                writer.writerow([x[i],y[i]])

        track = SymbolicTrack('tracks/temp.csv',track_width)        
        model = BicycleKineticModelWithSteerDot(params,track)
        return False,values
        

    # Telemetry values
    values = parsed[1]    
    return True,values

    
async def control_loop(ws):
    global steering,throttle,last_mpc_u,last_mpc_x,x_guess,u_guess,option,estimate_dt
    async for message in ws:
        
        
            
        if _check_mode(message) == 'Auto':                
            is_telemtry,telemetry  = _parse_telemetry(message)
            if(is_telemtry):    

                receive_time = int(1000*time.time())
                print('received at ' + str(receive_time))  
                phi0 = float(telemetry['psi'])
                #ptsx = telemetry['ptsx']
                #ptsy = telemetry['ptsy']receive_time
                v0 = float(telemetry['speed'])
                steering = float(telemetry['steering_angle'])
                throttle = float(telemetry['throttle'])
                x = float(telemetry['x'])
                y = float(telemetry['y'])

                
                

                #initial boundary
                real_tau,real_n = track.convertXYtoTN((x,y))
                real_state = casadi.DM([real_tau,real_n,phi0,v0,steering])
                real_u = casadi.DM([0,throttle])
                #real_states.append(real_state)
                #real_inputs.append(np.array([steering,throttle]))
                #print('real_state: '+str(real_state))current_time = time.time()
                

                #kappa = track.f_kappa(tau0)
                #phi_c = track.getPhiFromT(tau0)
                #print('phi_c: '+str(phi_c))
                #tangent_vec = track.getTangentVec(tau0)

                
                estimate_state = real_state + model.update(real_state,real_u)*estimate_dt
                #dtau = float(tau0 + v0*casadi.cos(phi0-phi_c+steering)/(casadi.norm_2(tangent_vec)*(1-n0*kappa)) *estimate_dt)
                #dn = float(n0 + v0*casadi.sin(phi0-phi_c+steering) *estimate_dt)
                #dphi = float(phi0 + v0/(params['lf']+params['lr']) * casadi.tan(steering)*estimate_dt)
                #dv = casadi.fmin(float(v0 + (5.5*throttle-0.1*v0)*estimate_dt),v_max)
                
                #estimated_state = np.array([dtau,dn,dphi,dv])
                #estimated_states.append(estimated_state)
                #print('estimated_state: '+str(estimated_state))
                                

                #if tau0<0:
                #    print("car is not on the track")
                #    return                
                
                                    
                #ocp params
                
                

                #define ocp 
                

                #X0 = casadi.DM(estimated_state)
                #print('X0=' + str(X0))

                #U0 = casadi.DM([0,throttle])   
                #print('U0=' + str(U0))         
                
                #x0_guess = casadi.DM(estimated_state)
                x0_guess = estimate_state
                x_guess[:,0] = x0_guess

                for k in range(N-1):
                    xt_guess = x0_guess + model.update(x0_guess,last_mpc_u[:,k+1])*estimate_dt
                    x_guess[:,k+1] = xt_guess
                    x0_guess = xt_guess
                #print(x_guess)
                x_guess[:,-1] = x_guess[:,-2] + model.update(x_guess[:,-2],last_mpc_u[:,-1])*estimate_dt
                u_guess[:,1:] = last_mpc_u[:,0:-1]
                u_guess[:,-1]=u_guess[:,-2]

                opti = casadi.Opti()
                X = opti.variable(nx,N+1)
                U = opti.variable(nu,N) 

                opti.set_initial(X,x_guess)
                opti.set_initial(U,u_guess)

                

                #X = [tau,n,phi,v]
                tau = X[0,:]
                n = X[1,:]
                phi = X[2,:]
                v = X[3,:]
                delta = X[4,:]
                # control input
                delta_dot = U[0,:]
                d = U[1,:]

                
                #dtau = 2
                ds = T*v_max/2
                #t = np.linspace(0,1,N+1)

                tau0 = float(estimate_state[0])
                n0 = float(estimate_state[1])
                v0 = float(estimate_state[3])
                #target boundary    
                s0 = track.getSFromT(tau0%track.max_t)
                st = (s0 + ds)%track.max_s        
                tau_t = float(track.getTFromS(st))
                while tau_t<tau0:
                    tau_t = tau_t + track.max_t 
                ref_tau = casadi.DM(np.linspace(tau0,tau_t,N+1)).T

                #objective
                opti.minimize(-0.4*(tau[-1]-ref_tau[-1]) + 0.0001*casadi.dot(delta_dot,delta_dot) + 0.0001*casadi.dot(n[-1],n[-1])  )
                #opti.minimize(-0.04*(tau[-1]-ref_tau[-1]))

                for k in range(N):
                    x_next = X[:,k] + estimate_dt*model.update(X[:,k],U[:,k])
                    opti.subject_to(X[:,k+1]==x_next)  
                    
                #initial condition
                opti.subject_to(X[:,0] == estimate_state)

                #state bound
                n_v = casadi.fmax(v_max,casadi.fabs(v0))
                n_c = casadi.fmax(track_width/2,casadi.fabs(n0))
                #print('nv = '+str(n_v))
                opti.subject_to(opti.bounded(v_min,v[0:-1],n_v+0.1))
                opti.subject_to(opti.bounded(v_min,v[-1],v_max))

                opti.subject_to(opti.bounded(-track_width/2,n[-1],track_width/2))
                opti.subject_to(opti.bounded(-n_c-0.1,n[0:-1],n_c+0.1))
                opti.subject_to(opti.bounded(delta_min,delta,delta_max))
                #print('nc='+str(n_c))
                
                #input bound
                opti.subject_to(opti.bounded(delta_dot_min,delta_dot,delta_dot_max))
                opti.subject_to(opti.bounded(d_min,d,d_max))                

                data={}

                opti.solver("ipopt",{},option) # set numerical backend
                try:        
                    sol = opti.solve()   # actual solve

                    #float(sol.value(delta)[1])
                    predicated_states.append(np.array(sol.value(X)[:,1]))

                    
                    data['steering_angle']="{:0.4f}".format((float(sol.value(delta)[1])+float(sol.value(delta)[0]))/2)
                    #data['steering_angle']="{:0.4f}".format(float(sol.value(delta)[0]))
                    data['throttle']="{:0.2f}".format(float(sol.value(d)[0]))

                    tau_sol = sol.value(tau)
                    n_sol = sol.value(n)
                    pts = track.convertParameterToPos(tau_sol,n_sol)
                    mpc_points_x = []
                    mpc_points_y = []
                    mpc_points_x.append(pts[i,0] for i in range(len(pts)))
                    mpc_points_y.append(pts[i,1] for i in range(len(pts)))
                    data['mpc_x'] = ','.join("{:0.4f}".format(pts[i,0]) for i in range(int(len(pts))))
                    data['mpc_y'] = ','.join("{:0.4f}".format(pts[i,1]) for i in range(int(len(pts))))

                    json_str = json.dumps(data)
                    msg = "42[\"steer\"," + json_str + "]";
                    #print('send'+msg)

                    send_steering = (float(sol.value(delta)[1])+float(sol.value(delta)[0]))/2
                    send_throttle = (sol.value(d)[0])

                    last_mpc_u = sol.value(U)
                    last_mpc_x = sol.value(X)

                except Exception as e:
                    print(e)
                    print('control fails')

                while (1000*time.time()-receive_time)<=dt*1000:
                    continue
                #await asyncio.sleep(dt)
                send_time = int(1000*time.time())
                sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
                #print(sql_query)
                #print([float(telemetry['sending_time']),receive_time,send_time,float(telemetry['sending_time']),x,y,phi0,v0,steering,throttle,send_steering,send_throttle])
                cur.execute(sql_query,
                    (float(telemetry['sending_time']),receive_time,send_time,float(telemetry['sending_time']),x,y,phi0,v0,steering,throttle,send_steering,send_throttle))
                con.commit()
                steering = send_steering
                throttle = send_throttle
                await ws.send(msg)            
            else:
                print("Falling back to manual mode")
                #cur.close()
                #with open('output/result.csv', 'w', newline='') as csvfile:
                #    writer = csv.writer(csvfile)
                #    for i in range(len(estimated_states)):
                #        writer.writerow(np.concatenate([real_states[i],estimated_states[i]]))
        
            



asyncio.get_event_loop().run_until_complete(websockets.serve(control_loop, 'localhost', 4567))
asyncio.get_event_loop().run_forever()

#async def main():
#    async with serve(control_loop, "localhost", 8765):
#        await asyncio.Future()  # run forever

#asyncio.run(main())