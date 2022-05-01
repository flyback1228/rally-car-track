import json
import asyncio
import websockets
import csv
from track import SymbolicTrack
import yaml
from dynamics_models import BicycleKineticModelByParametricArc
import time
import casadi
import numpy as np
import sqlite3
from datetime import datetime

track_width = 7.0
#track = SymbolicTrack('tracks//temp_nwh.csv',track_width)
track = None
with open('params//racecar_nwh.yaml') as file:
    params = yaml.safe_load(file)

d_min = params['d_min']
d_max = params['d_max']
v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min']
delta_max = params['delta_max']

model = BicycleKineticModelByParametricArc(params,track)

real_states=[]
real_inputs=[]
estimated_states=[]
#predicated_states=[]

steering = 0
throttle = 0

N = 6
dt = 0.25
nx = model.nx
nu = model.nu

coeff = casadi.DM(np.logspace(0.1,0.8,N+1)).T
#print(coeff)
first_it = True
estimate_dt = dt/2+0.05

def getRef(x0):
    t = casadi.MX.sym('t')
    #f = casadi.Function('f',[t],[track.getCurvatureSym(t)])
    X_ref = casadi.DM(nx,N+1)
    #U_ref = casadi.DM(nu,N)  
    v0 = float(x0[3])
    tau0=float(x0[0])
    n0=float(x0[1])
    s0 = track.getSFromT(tau0%track.max_t)   

    X_ref[:,0]=x0
    vt = v0
    
    for i in range(N):
        for j in range(20):
            #vt = min(v_max,v0 + 5.9*d_max*dt/10)
            st = (s0 + v0*estimate_dt/20)%track.max_s
            tau_t = track.getTFromS(st)
            kappt_t = track.getAvgKappaFromS(st)
            v0=min((2000.0*kappt_t*kappt_t-80*kappt_t+1)*v_max,vt + 8*estimate_dt/20)
            vt = v0
            s0 = st

        #vt = min(v_max,v0 + 5.9*d_max*dt)
        #st = (s0 + (v0+vt)*dt/2)%track.max_s
        #tau_t = track.getTFromS(st)
        phi_t = track.getPhiFromT(tau_t)
        while(tau_t<tau0):
            tau_t = tau_t+ track.max_t        
        
        X_ref[:,i+1] = casadi.DM([tau_t,n0/(N+1)*(N-i),phi_t,v0])
        tau0 = tau_t
    return X_ref



dae_x = casadi.MX.sym('dae_x',model.nx)
dae_u = casadi.MX.sym('dae_u',model.nu)
dae_ode = model.update(dae_x,dae_u)
dae = {'x':dae_x, 'p':dae_u, 'ode':dae_ode}
#integrator = casadi.integrator('F','cvodes',dae,{'tf':dt})
#estimate_integrator = casadi.integrator('F','cvodes',dae,{'tf':estimate_dt})

start_time = int(1000*time.time())

x_guess = casadi.DM.zeros(nx,N+1)
u_guess = casadi.DM.zeros(nu,N)

#last_mpc_x = casadi.DM.zeros(nx,N+1)
last_mpc_u = casadi.DM.zeros(nu,N)

#solve options
send_time = 0
option = {} 
option['print_level']=0
casadi_option = {} 
#casadi_option['print_time']=False
option['max_cpu_time']=500

con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = now.strftime("_%m_%d_%Y_%H_%M_%S")

cur.execute('''CREATE TABLE {} (unity_sent_time real, 
            python_receved_time real, 
            python_sent_time real, 
            unity_received_type real,
            real_x real,
            real_y real,
            real_psi real,
            real_v real,
            received_steer real, 
            received_throttle real, 
            sent_steer real, 
            sent_throttle real,
            sent_front_brake real,
            sent_rear_brake real,
            wheel1_omega real,
            wheel1_motortorque real,
            wheel1_braketorque real,
            wheel1_lambda real,
            wheel1_alpha real,
            wheel1_fx real,
            wheel1_fy real,
            wheel1_fz real,
            wheel1_steer real,
            wheel2_omega real,
            wheel2_motortorque real,
            wheel2_braketorque real,
            wheel2_lambda real,
            wheel2_alpha real,
            wheel2_fx real,
            wheel2_fy real,
            wheel2_fz real,
            wheel2_steer real,
            wheel3_omega real,
            wheel3_motortorque real,
            wheel3_braketorque real,
            wheel3_lambda real,
            wheel3_alpha real,
            wheel3_fx real,
            wheel3_fy real,
            wheel3_fz real,
            wheel4_omega real,
            wheel4_motortorque real,
            wheel4_braketorque real,
            wheel4_lambda real,
            wheel4_alpha real,
            wheel4_fx real,
            wheel4_fy real,
            wheel4_fz real)'''.format(table_name))



def _check_mode(msg):
    return  "Auto" if msg and msg[:2] == '42' else "Manual"

def _parse_telemetry(msg,ws):
    msg_json = msg[2:]
    parsed = json.loads(msg_json)
    
    msg_type = parsed[0]
    assert msg_type == 'telemetry' or msg_type == 'waypoints', "Invalid message type {}".format(msg_type) 
    
    if(msg_type == 'waypoints'):
        global track,model
        values = parsed[1] 
        x = values['ptsx']
        y = values['ptsy']
        with open('tracks/temp_nwh.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(x)):
                writer.writerow([x[i],y[i]])

        track = SymbolicTrack('tracks/temp_nwh.csv',track_width)        
        model = BicycleKineticModelByParametricArc(params,track)
        
        
        return False,parsed[1]

    # Telemetry values
    return True,parsed[1]

    
async def control_loop(ws):
    global steering,throttle,last_mpc_u,last_mpc_x,x_guess,u_guess,option,estimate_dt,start_time,first_it,send_time
    async for message in ws:
        
        #dae = {'x':X, 'p':U, 'ode':model.update(X,U)}
        
      
        if _check_mode(message) == 'Auto':                
            msg_type,telemetry  = _parse_telemetry(message,ws)
            
            if(msg_type):
                #print(message)
                opti = casadi.Opti()
                X = opti.variable(nx,N+1)
                U = opti.variable(nu,N) 

                receive_time = int(1000*time.time())-start_time
                #print('received at ' + str(receive_time))  

                phi0 = float(telemetry['psi'])
                #ptsx = telemetry['ptsx']
                #ptsy = telemetry['ptsy']
                v0 = float(telemetry['speed'])
                received_steering = float(telemetry['steering_angle'])
                #received_throttle = float(telemetry['throttle'])
                received_throttle = 0.0;
                x = float(telemetry['x'])
                y = float(telemetry['y'])
                wheel_info = []
                #print(telemetry['wheel_0']);
                wheel_info.append(([float(x) for x in telemetry['wheel_0'].split(',')]))
                wheel_info.append(([float(x) for x in telemetry['wheel_1'].split(',')]))
                wheel_info.append(([float(x) for x in telemetry['wheel_2'].split(',')]))
                wheel_info.append(([float(x) for x in telemetry['wheel_3'].split(',')]))
                                
                #initial boundary
                real_tau,real_n = track.convertXYtoTN((x,y))
                real_state = casadi.DM([real_tau,real_n,phi0,v0])
                real_u = casadi.DM([steering,throttle])
                #real_states.append(real_state)
                #real_inputs.append(np.array([steering,throttle]))
                #print('real_state: '+str(real_state))current_time = time.time()
                

                #kappa = track.f_kappa(tau0)
                #phi_c = track.getPhiFromT(tau0)
                #print('phi_c: '+str(phi_c))
                #tangent_vec = track.getTangentVec(tau0)

                
                estimate_state = real_state + model.update(real_state,real_u)*estimate_dt
                #Fk = estimate_integrator(x0=real_state, p=real_u)
                #estimate_state = Fk['xf']
                
                
                #x0_guess = casadi.DM(estimated_state)
                x0_guess = estimate_state
                x_guess[:,0] = x0_guess

                for k in range(N-1):
                    #Fk = estimate_integrator(x0=x0_guess, p=last_mpc_u[:,k+1])
                    #xt_guess = Fk['xf']
                    xt_guess = x0_guess + model.update(x0_guess,last_mpc_u[:,k+1])*estimate_dt
                    x_guess[:,k+1] = xt_guess
                    x0_guess = xt_guess
                #print(x_guess)
                #Fk = estimate_integrator(x0=x_guess[:,-2], p=last_mpc_u[:,-1])
                #x_guess[:,-1] = Fk['xf']
                x_guess[:,-1] = x_guess[:,-2] + model.update(x_guess[:,-2],last_mpc_u[:,-1])*estimate_dt
                u_guess[:,0:-1] = last_mpc_u[:,1:]
                u_guess[:,-1]=u_guess[:,-2]

                X_ref = getRef(estimate_state)
                #if(first_it):
                #    X_ref = getRef(x0_guess)
                #    first_it = False
                #else:
                #    X_ref = x_guess

                


                

                #X = [tau,n,phi,v]
                tau = X[0,:]
                n = X[1,:]
                phi = X[2,:]
                v = X[3,:]

                # control input
                delta = U[0,:]
                d = U[1,:]


                ds = N*dt*v_max
                tau0 = float(estimate_state[0])
                #target boundary    delta_min

                tau_error = tau - X_ref[0,:]
                n_error=n-X_ref[1,:]
                v_error = v - X_ref[3,:]
                phi_error = casadi.cos(phi)*casadi.cos(X_ref[2,:]) + casadi.sin(phi)*casadi.sin(X_ref[2,:])
                #opti.minimize(0.1*casadi.dot(tau_error,tau_error) +10*casadi.dot(delta[1:]-delta[0:-1],delta[1:]-delta[0:-1] ))
                #opti.minimize(0.1*casadi.dot(tau[-1]-tau_t,tau[-1]-tau_t) + 0.4*casadi.dot(tau_error,tau_error) + 0.0001*casadi.dot(n_error,n_error) - 40.0*casadi.dot(phi_error,phi_error)+10*casadi.dot(delta[1:]-delta[0:-1],delta[1:]-delta[0:-1] ))
                opti.minimize(0.0001*casadi.dot(tau_error,tau_error) + 0.0000001*casadi.sum2(n_error*n_error*coeff)+ 0.01*casadi.dot(v_error,v_error) +0.00005*casadi.dot(delta[1:]-delta[0:-1],delta[1:]-delta[0:-1] ))


                for k in range(N):
                    #k1 = model.update(X[:,k],U[:,k])
                    #k2 = model.update(X[:,k]+estimate_dt/2*k1,U[:,k])
                    #k3 = model.update(X[:,k]+estimate_dt/2*k2,U[:,k])
                    #k4 = model.update(X[:,k]+estimate_dt*k3,U[:,k])
                    #x_next = X[:,k]+estimate_dt/6*(k1 +2*k2 +2*k3 +k4)
                    x_next = X[:,k] + estimate_dt*model.update(X[:,k],U[:,k])
                    #Fk = estimate_integrator(x0=X[:,k], p=U[:,k])
                    #x_next = Fk['xf']
                    opti.subject_to(X[:,k+1]==x_next)  
                    #opti.subject_to(X[3,k]*casadi.fabs(U[0,k])<=15.0)
                    
                #initial condition
                opti.subject_to(X[:,0] == estimate_state)

                #state bound
                n_v = casadi.fmax(v_max,casadi.fabs(v0))
                n_c = casadi.fmax(track_width/2,casadi.fabs(real_n))
                #print('nv = '+str(n_v))
                #opti.subject_to(opti.bounded(v_min,v[0:-1],n_v+0.1))
                opti.subject_to(opti.bounded(v_min,v[-1],v_max))
                
                #opti.subject_to(v<=5.0/(track.f_kappa(tau)+0.001))
                #opti.subject_to(v*v*track.f_kappa(tau)*track.f_kappa(tau)<=0.25)

                opti.subject_to(opti.bounded(-track_width/2,n[-1],track_width/2))
                opti.subject_to(opti.bounded(-n_c-0.1,n[0:-1],n_c+0.1))
                #opti.subject_to(v[1:]*delta*delta<=0.1)

                #print('nc='+str(n_c))
                
                #input bound
                opti.subject_to(opti.bounded(delta_min,delta,delta_max))
                opti.subject_to(opti.bounded(-1.8,d,d_max))  
                   
                #opti.subject_to(b*d==0)        
                
                opti.solver("ipopt",casadi_option,option) # set numerical backend
                data={}
                mpc_points_x = []
                mpc_points_y = []
                try:      

                    opti.set_initial(X,x_guess)
                    opti.set_initial(U,u_guess)
                    sol = opti.solve()   # actual solve
                    #predicated_states.append(np.array(sol.value(X)[:,1]))
                    tau_sol = sol.value(tau)
                    n_sol = sol.value(n)
                    pts = track.convertParameterToPos(tau_sol,n_sol,N+1)
                    #steering = float((sol.value(delta)[0] +sol.value(delta)[1])/2)
                    steering = float(sol.value(delta)[0])
                    throttle = float(sol.value(d)[0])
                    last_mpc_u = sol.value(U)
                    
                    #print(steering)
                    data['steering_angle']="{:0.4f}".format(steering)
                    
                    mpc_points_x.append(pts[i,0] for i in range(len(pts)))
                    mpc_points_y.append(pts[i,1] for i in range(len(pts)))
                    data['mpc_x'] = ','.join("{:0.4f}".format(pts[i,0]) for i in range(int(len(pts))))
                    data['mpc_y'] = ','.join("{:0.4f}".format(pts[i,1]) for i in range(int(len(pts)))) 

                    ref_pts = track.convertParameterToPos(X_ref[0,:],X_ref[1,:],N+1)

                    data['ref_x'] = ','.join("{:0.4f}".format(ref_pts[i,0]) for i in range(int(len(ref_pts))))
                    data['ref_y'] = ','.join("{:0.4f}".format(ref_pts[i,1]) for i in range(int(len(ref_pts)))) 
 

                    if(throttle>0):
                        data['throttle']="{:0.2f}".format(throttle)
                        data['brake'] = "0.0,0.0";   
                    else:
                        data['throttle']="0.0"
                        data['brake'] = f"{-throttle/1.8},{-throttle/1.8}"
                    
                except Exception as e:
                    print(e)
                    #predicated_states.append(np.array(opti.debug.value(X)[:,1]))
                   

                    tau_sol = opti.debug.value(tau)
                    n_sol = opti.debug.value(n)
                    pts = track.convertParameterToPos(tau_sol,n_sol,N+1)    
                    #steering = float(opti.debug.value(delta)[0])
                    #throttle = (opti.debug.value(d)[0])
                    last_mpc_u = u_guess
                    steering=float((u_guess[0,0]+u_guess[0,1])/2)       
                    throttle = float(u_guess[1,0])         
                    data['steering_angle']="{:0.4f}".format(float(u_guess[0,0]))
                    throttle = float(u_guess[1,0])
                    #temp_t=np.array(N)
                    #temp_n = np.array(N)
                    #temp_t[i] = float(x_guess[i,0]) for i in range(N)
                    #temp_n.append(float(x_guess[i,1]) for i in range(N))
                    mpc_points_x.append(pts[i,0] for i in range(len(pts)))
                    mpc_points_y.append(pts[i,1] for i in range(len(pts)))
 
                    data['mpc_x'] = ','.join("{:0.4f}".format(pts[i,0]) for i in range(int(len(pts))))
                    data['mpc_y'] = ','.join("{:0.4f}".format(pts[i,1]) for i in range(int(len(pts))))   

                    ref_pts = track.convertParameterToPos(X_ref[0,:],X_ref[1,:],N+1)

                    data['ref_x'] = ','.join("{:0.4f}".format(pts[i,0]) for i in range(int(len(ref_pts))))
                    data['ref_y'] = ','.join("{:0.4f}".format(pts[i,1]) for i in range(int(len(ref_pts))))  
                    
                    
                    if(throttle>0):
                        data['throttle']="{:0.2f}".format(throttle)
                        data['brake'] = "0.0,0.0";   
                    else:
                        data['throttle']="0.0"
                        data['brake'] = f"{-throttle},{-throttle}"
                        
                #float(sol.value(delta)[1])
                
                
                json_str = json.dumps(data)
                msg = "42[\"steer\"," + json_str + "]"

                #while ((time.time())*1000-(receive_time+start_time))<dt*1000:
                #    continue
                applying_time = float(telemetry['applying_time'])
                #print(time.tim)
                #print(applying_time)
                if throttle<0:
                    sql_throttle = 0
                    front_brake = -throttle
                    rear_brake = - throttle
                else:
                    sql_throttle = throttle
                    front_brake = 0
                    rear_brake = 0
                sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
                cur.execute(sql_query,(float(telemetry['sending_time'])-start_time,receive_time,send_time,applying_time-start_time,x,y,phi0,v0,received_steering,received_throttle,steering,sql_throttle,front_brake,rear_brake,
                     wheel_info[0][0],wheel_info[0][1],wheel_info[0][2],wheel_info[0][3],wheel_info[0][4],wheel_info[0][5],wheel_info[0][6],wheel_info[0][7],wheel_info[0][8],
                     wheel_info[1][0],wheel_info[1][1],wheel_info[1][2],wheel_info[1][3],wheel_info[1][4],wheel_info[1][5],wheel_info[1][6],wheel_info[1][7],wheel_info[1][8],
                     wheel_info[2][0],wheel_info[2][1],wheel_info[2][2],wheel_info[2][3],wheel_info[2][4],wheel_info[2][5],wheel_info[2][6],wheel_info[2][7],
                     wheel_info[3][0],wheel_info[3][1],wheel_info[3][2],wheel_info[3][3],wheel_info[3][4],wheel_info[3][5],wheel_info[3][6],wheel_info[3][7]))
                con.commit()
                while ((time.time())*1000-applying_time)<dt*1000/2:
                    continue
                send_time = int(1000*time.time())-start_time
                
                await ws.send(msg)  
            else:
                data ={}
                data['steering_angle']="0.00"
                data['mpc_x'] = ','.join("{:0.2f}".format(0) for i in range(int(N+1)))
                data['mpc_y'] = ','.join("{:0.2f}".format(0) for i in range(int(N+1))) 
            

                data['ref_x'] = ','.join("{:0.2f}".format(0) for i in range(int(N+1)))
                data['ref_y'] = ','.join("{:0.2f}".format(0) for i in range(int(N+1))) 

                data['throttle']="0.0"
                data['brake'] = "0.0,0.0" 
                json_str = json.dumps(data)
                msg = "42[\"steer\"," + json_str + "]"
                await ws.send(msg)     
                         


asyncio.get_event_loop().run_until_complete(websockets.serve(control_loop, 'localhost', 4567))
asyncio.get_event_loop().run_forever()