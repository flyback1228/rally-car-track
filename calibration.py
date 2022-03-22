from cmath import pi
import imp
import json
import asyncio
import websockets
import csv
import matplotlib.pyplot as plt
from track import *
import yaml
from dynamics_models import *
import time

track_width = 8.0
track = SymbolicTrack('tracks/temp.csv',track_width)
with open('params/racecar.yaml') as file:
        params = yaml.load(file)

d_min = params['d_min']
d_max = params['d_max']
v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min']  # minimum steering angle [rad]
delta_max = params['delta_max']
model = BicycleKineticModelByParametricArc(params,track)

real_states=[]
estimated_states=[]
predicated_states=[]

state_history=[]
estimated_state_history=[]
time_history =[]
last_time = time.time()

N = 8
T = 1.6
dt = T/N
index = 0
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

        track = SymbolicTrack('tracks/temp.csv',8.0)        
        model = BicycleKineticModelByParametricArc(params,track)
        return False,values
        

    # Telemetry values
    values = parsed[1]    
    return True,values
    
    
async def control_loop(ws):
    global index,last_time
    async for message in ws:
        current_time = time.time()
        time_history.append(current_time-last_time)
        
        print('recevint time: '+str(current_time))

        print('receive:'+message)
        if _check_mode(message) == 'Auto':
            
            is_telemtry,telemetry  = _parse_telemetry(message)
            if(is_telemtry):
                index = index + 1
                phi0 = float(telemetry['psi'])
                #ptsx = telemetry['ptsx']
                #ptsy = telemetry['ptsy']
                v0 = float(telemetry['speed'])
                steering = float(telemetry['steering_angle'])
                throttle = float(telemetry['throttle'])
                x = float(telemetry['x'])
                y = float(telemetry['y'])
                #print('speed: '+str(v0))

                state_history.append([current_time-last_time,x,y,phi0,v0])
                
                #Implement your model predictive control here.
                track_length_tau = track.max_t

                #initial boundary
                tau0,n0 = track.convertXYtoTN((x,y))
                real_state = np.array([tau0,n0,phi0,v0])
                real_states.append(real_state)
                print('real state: ' + str(real_state))
                if estimated_states:
                    print('estimated state: '+str(estimated_states[-1]))

                kappa = track.f_kappa(tau0)
                phi_c = track.getPhiFromT(tau0)
                tangent_vec = track.getTangentVec(tau0)

                print('phi_c = '+str(phi_c))
                print('steering = '+str(steering))
                print('v0 = '+str(v0))

                dtau = float(tau0 + v0*casadi.cos(phi0-phi_c+steering)/(casadi.norm_2(tangent_vec)*(1-n0*kappa)) *dt)
                print('dtau = '+str(dtau-tau0))
                print('(1-n0*kappa) = '+str((1-n0*kappa)))
                dn = float(n0 + v0*casadi.sin(phi0-phi_c+steering) *dt)
                dphi = float(phi0 + v0/(params['lf']+params['lr']) * casadi.tan(steering)*dt)
                
                dv = float(v0 + (5.5*throttle-0.1*v0)*dt)
                
                estimated_state = np.array([dtau,dn,dphi,dv])
                estimated_states.append(estimated_state)     

                #print(estimated_state)

                new_states = [casadi.DM(real_state)]
                tau_array = [tau0]
                n_array = [n0]                
                for i in range(4):
                    ds = model.update(new_states[-1],[steering,throttle])  
                    new_state = new_states[-1]+ds*(dt+0.02)
                    new_states.append(new_state)  
                    tau_array.append(float(new_state[0] ))   
                    n_array.append(float(new_state[1]))
                print(new_states)

                xy = track.convertParameterToPos([new_states[1][0]],[new_states[1][1]])
                estimated_state_history.append([xy[0,0],xy[0,1],float(new_states[1][2]),float(new_states[1][3])])

                data={}
                data['steering_angle']=str(2.0/180*pi)
                data['throttle']=str(1)

                pts = track.convertParameterToPos(tau_array,n_array)
                #mpc_points_x = []
                #mpc_points_y = []
                #mpc_points_x.append(pts[i+1,0] for i in range(len(pts)-1))
                #mpc_points_y.append(pts[i+1,1] for i in range(len(pts)-1))
                data['mpc_x'] = ','.join(str(pts[i+1,0]) for i in range(len(pts)-1))
                data['mpc_y'] = ','.join(str(pts[i+1,1]) for i in range(len(pts)-1))


                json_str = json.dumps(data)
                #print(json_str)
                msg = "42[\"steer\"," + json_str + "]";
                print('send'+msg)

                while (time.time()-current_time)<dt:
                    continue
                print('sending time: '+str(time.time()))
                
                if(index == 20 ):
                    with open('output/state.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)    
                        for i in range(len(state_history)):          
                            writer.writerow(state_history[i]+estimated_state_history[i])
                        
                last_time=current_time
                #await asyncio.sleep(dt)
                await ws.send(msg)            
        else:
            print("Falling back to manual mode")


asyncio.get_event_loop().run_until_complete(websockets.serve(control_loop, 'localhost', 4567))
asyncio.get_event_loop().run_forever()

#async def main():
#    async with serve(control_loop, "localhost", 8765):
#        await asyncio.Future()  # run forever

#asyncio.run(main())