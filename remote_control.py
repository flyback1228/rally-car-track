import json
import asyncio
import websockets
import csv
import matplotlib.pyplot as plt
from track import SymbolicTrack
import yaml
from dynamics_models import BicycleKineticModelByParametricArc
import time
import casadi
import numpy as np
import sqlite3
from datetime import datetime
from scipy.spatial import KDTree


def _check_mode(msg):
    return "Auto" if msg and msg[:2] == '42' else "Manual"


class RemoteController:
    def __init__(self, global_table_name, save_to_database, pts_per_second=30):
        self.track_width = 7.0
        with open('params/racecar_nwh.yaml') as file:
            params = yaml.safe_load(file)
        with open('params/nwh_tire.yaml') as file:
            tire_params = yaml.safe_load(file)
        self.d_min = params['d_min']
        self.d_max = params['d_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.delta_min = params['delta_min']
        self.delta_max = params['delta_max']
        self.lf = params['lf']
        self.lr = params['lr']
        self.mass = params['m']
        self.Iz = params['Iz']

        self.save_to_database = save_to_database

        self.cd = 6000
        self.cm1 = 0.3

        self.cm2 = 30
        self.cm0 = 500
        self.cbf = 7000
        self.cbr = 7000

        self.N = 10
        self.T = 1.0
        self.dt = self.T / self.N
        self.nx = 6
        self.nu = 4

        self.pts_per_second = pts_per_second
        self.interval = self.T * self.pts_per_second / self.N

        self.B_lat = tire_params['B_lat']
        self.C_lat = tire_params['C_lat']
        self.D_lat = tire_params['D_lat']

        self.Fz = np.array([self.mass * 9.8 / 2, self.mass * 9.8 / 2])

        self.option = {'print_level': 0, 'max_cpu_time': self.dt*100}
        self.casadi_option = {}

        self.start_time = time.time() * 1000

        con = sqlite3.connect('output/sql_data.db')
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {global_table_name}")
        data = cur.fetchall()
        con.close()

        np_data = np.array(data)
        rows = len(np_data)
        self.global_data = (np_data[:, :-3]).astype(float)
        ptxs = np_data[:, -3]
        ptys = np_data[:, -2]
        self.global_pts = np.array(
            [[np.fromstring(ptxs[i], dtype=float, sep=',')[0], np.fromstring(ptys[i], dtype=float, sep=',')[0]] for i in
             range(rows)])
        self.search_tree = KDTree(self.global_pts)

        if save_to_database:
            self.con = sqlite3.connect('output/sql_data.db')
            self.cur = self.con.cursor()
            now = datetime.now()
            self.unity_table_name = now.strftime("_%m_%d_%Y_%H_%M_%S")
            self.cur.execute(
                "CREATE TABLE {} ("
                "unity_sent_time real, "
                "python_receved_time real, "
                "python_sent_time real, "
                "unity_received_type real,"
                "real_x real,"
                "real_y real,"
                "real_psi real,"
                "real_vx real,"
                "real_vy real"
                "received_steer real, "
                "received_throttle real, "
                "sent_steer real, "
                "sent_throttle real)".format(
                    self.unity_table_name))
        asyncio.get_event_loop().run_until_complete(websockets.serve(self.control_loop, 'localhost', 4567))
        asyncio.get_event_loop().run_forever()

    def parse_telemetry(self, msg):
        msg_json = msg[2:]
        parsed = json.loads(msg_json)

        msg_type = parsed[0]
        assert msg_type == 'telemetry' or msg_type == 'waypoints', "Invalid message type {}".format(msg_type)

        if msg_type == 'waypoints':
            values = parsed[1]
            x = values['ptsx']
            y = values['ptsy']
            with open('tracks/temp.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(x)):
                    writer.writerow([x[i], y[i]])

            #track = SymbolicTrack('tracks/temp.csv', self.track_width)
            # model = BicycleKineticModelByParametricArc(params, track)
            return False, parsed[1]
        # Telemetry values
        return True, parsed[1]

    async def control_loop(self, ws):
        async for message in ws:
            # print(message)
            if _check_mode(message) == 'Auto':
                msg_type, telemetry = self.parse_telemetry(message)
                if msg_type:
                    opti = casadi.Opti()

                    receive_time = int(1000 * time.time()) - self.start_time

                    phi0 = float(telemetry['psi'])
                    v0 = float(telemetry['speed'])
                    steering = float(telemetry['steering_angle'])
                    throttle = float(telemetry['throttle'])
                    pt_x = float(telemetry['x'])
                    pt_y = float(telemetry['y'])
                    vx0 = float(telemetry['vx'])

                    vy0 = float(telemetry['vy'])
                    omega0 = float(telemetry['omega'])
                    if vx0 <= 0:
                        vx0 = 0.01
                    if vy0==0:
                        vy0=0.01
                    if omega0==0:
                        omega0=0.001

                    x0 = casadi.DM([pt_x, pt_y, phi0, vx0, vy0, omega0])

                    dist, index = self.search_tree.query([pt_x, pt_y])
                    index = np.arange(index, self.N * self.interval + 1, self.interval).astype(int)
                    ref_pos = self.global_pts[index, :]
                    ref_phi = self.global_data[index, 1]
                    ref_vx = self.global_data[index, 2]
                    ref_vy = self.global_data[index, 3]
                    ref_steer = self.global_data[index, 4]
                    ref_omega = self.global_data[index, 5]
                    ref_throttle = self.global_data[index, 15]
                    ref_front_brake = self.global_data[index, 16]
                    ref_rear_brake = self.global_data[index, 17]

                    x = opti.variable(self.nx, self.N + 1)
                    u = opti.variable(self.nu, self.N)
                    x_dot = opti.variable(self.nx, self.N)

                    pt_sym_array = x[0:2, :]
                    phi_sym_array = x[2, :]
                    vx_sym_array = x[3, :]
                    vy_sym_array = x[4, :]
                    omega_sym_array = x[5, :]

                    throttle_array = u[0, :]
                    front_brake_array = u[1, :]
                    rear_brake_array = u[2, :]
                    delta_sym_array = u[3, :]

                    x_guess = np.vstack((ref_pos[:,0],ref_pos[:,1],ref_phi,ref_vx,ref_vy,ref_omega))
                    x_guess[:, 0] = np.array([pt_x, pt_y, phi0, vx0, vy0, omega0])

                    print(x_guess)

                    fx_f_sym_array = self.cd * throttle_array \
                                     - self.cm0 - self.cm1 * vx_sym_array[0:-1] * vx_sym_array[0:-1] \
                                     - self.cm2 * vy_sym_array[0:-1] * vy_sym_array[0:-1] \
                                     - self.cbf * front_brake_array
                    fx_r_sym_array = self.cd * throttle_array \
                                     - self.cm0 - self.cm1 * vx_sym_array[0:-1] * vx_sym_array[0:-1] \
                                     - self.cm2 * vy_sym_array[0:-1] * vy_sym_array[0:-1] \
                                     - self.cbr * rear_brake_array

                    alpha_f = -casadi.atan2(omega_sym_array[0:-1] * self.lf + vy_sym_array[0:-1],
                                            vx_sym_array[0:-1]) + delta_sym_array
                    alpha_r = casadi.atan2(omega_sym_array[0:-1] * self.lr - vy_sym_array[0:-1], vx_sym_array[0:-1])

                    fy_f_sym_array = self.Fz[0] * self.D_lat * casadi.sin(self.C_lat * casadi.atan(self.B_lat * alpha_f))
                    fy_r_sym_array = self.Fz[1] * self.D_lat * casadi.sin(self.C_lat * casadi.atan(self.B_lat * alpha_r))

                    x_dot[0, :] = vx_sym_array[0:-1] * casadi.cos(phi_sym_array[0:-1]) \
                                  - vy_sym_array[0:-1] * casadi.sin(phi_sym_array[0:-1])
                    x_dot[1, :] = vx_sym_array[0:-1] * casadi.sin(phi_sym_array[0:-1]) \
                                  + vy_sym_array[0:-1] * casadi.cos(phi_sym_array[0:-1])
                    x_dot[2, :] = omega_sym_array[0:-1]  # phi_dot
                    x_dot[3, :] = 1 / self.mass \
                                  * (fx_r_sym_array + fx_f_sym_array * casadi.cos(
                        delta_sym_array) - fy_f_sym_array * casadi.sin(
                        delta_sym_array) + self.mass * vy_sym_array[0:-1] * omega_sym_array[0:-1])
                    x_dot[4, :] = 1 / self.mass \
                                  * (fy_r_sym_array + fx_f_sym_array * casadi.sin(
                        delta_sym_array) + fy_f_sym_array * casadi.cos(
                        delta_sym_array) - self.mass * vx_sym_array[0:-1] * omega_sym_array[0:-1])  # vydot
                    x_dot[5, :] = 1 / self.Iz \
                                  * (fy_f_sym_array * self.lf * casadi.cos(
                        delta_sym_array) + fx_f_sym_array * self.lf * casadi.sin(
                        delta_sym_array) - fy_r_sym_array * self.lr)  # omegadot

                    # objective
                    opti.minimize(10 * casadi.norm_fro((pt_sym_array - ref_pos.T))
                                  #+ 0.01 * casadi.norm_fro(phi_sym_array.T - ref_phi)
                                  #+ casadi.norm_fro(vx_sym_array.T - ref_vx)
                                  #+ 2 * casadi.norm_fro(throttle_array.T - ref_throttle[0:-1])
                                  #+ casadi.norm_fro(delta_sym_array.T - ref_steer[0:-1])
                                  #+ casadi.norm_fro(front_brake_array.T - ref_front_brake[0:-1])
                                  #+ casadi.norm_fro(rear_brake_array.T - ref_rear_brake[0:-1])
                                )

                    opti.subject_to(x[:, 0] == x0)
                    # dynamics
                    opti.subject_to(x[:, 1:] == x[:, 0:-1] + self.dt * x_dot)

                    # input bound
                    opti.subject_to(opti.bounded(self.delta_min, delta_sym_array, self.delta_max))
                    opti.subject_to(opti.bounded(0, throttle_array, 1))
                    opti.subject_to(opti.bounded(0, front_brake_array, 1))
                    opti.subject_to(opti.bounded(0, rear_brake_array, 1))

                    opti.solver("ipopt", self.casadi_option, self.option)  # set numerical backend
                    data = {}
                    opti.set_initial(x, x_guess)
                    try:
                        sol = opti.solve()  # actual solve

                        sol_pt = sol.value(pt_sym_array)
                        data['steering_angle'] = "{:0.4f}".format(float(sol.value(delta_sym_array)[0]))
                        data['throttle'] = "{:0.2f}".format(float(sol.value(throttle_array)[0]))
                        data['brake'] = "{:0.2f},{:0.2f}".format(float(sol.value(front_brake_array)[0]),
                                                                 float(sol.value(rear_brake_array)[0]))
                        # mpc_points_x.append(float(sol_pt[0, i]) for i in range(self.N))
                        # mpc_points_y.append(float(sol_pt[1, i]) for i in range(self.N))
                        data['mpc_x'] = ','.join("{:0.4f}".format(float(sol_pt[0, i])) for i in range(self.N))
                        data['mpc_y'] = ','.join("{:0.4f}".format(float(sol_pt[1, i])) for i in range(self.N))

                        data['ref_x'] = ','.join("{:0.4f}".format(ref_pos[i, 0]) for i in range(self.N))
                        data['ref_y'] = ','.join("{:0.4f}".format(ref_pos[i, 1]) for i in range(self.N))

                    except Exception as e:
                        print(e)

                    json_str = json.dumps(data)
                    msg = "42[\"steer\"," + json_str + "]"

                    applying_time = float(telemetry['applying_time'])

                    while ((time.time()) * 1000 - applying_time) < self.dt * 1000:
                        continue

                    if self.save_to_database:
                        send_time = int(1000 * time.time()) - self.start_time
                        sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?)".format(self.unity_table_name)
                        self.cur.execute(sql_query,
                                         (float(telemetry['sending_time']) - self.sstart_time,
                                          receive_time,
                                          send_time,
                                          float(telemetry['applying_time']) - self.sstart_time,
                                          pt_x,
                                          pt_y,
                                          phi0,
                                          vx0,
                                          vy0,
                                          steering,
                                          throttle,
                                          float(sol.value(delta_sym_array)[0]),
                                          float(sol.value(throttle_array)[0])))
                        self.con.commit()
                    await ws.send(msg)


if __name__ == '__main__':
    remote_contrller = RemoteController('unity_06_02_2022_16_14_26', False)
