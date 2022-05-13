import casadi
import yaml
from track import *
from tire_model import SimplePacTireMode
import sqlite3
from datetime import datetime
import math
import copy

track_width = 6
track = SymbolicTrack('tracks/temp_nwh.csv', track_width)

# define tire mode
with open('params/nwh_tire.yaml') as file:
    front_tire_params = yaml.safe_load(file)
front_tire_model = SimplePacTireMode(front_tire_params)

with open('params/nwh_tire.yaml') as file:
    rear_tire_params = yaml.safe_load(file)
rear_tire_model = SimplePacTireMode(rear_tire_params)

# define model
with open('params/racecar_nwh.yaml') as file:
    params = yaml.safe_load(file)

lf = params['lf']
lr = params['lr']
mass = params['m']
Iz = params['Iz']

# parameters
v_min = params['v_min']
v_max = params['v_max']
delta_min = params['delta_min']  # minimum steering angle [rad]
delta_max = params['delta_max']
delta_dot_min = params['delta_dot_min']
delta_dot_max = params['delta_dot_max']

# initial state
tau0 = 25
s0 = track.getSFromT(tau0)
init_s0 = float(s0)
v0 = 0.01
phi0 = track.getPhiFromT(tau0)
pt = track.pt_t(tau0)
y0 = casadi.vertcat(pt[0], pt[1], phi0, v0, 0, 0, 0)

# mpc params
N = 90
nx = 7
nu = 4
T = 3.0
dt = T / N

# sql settings
con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
now = datetime.now()
table_name = 'unity' + now.strftime("_%m_%d_%Y_%H_%M_%S")
cur.execute(
    "CREATE TABLE {} (computation_time real,phi real, vx real, vy real, steer real,omega real,front_wheel_alpha real,"
    "rear_wheel_alpha real,front_fx real, rear_fx real,front_fy real, rear_fy real,s real,ds real,poly_order real,"
    "ptx text,pty text,coeff text)".format(
        table_name))

# casadi options
option = {'max_iter': 3000, 'tol': 1e-6, 'print_level': 0}
# option['linear_solver'] = 'ma27'

casadi_option = {}
# casadi_option['print_time']=False

# Fz, need work
Fz = casadi.DM.ones(2) * params['m'] / 2 * 9.81

Fx_f_max = front_tire_model.getMaxLongitudinalForce(Fz[0])
Fx_r_max = rear_tire_model.getMaxLongitudinalForce(Fz[1])

Fy_f_max = front_tire_model.getMaxLateralForce(Fz[0])
Fy_r_max = rear_tire_model.getMaxLateralForce(Fz[1])

sol_x = None
sol_u = None

# throttle parmas, F_f = cd*d -cm1*v-cm0
cd = 5000
cm1 = 0.3

cm2 = 50
cm0 = 0
cb = 10000

guess_X = casadi.DM.zeros(nx, N + 1)
guess_U = casadi.DM.zeros(nu, N)


# forward model using (x,y)
def forwardModel(X, U):
    phi = X[2]
    vx = X[3]
    vy = X[4, :]
    omega = X[5]
    delta = X[6]

    # control input
    delta_dot = U[0]
    # front_throttle = U[1]
    # rear_throttle=U[2]
    throttle = U[1]
    front_brake = U[2]
    rear_brake = U[3]

    fx_f = cd * throttle - cm0 - cm1 * vx * vx - cm2 * vy * vy - cb * front_brake
    fx_r = cd * throttle - cm0 - cm1 * vx * vx - cm2 * vy * vy - cb * rear_brake

    alpha_f = -casadi.atan2(omega * lf + vy, vx) + delta
    alpha_r = casadi.atan2(omega * lr - vy, vx)

    fy_f = front_tire_model.getLateralForce(alpha_f, Fz[0])
    fy_r = rear_tire_model.getLateralForce(alpha_r, Fz[1])

    # front_rate = casadi.fmax((fy_f_sym_array*fy_f_sym_array+fx_f_sym_array*fx_f_sym_array)/(Fy_f_max*Fy_f_max),1)
    # rear_rate = casadi.fmax((fy_r_sym_array*fy_r_sym_array+fx_r_sym_array*fx_r_sym_array)/(Fy_r_max*Fy_r_max),1)

    # fy_f_sym_array=fy_f_sym_array/(front_rate**0.5)
    # fy_r_sym_array=fy_r_sym_array/(rear_rate**0.5)
    # fx_f_sym_array=fx_f_sym_array/(front_rate**0.5)
    # fx_r_sym_array=fx_r_sym_array/(rear_rate**0.5)
    return casadi.veccat(
        vx * casadi.cos(phi) - vy * casadi.sin(phi),  # x_dot
        vx * casadi.sin(phi) + vy * casadi.cos(phi),  # n_dot
        omega,  # phi_dot
        1 / mass * (fx_r + fx_f * casadi.cos(delta) - fy_f * casadi.sin(delta) + mass * vy * omega),  # vxdot
        1 / mass * (fy_r + fx_f * casadi.sin(delta) + fy_f * casadi.cos(delta) - mass * vx * omega),  # vydot
        1 / Iz * (fy_f * lf * casadi.cos(delta) + fx_f * lf * casadi.sin(delta) - fy_r * lr),  # omegadot
        delta_dot
    )


# convert global (x,y) to local (s,n)
def convertXYtoSN(track, coeff, pos, s0_guess=0):
    x1 = float(pos[0])
    y1 = float(pos[1])
    ref_t, _ = track.convertXYtoTN([x1, y1])
    assert (ref_t != -1)

    x = np.poly1d(coeff[:, 0])
    y = np.poly1d(coeff[:, 1])

    dx = np.polyder(x)
    dy = np.polyder(y)
    f = -dx * (x - x1) - dy * (y - y1)
    raw_roots = np.roots(f)
    raw_roots = np.real(raw_roots[np.imag(raw_roots) == 0])

    abs_root = np.abs(raw_roots - s0_guess)
    idx = np.argmin(abs_root)
    root = raw_roots[idx]
    assert (np.imag(root) == 0)
    root = np.real(root)

    x_org = x(root)
    y_org = y(root)
    dx_val = dx(root)
    dy_val = dy(root)

    new_x = x1 - x_org
    new_y = y1 - y_org

    theta = math.atan2(dy_val, dx_val)
    M = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    new_pos = np.matmul(M, np.array([new_x, new_y]))
    return [root, new_pos[1]]


# convert local (s,n) to global(x,y)
def convertSNtoXY(coeff, pos):
    t = float(pos[0])
    n = float(pos[1])

    x = np.poly1d(coeff[:, 0])
    y = np.poly1d(coeff[:, 1])

    x0 = x(t)
    y0 = y(t)

    dx = np.polyder(x)
    dy = np.polyder(y)

    theta = math.atan2(dy(t), dx(t))
    return [x0 - n * math.sin(theta), y0 + n * math.cos(theta)]


def optimize(y):
    # define ocp
    global guess_X, guess_U, sol_x, sol_u
    ds = 100

    tau0, _ = track.convertXYtoTN([float(y[0]), float(y[1])])
    s0 = float(track.getSFromT(tau0))

    # find the polymonial fitting range, from -2 to s0+ds, ds is estimated from last step
    if sol_x is not None:
        ds = max(float(sol_x[0, -1]) + cd / 50 * dt, 80)
    s0_array = np.arange(s0 - 2, s0 + ds, min(ds / 100, 0.2))

    # tau_array = casadi.reshape(track.getTFromS(s0_array),1,len(s0_array))
    pos = np.zeros([len(s0_array), 2])

    for i in range(len(s0_array)):
        temp_s = s0_array[i]
        if temp_s < 0:
            temp_t = track.getTFromS(temp_s + track.max_s)
        elif temp_s > track.max_s:
            temp_t = track.getTFromS(temp_s - track.max_s)
        else:
            temp_t = track.getTFromS(temp_s)
        pos[i, :] = track.pt_t(temp_t)

        # polynomial fitting
    x_axis = s0_array - s0
    sum_array = []
    for order in range(1, 20):
        coeff = np.polyfit(x_axis, pos, order)
        x_poly = np.polyval(coeff[:, 0], x_axis)
        y_poly = np.polyval(coeff[:, 1], x_axis)
        poly_val = np.vstack([x_poly, y_poly]).T
        norm_array = np.linalg.norm(poly_val - pos, axis=-1)
        sum_norm = np.sum(norm_array)
        sum_array.append(sum_norm)
        if sum_norm / len(x_axis) < 0.05:
            break

    best_order = np.argmin(sum_array) + 1
    print(f"ds: {ds}, order: {best_order}, sum norm {sum_array[best_order - 1]}")
    # weight = np.linspace(2,0.5,len(x_axis)) #weight can be added
    coeff = np.polyfit(x_axis, pos, best_order)

    # get the new (s0,n0) in new frame
    try:
        s0_new, n0_new = convertXYtoSN(track, coeff, y[0:2])
    except Exception as e:
        print(e)
    X0 = casadi.veccat(float(s0_new), float(n0_new), float(y[2]), float(y[3]), float(y[4]), float(y[5]),
                       float(y[6]))

    # construct the symbolic polynomial
    s = casadi.MX.sym('s')
    pos_mx = coeff[0, :] * casadi.power(s, best_order)
    for i in range(1, best_order):
        pos_mx += coeff[i, :] * casadi.power(s, best_order - i)
    pos_mx += coeff[-1, :]
    jac = casadi.jacobian(pos_mx, s)
    hes = casadi.jacobian(jac, s)
    kappa_mx = (jac[0] * hes[1] - jac[1] * hes[0]) / casadi.power(casadi.norm_2(jac), 3)
    phi_mx = casadi.arctan2(jac[1], jac[0])
    f_kappa = casadi.Function('kappa', [s], [kappa_mx])  # function of curvature
    f_phi = casadi.Function('phi', [s], [phi_mx])  # function of phi

    # define opti
    opti = casadi.Opti()
    X = opti.variable(nx, N + 1)
    X_dot = opti.variable(nx, N)
    U = opti.variable(nu, N)

    # states
    s_sym_array = X[0, :]
    n_sym_array = X[1, :]
    phi_sym_array = X[2, :]
    vx_sym_array = X[3, :]
    vy_sym_array = X[4, :]
    omega_sym_array = X[5, :]
    delta_sym_array = X[6, :]

    # control input
    delta_dot_sym_array = U[0, :]
    # front_throttle_array = U[1,:]
    # rear_throttle_array=U[2,:]
    throttle_array = U[1, :]
    front_brake_array = U[2, :]
    rear_brake_array = U[3, :]

    kappa_sym_array = f_kappa(s_sym_array[0:-1])
    dphi_c_sym_array = phi_sym_array[0:-1] - f_phi(s_sym_array[0:-1])

    fx_f_sym_array = cd * throttle_array - cm0 - cm1 * vx_sym_array[0:-1] * vx_sym_array[0:-1] - cm2 * vy_sym_array[0:-1] * vy_sym_array[0:-1] - cb * front_brake_array
    fx_r_sym_array = cd * throttle_array - cm0 - cm1 * vx_sym_array[0:-1] * vx_sym_array[0:-1] - cm2 * vy_sym_array[0:-1] * vy_sym_array[0:-1] - cb * rear_brake_array

    alpha_f = -casadi.atan2(omega_sym_array[0:-1] * lf + vy_sym_array[0:-1], vx_sym_array[0:-1]) + delta_sym_array[0:-1]
    alpha_r = casadi.atan2(omega_sym_array[0:-1] * lr - vy_sym_array[0:-1], vx_sym_array[0:-1])

    fy_f_sym_array = front_tire_model.getLateralForce(alpha_f, Fz[0])
    fy_r_sym_array = rear_tire_model.getLateralForce(alpha_r, Fz[1])

    # front_rate = casadi.fmax((fy_f_sym_array*fy_f_sym_array+fx_f_sym_array*fx_f_sym_array)/(Fy_f_max*Fy_f_max),1)
    # rear_rate = casadi.fmax((fy_r_sym_array*fy_r_sym_array+fx_r_sym_array*fx_r_sym_array)/(Fy_r_max*Fy_r_max),1)

    # fy_f_sym_array=fy_f_sym_array/(front_rate**0.5)
    # fy_r_sym_array=fy_r_sym_array/(rear_rate**0.5)
    # fx_f_sym_array=fx_f_sym_array/(front_rate**0.5)
    # fx_r_sym_array=fx_r_sym_array/(rear_rate**0.5)

    X_dot[0, :] = (vx_sym_array[0:-1] * casadi.cos(dphi_c_sym_array) - vy_sym_array[0:-1] * casadi.sin(
        dphi_c_sym_array)) / (1 - n_sym_array[0:-1] * kappa_sym_array)  # s_dot
    X_dot[1, :] = vx_sym_array[0:-1] * casadi.sin(dphi_c_sym_array) + vy_sym_array[0:-1] * casadi.cos(dphi_c_sym_array)  # n_dot
    X_dot[2, :] = omega_sym_array[0:-1]  # phi_dot
    X_dot[3, :] = 1 / mass * (fx_r_sym_array + fx_f_sym_array * casadi.cos(delta_sym_array[0:-1]) - fy_f_sym_array * casadi.sin(delta_sym_array[0:-1]) + mass * vy_sym_array[0:-1] * omega_sym_array[0:-1])  # vxdot
    X_dot[4, :] = 1 / mass * (fy_r_sym_array + fx_f_sym_array * casadi.sin(delta_sym_array[0:-1]) + fy_f_sym_array * casadi.cos(delta_sym_array[0:-1]) - mass * vx_sym_array[0:-1] * omega_sym_array[0:-1])  # vydot
    X_dot[5, :] = 1 / Iz * (fy_f_sym_array * lf * casadi.cos(delta_sym_array[0:-1]) + fx_f_sym_array * lf * casadi.sin(delta_sym_array[0:-1]) - fy_r_sym_array * lr)  # omegadot
    X_dot[6, :] = delta_dot_sym_array

    # objective
    n_obj = (casadi.atan(5 * (n_sym_array ** 2 - (track_width / 2) ** 2)) + casadi.pi / 2) * 50
    opti.minimize(-1000 * (s_sym_array[-1]) + casadi.dot(n_obj, n_obj))

    # initial condition
    opti.subject_to(X[:, 0] == X0)

    # dynamics
    opti.subject_to(X[:, 1:] == X[:, 0:-1] + dt * X_dot)

    # state bound
    # opti.subject_to(opti.bounded(0.0,vx_sym_array,v_max))
    opti.subject_to(opti.bounded(-track_width / 2, n_sym_array, track_width / 2))
    opti.subject_to(opti.bounded(delta_min, delta_sym_array, delta_max))

    # input bound
    opti.subject_to(opti.bounded(delta_dot_min, delta_dot_sym_array, delta_dot_max))
    opti.subject_to(opti.bounded(0, throttle_array, 1))
    opti.subject_to(opti.bounded(0, front_brake_array, 1))
    opti.subject_to(opti.bounded(0, rear_brake_array, 1))

    # initial guess:
    if sol_x is None:
        for i in range(N):
            guess_X[:, i] = X0
    else:
        guess_X[:, 0] = X0
        guess_U[:, 0:-1] = sol_u[:, 1:]
        guess_U[:, -1] = sol_u[:, -1]
        yt = copy.deepcopy(y)
        for i in range(1, N + 1):
            yt = yt + dt * forwardModel(yt, guess_U[:, i - 1])
            t_guess, n_guess = convertXYtoSN(track, coeff, yt[0:2], float(sol_x[0, i]))
            guess_X[:, i] = casadi.vertcat(float(t_guess), float(n_guess), float(yt[2]), float(yt[3]), float(yt[4]),
                                           float(yt[5]), float(yt[6]))

    ds_guess = guess_X[0, 1:] - guess_X[0, 0:-1]

    assert (np.all(np.array(ds_guess)) >= 0)

    opti.set_initial(X, guess_X)
    opti.set_initial(U, guess_U)

    opti.solver("ipopt", {}, option)
    try:
        sol = opti.solve()  # actual solve
    except Exception as e:
        plt.plot(pos[:, 0], pos[:, 1])
        plt.show()
        print(guess_X)
        print(e)

        exit()

    computation_time = sol.stats()['t_proc_total']

    sol_x = sol.value(X)
    sol_u = sol.value(U)

    # using the (x,y) model to check the parametric model
    plane_x = casadi.DM.zeros(nx, N + 1)
    plane_x[:, 0] = y
    for i in range(1, N + 1):
        plane_x[:, i] = plane_x[:, i - 1] + dt * forwardModel(plane_x[:, i - 1], sol_u[:, i - 1])

    sql_ptx = ','.join("{:0.2f}".format(float(plane_x[0, i])) for i in range(N + 1))
    sql_pty = ','.join("{:0.2f}".format(float(plane_x[1, i])) for i in range(N + 1))

    co = np.resize(coeff, (2 * len(coeff),))
    sql_coeff = ','.join("{:0.3e}".format(co[i]) for i in range(len(co)))

    sql_phi = float(sol.value(phi_sym_array)[0])
    sql_vx = float(sol.value(vx_sym_array)[0])
    sql_vy = float(sol.value(vy_sym_array)[0])

    sql_front_fx = float(sol.value(fx_f_sym_array)[0])
    sql_rear_fx = float(sol.value(fx_r_sym_array)[0])
    sql_front_fy = float(sol.value(fy_f_sym_array)[0])
    sql_rear_fy = float(sol.value(fy_r_sym_array)[0])

    sql_front_wheel_alpha = float(sol.value(alpha_f)[0])
    sql_rear_wheel_alpha = float(sol.value(alpha_r)[0])

    sql_steer = float(sol.value(delta_sym_array)[0])
    sql_omega = float(sol.value(omega_sym_array)[0])

    sql_query = "insert into {} values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(table_name)
    cur.execute(sql_query,
                (computation_time, sql_phi, sql_vx, sql_vy, sql_steer, sql_omega, sql_front_wheel_alpha,
                 sql_rear_wheel_alpha, sql_front_fx, sql_rear_fx, sql_front_fy, sql_rear_fy, s0, ds, float(best_order),
                 sql_ptx, sql_pty, sql_coeff))
    con.commit()

    return plane_x[:, 1]


if __name__ == '__main__':

    while s0 < init_s0 + track.max_s:
        y0 = optimize(y0)
        temp_t, temp_n = track.convertXYtoTN([float(y0[0]), float(y0[1])])
        s0 = track.getSFromT(temp_t)
        if s0 - init_s0 < 0:
            s0 = s0 + track.max_s
        print(f"finished: {float((s0 - init_s0) / track.max_s) * 100:.2f}%")
        print(y0)
