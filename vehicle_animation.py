import imp
from turtle import position
import numpy as np
from track import SymbolicTrack
import matplotlib.pyplot as plt
from matplotlib import animation
import functools
import sqlite3
import yaml

def plotVehicle(position,theta,vehicle_length,vehicle_width,ax):
    rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).reshape(2,2)
    #print(rot_mat)
    #print(np.array([[-vehicle_length/2],[0]]))
    
    tail = np.matmul(rot_mat,np.array([[-vehicle_length/2],[0]])).reshape(2,)+position
    
    header = np.matmul(rot_mat,np.array([vehicle_length/2,0]))+position
    left_light_start = np.matmul(rot_mat,np.array([vehicle_length/2,vehicle_width/4]))+position
    left_light_end = np.matmul(rot_mat,np.array([vehicle_length/2+vehicle_length/5,vehicle_width/4]))+position
    right_light_start = np.matmul(rot_mat,np.array([vehicle_length/2,-vehicle_width/4]))+position
    right_light_end = np.matmul(rot_mat,np.array([vehicle_length/2+vehicle_length/5,-vehicle_width/4]))+position
    
    ax.plot([tail[0],header[0]], [tail[1],header[1]], lw=8,solid_capstyle="butt", zorder=1, color='c' )
    ax.plot([left_light_start[0],left_light_end[0]], [left_light_start[1],left_light_end[1]], lw=4,solid_capstyle="butt", zorder=1, color='m' )
    ax.plot([right_light_start[0],right_light_end[0]], [right_light_start[1],right_light_end[1]], lw=4,solid_capstyle="butt", zorder=1, color='m' )
    
   
def animate(x_ax,data,vehicle_length,vehicle_width,lines,i):
    #sql_phi,sql_vx,sql_vy,sql_steer,sql_d,sql_omega,sql_front_wheel_omega,sql_rear_wheel_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_wheel_lamb,sql_rear_wheel_lamb,sql_front_brake,sql_rear_brake,sql_ptx,sql_pty
    theta = data[i][1]
    ptx_str = data[i][15]
    pty_str = data[i][16]
    ptx = np.fromstring(ptx_str, dtype=float, sep=',')
    pty = np.fromstring(pty_str, dtype=float, sep=',')
    
    vx = data[i][2]
    vy = data[i][3]
    steer = data[i][4]
    d = data[i][5]
    
    omega = data[i][6]
    
    front_wheel_omega = data[i][7]
    rear_wheel_omega = data[i][8]
    front_wheel_alpha = data[i][9]
    rear_wheel_alpha = data[i][10]
    front_wheel_lambda = data[i][11]
    rear_wheel_lambda = data[i][12]
    front_wheel_brake = data[i][13]
    rear_wheel_brake = data[i][14]
    
    position = np.array([ptx[0],pty[0]])
            
    rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).reshape(2,2)
    #print(rot_mat)
    #print(np.array([[-vehicle_length/2],[0]]))
    
    tail = np.matmul(rot_mat,np.array([[-vehicle_length/2],[0]])).reshape(2,)+position
    
    header = np.matmul(rot_mat,np.array([vehicle_length/2,0]))+position
    left_light_start = np.matmul(rot_mat,np.array([vehicle_length/2,vehicle_width/4]))+position
    left_light_end = np.matmul(rot_mat,np.array([vehicle_length/2+vehicle_length/5,vehicle_width/4]))+position
    right_light_start = np.matmul(rot_mat,np.array([vehicle_length/2,-vehicle_width/4]))+position
    right_light_end = np.matmul(rot_mat,np.array([vehicle_length/2+vehicle_length/5,-vehicle_width/4]))+position       
    
    lines[0].set_data([tail[0],header[0]], [tail[1],header[1]])
    lines[1].set_data([left_light_start[0],left_light_end[0]],[left_light_start[1],left_light_end[1]])
    lines[2].set_data([right_light_start[0],right_light_end[0]],[right_light_start[1],right_light_end[1]])
    lines[3].set_data(ptx,pty)
    
    lines[4].set_data(x_ax[0:i],vx)
    lines[5].set_data(x_ax[0:i],vy)
    lines[6].set_data(x_ax[0:i],front_wheel_omega)
    lines[7].set_data(x_ax[0:i],rear_wheel_omega)
    
    lines[8].set_data(x_ax[0:i],front_wheel_brake)
    lines[9].set_data(x_ax[0:i],rear_wheel_brake)
    
    lines[10].set_data(x_ax[0:i],front_wheel_alpha)
    lines[11].set_data(x_ax[0:i],rear_wheel_alpha)
    lines[12].set_data(x_ax[0:i],front_wheel_lambda)
    lines[13].set_data(x_ax[0:i],rear_wheel_lambda)
    
    lines[14].set_data(x_ax[0:i],steer)
    lines[15].set_data(x_ax[0:i],d)
    
    lines[16].set_data(x_ax[0:i],omega)
    lines[17].set_data(x_ax[0:i],theta)
    
    return lines

def plot(track,data,vehicle_length,vehicle_width,dt):
    fig,ax = plt.subplots()
    num_rows = len(data)
    total_time = num_rows*dt;

    v_max = 0
    computation_time = []
    for i in range(num_rows):
        v_max = max(v_max,data[i][2])
        computation_time.append(data[i][0])

    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(5,2,2,xlim=(-total_time/10,total_time*1.1),ylim=(-1-v_max*0.1,v_max*1.1))
    ax3 = plt.subplot(5,2,4,xlim=(-total_time/10,total_time*1.1),ylim=(-0.1,0.6))
    ax4 = plt.subplot(5,2,6,xlim=(-total_time/10,total_time*1.1))
    ax5 = plt.subplot(5,2,8,xlim=(-total_time/10,total_time*1.1))
    ax6 = plt.subplot(5,2,10,xlim=(-total_time/10,total_time*1.1))

    ax2.get_xaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    ax4.get_xaxis().set_ticks([])
    ax5.get_xaxis().set_ticks([])

    lines = []
    lines.append(ax1.plot([], [], lw=8,solid_capstyle="butt", zorder=1, color='c' )[0])
    lines.append(ax1.plot([], [], lw=4,solid_capstyle="butt", zorder=1, color='m' )[0])
    lines.append(ax1.plot([], [], lw=4,solid_capstyle="butt", zorder=1, color='b' )[0])
    lines.append(ax1.plot([],[])[0])

    lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='vx')[0])
    lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='vy' )[0])
    lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='front wheel speed' )[0])
    lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='rear wheel speed' )[0])

    lines.append(ax3.plot([], [],solid_capstyle="butt", zorder=1,label='front brake' )[0])
    lines.append(ax3.plot([], [],solid_capstyle="butt", zorder=1,label='rear brake' )[0])

    lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='front alpha')[0])
    lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='rear alpha' )[0])
    lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='front lambda' )[0])
    lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='front lambda' )[0])

    lines.append(ax5.plot([], [],solid_capstyle="butt", zorder=1 ,label='steer')[0])
    lines.append(ax5.plot([], [],solid_capstyle="butt", zorder=1 ,label='throttle')[0])

    lines.append(ax6.plot([], [],solid_capstyle="butt", zorder=1 ,label='omega' )[0])
    lines.append(ax6.plot([], [],solid_capstyle="butt", zorder=1 ,label='phi' )[0])
    
    fig.set_size_inches(21, 10.5)
    track.plot(ax1)
    x_ax = np.linspace(0,total_time,num_rows+1)
    
    anim = animation.FuncAnimation(fig, functools.partial(animate,x_ax,data,vehicle_length,vehicle_width,lines),interval=dt,frames=num_rows,repeat=True, blit=True)
    #anim.save('image/twowheelwithbrake_sim.gif', writer=animation.PillowWriter(fps=10),dpi=100)
    
    
    
    fig2 = plt.figure()
    plt.scatter(range(num_rows),computation_time)
    plt.show()


if __name__=='__main__':
    table_name = 'global_03_29_2022_11_31_31'
    con = sqlite3.connect('output/sql_data.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM {}".format(table_name))
    data = cur.fetchall()    
    dt =0.1
    track_width = 4   
    track = SymbolicTrack('tracks/temp.csv',track_width)
    
    with open('params/racecar.yaml') as file:
        params = yaml.load(file)
    vehicle_length = (params['lf']+params['lr'])*1.2
    vehicle_width = params['width']
    plot(track,data,vehicle_length,vehicle_width,0.1)