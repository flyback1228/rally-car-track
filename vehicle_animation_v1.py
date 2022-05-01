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
    
   
def animate(x_ax,data,ptxs,ptys,vehicle_length,vehicle_width,lines,i):
    #sql_phi,sql_vx,sql_vy,sql_steer,sql_d,sql_omega,sql_front_wheel_omega,sql_rear_wheel_omega,sql_front_wheel_alpha,sql_rear_wheel_alpha,sql_front_wheel_lamb,sql_rear_wheel_lamb,sql_front_brake,sql_rear_brake,sql_ptx,sql_pty
    theta = data[0:i+1,1]

    ptx = np.fromstring(ptxs[i], dtype=float, sep=',')
    pty = np.fromstring(ptys[i], dtype=float, sep=',')
    
    vx = data[0:i+1,2]
    vy = data[0:i+1,3]
    steer = data[0:i+1,4]

    front_wheel_fx = data[0:i+1,8]
    rear_wheel_fx = data[0:i+1,9]
    front_wheel_fy = data[0:i+1,10]
    rear_wheel_fy = data[0:i+1,11]
    
    omega = data[0:i+1,5]
    
    
    front_wheel_alpha = data[0:i+1,6]
    rear_wheel_alpha = data[0:i+1,7]
    
    position = np.array([ptx[0],pty[0]])
            
    rot_mat = np.array([[np.cos(theta[-1]),-np.sin(theta[-1])],[np.sin(theta[-1]),np.cos(theta[-1])]]).reshape(2,2)
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
    
    lines[4].set_data(x_ax[0:i+1],vx)
    lines[5].set_data(x_ax[0:i+1],vy)
    
    lines[6].set_data(x_ax[0:i+1],front_wheel_fx)
    lines[7].set_data(x_ax[0:i+1],rear_wheel_fx)
    lines[8].set_data(x_ax[0:i+1],front_wheel_fy)
    lines[9].set_data(x_ax[0:i+1],rear_wheel_fy)
    
    lines[10].set_data(x_ax[0:i+1],front_wheel_alpha)
    lines[11].set_data(x_ax[0:i+1],rear_wheel_alpha)
    
    lines[12].set_data(x_ax[0:i+1],steer)
    lines[13].set_data(x_ax[0:i+1],omega)
    lines[14].set_data(x_ax[0:i+1],theta)
    
    return lines

def plot(track,data,params,dt):
    fig,ax = plt.subplots()
    num_rows = len(data)
    total_time = num_rows*dt;

    np_data = np.array(data)
    mpc_data = (np_data[:,:-3]).astype(float)
    ptxs = np_data[:,-3]
    ptys = np_data[:,-2]

    #with open('params/racecar.yaml') as file:
    #    params = yaml.load(file)
    vehicle_length = (params['lf']+params['lr'])*1.2
    vehicle_width = params['width']
    
    v_max1 = np.amax(mpc_data[:,2])
    v_max2 = np.amax(mpc_data[:,3])
    #v_max3 = np.amax(mpc_data[:,8])
    v_max = np.amax([v_max1,v_max2])
    v_min1 = np.amin(mpc_data[:,2])
    v_min2 = np.amin(mpc_data[:,3])
    v_min = np.amin([v_min1,v_min2])
    #v_min = np.amin([np.amin(mpc_data[:,3]),np.amin(mpc_data[:,2])])
    
    front_wheel_fx = mpc_data[:,8]
    rear_wheel_fx = mpc_data[:,9]
    front_wheel_fy = mpc_data[:,10]
    rear_wheel_fy = mpc_data[:,11]
    
    front_wheel_alpha = mpc_data[:,6]
    rear_wheel_alpha = mpc_data[:,7]
    #front_wheel_lambda = mpc_data[:,11]
    #rear_wheel_lambda = mpc_data[:,12]


    #d = mpc_data[:,5]
    steer = mpc_data[:,4]
    phi = mpc_data[:,1]
    omega = mpc_data[:,5]

    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(4,2,2,xlim=(-total_time/10,total_time*1.1),ylim=(v_min-np.abs(v_min)*0.1,v_max*1.2))
    ax3 = plt.subplot(4,2,4,xlim=(-total_time/10,total_time*1.1),ylim=(-0.1,0.6))
    ax4 = plt.subplot(4,2,6,xlim=(-total_time/10,total_time*1.1))
    ax5 = plt.subplot(4,2,8,xlim=(-total_time/10,total_time*1.1))


    ax2.get_xaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    ax4.get_xaxis().set_ticks([])
    ax5.get_xaxis().set_ticks([])
    


    lines = []
    lines.append(ax1.plot([], [], lw=8,solid_capstyle="butt", zorder=1, color='c' )[0])
    lines.append(ax1.plot([], [], lw=4,solid_capstyle="butt", zorder=1, color='m' )[0])
    lines.append(ax1.plot([], [], lw=4,solid_capstyle="butt", zorder=1, color='r' )[0])
    lines.append(ax1.plot([],[], lw=2,solid_capstyle="butt", zorder=1, color='r')[0])

    lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='vx')[0])
    lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='vy' )[0])
    #lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='front wheel speed' )[0])
    #lines.append(ax2.plot([], [],solid_capstyle="butt", zorder=1,label='rear wheel speed' )[0])

    lines.append(ax3.plot([], [],solid_capstyle="butt", zorder=1,label='front fx' )[0])
    lines.append(ax3.plot([], [],solid_capstyle="butt", zorder=1,label='rear fx' )[0])
    lines.append(ax3.plot([], [],solid_capstyle="butt", zorder=1,label='front fy' )[0])
    lines.append(ax3.plot([], [],solid_capstyle="butt", zorder=1,label='rear fy' )[0])

    lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='front alpha')[0])
    lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='rear alpha' )[0])
    #lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='front lambda' )[0])
    #lines.append(ax4.plot([], [],solid_capstyle="butt", zorder=1,label='front lambda' )[0])

    lines.append(ax5.plot([], [],solid_capstyle="butt", zorder=1 ,label='steer')[0])
    lines.append(ax5.plot([], [],solid_capstyle="butt", zorder=1 ,label='omega' )[0])
    lines.append(ax5.plot([], [],solid_capstyle="butt", zorder=1 ,label='phi' )[0])
    
    ax3_max = max([max(front_wheel_fx),max(front_wheel_fy),max(rear_wheel_fx),max(rear_wheel_fy)])
    ax3_min = min([min(front_wheel_fx),min(front_wheel_fy),min(rear_wheel_fx),min(rear_wheel_fy)])
    ax3.set_ylim(ax3_min-abs(ax3_min/10),ax3_max+abs(ax3_max)/10)
    
    ax4_max = max([max(front_wheel_alpha),max(rear_wheel_alpha)])
    ax4_min = min([min(front_wheel_alpha),min(rear_wheel_alpha)])
    ax4.set_ylim(ax4_min-abs(ax4_min/10),ax4_max+abs(ax4_max)/10)
    
    
    
    ax5_max = max(max(steer),max(omega),max(phi))
    ax5_min = min(min(steer),min(omega),min(phi))
    ax5.set_ylim(ax5_min-abs(ax5_min/10),ax5_max+abs(ax5_max)/10)
    
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    
    fig.set_size_inches(21, 10.5)
    track.plot(ax1)
    x_ax = np.linspace(0,total_time,num_rows+1)
    
    anim = animation.FuncAnimation(fig, functools.partial(animate,x_ax,mpc_data,ptxs,ptys,vehicle_length,vehicle_width,lines),interval=dt,frames=num_rows,repeat=True, blit=True)
    #anim.save('image/twowheelwithbrake_sim.gif', writer=animation.PillowWriter(fps=10),dpi=100)
        
    #fig2 = plt.figure()
    #plt.scatter(range(num_rows),mpc_data[:,0])
    #plt.title("computational time")
    plt.show()


if __name__=='__main__':
    table_name = 'unity_04_26_2022_12_30_15'
    
    con = sqlite3.connect('output/sql_data.db')
    cur = con.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    data = cur.fetchall() 
    
       
    dt =0.1
    track_width = 6  
    track = SymbolicTrack('tracks/temp_nwh.csv',track_width)
    
    with open('params/racecar_nwh.yaml') as file:
        params = yaml.safe_load(file)
    plot(track,data,params,0.1)