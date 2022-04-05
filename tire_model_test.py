from cProfile import label
import casadi
from matplotlib.pyplot import figure
from track import *
from dynamics_models import *
from tire_model import *

#define tire mode
with open('params/racecar_simple_tire_front.yaml') as file:
    front_tire_params = yaml.load(file)
    front_B_long = front_tire_params['B_long']
    front_C_long = front_tire_params['C_long']
    front_D_long = front_tire_params['D_long']
    front_B_lat = front_tire_params['B_lat']
    front_C_lat = front_tire_params['C_lat']
    front_D_lat = front_tire_params['D_lat']



Fz = 1
alpha = np.linspace(-1,1,100)
lamb = np.linspace(-1,1,100)

front_fx = front_D_long*casadi.sin(front_C_long*casadi.atan(front_B_long*lamb))  
front_fy = front_D_lat*casadi.sin(front_C_lat*casadi.atan(front_B_lat*alpha)) 


front_fx_simple = 0.4*lamb
front_fx_simple = casadi.fmin(front_fx_simple,0.17)
front_fx_simple = casadi.fmax(front_fx_simple,-0.17)

fx_unity = 2.5*lamb
index = np.where(lamb<-0.4)
fx_unity[index] = -1.25*lamb[index]-1.5

index = np.where(lamb>0.4)
fx_unity[index] = -1.25*lamb[index]+1.5

index = np.where(lamb<-0.8)
fx_unity[index] = -0.5

index = np.where(lamb>0.8)
fx_unity[index] = 0.5

fy_unity = 5*alpha
index = np.where(alpha<-0.2)
fy_unity[index] = -0.25/0.3*alpha[index]-7.0/6.0

index = np.where(alpha>0.2)
fy_unity[index] = -0.25/0.3*alpha[index]+7.0/6.0

index = np.where(alpha<-0.5)
fy_unity[index] = -0.75

index = np.where(alpha>0.5)
fy_unity[index] = 0.75



plt.plot(lamb,front_fx,label='fx_simple_pac')
plt.plot(alpha,front_fy,label='fy_simple_pac')

plt.plot(lamb,fx_unity,label='fx_unity')
plt.plot(alpha,fy_unity,label='fy_unity')


#plt.plot(lamb,front_fx_simple,label='fy_simple')
plt.legend()
plt.show()