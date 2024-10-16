from abc import ABC, abstractmethod
from ast import Param
from casadi.casadi import tangent
import yaml
import casadi

class DynamicsModel(ABC):
    def __init__(self) -> None: 
        self.nx = 0
        self.nu = 0

    @abstractmethod
    def update(self,x,u):
        pass

class BicycleKineticModel(DynamicsModel):
    def __init__(self, params) -> None:
        super().__init__()
        self.lf = params['lf']
        self.lr = params['lr']
        self.nx = 4
        self.nu = 2

    def update(self,x,u):        
        
        #x = casadi.MX.sym('x',self.nx)
        #u = casadi.MX.sym('u',self.nu)
        phi = x[2]
        vx = x[3]

        delta = u[0]
        d = u[1]

        dot_x = casadi.vertcat(
            vx*casadi.cos(phi),
            vx*casadi.sin(phi),
            vx/(self.lf+self.lr) * casadi.tan(delta),
            d
        )
        #ode = casadi.Function('ode',[x,u],[dot_x],['state','input'],['dxdt'])
        return dot_x

class BicycleKineticModelByParametricArc(DynamicsModel):
    def __init__(self, params,track) -> None:
        super().__init__()
        self.lf = params['lf']
        self.lr = params['lr']
        self.nx = 4
        self.nu = 2
        self.track = track


    def update(self,x,u):        
        
        #x = casadi.MX.sym('x',self.nx)
        #u = casadi.MX.sym('u',self.nu)
        if self.track is None:
            return
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]

        delta = u[0]
        d = u[1]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t)

        t_dot = vx*casadi.cos(phi-phi_c+delta)/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c+delta)
        phi_dot = vx/(self.lf+self.lr) * casadi.tan(delta)
        #v_dot = 5.9*d-0.136*vx
        v_dot = 7.9*d
        
        dot_x = casadi.vertcat(
            t_dot,
            n_dot,
            phi_dot,
            v_dot
        )
        #ode = casadi.Function('ode',[x,u],[dot_x],['state','input'],['dxdt'])
        return dot_x
    
class BicycleKineticModelByParametricArcWithBrake(DynamicsModel):
    def __init__(self, params,track) -> None:
        super().__init__()
        self.lf = params['lf']
        self.lr = params['lr']
        self.nx = 4
        self.nu = 3
        self.track = track


    def update(self,x,u):        
        
        #x = casadi.MX.sym('x',self.nx)
        #u = casadi.MX.sym('u',self.nu)
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]

        delta = u[0]
        d = u[1]
        b = u[2]
        
        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t)

        t_dot = vx*casadi.cos(phi-phi_c+delta)/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c+delta)
        phi_dot = vx/(self.lf+self.lr) * casadi.tan(delta)
        v_dot = 7.9*d-14.26*b
        
        dot_x = casadi.vertcat(
            t_dot,
            n_dot,
            phi_dot,
            v_dot
        )
        #ode = casadi.Function('ode',[x,u],[dot_x],['state','input'],['dxdt'])
        return dot_x    

class BicycleKineticModelWithSteerDot(DynamicsModel):
    def __init__(self, params,track) -> None:
        super().__init__()
        self.lf = params['lf']
        self.lr = params['lr']
        self.nx = 5
        self.nu = 2
        self.track = track


    def update(self,x,u):        
        
        #x = casadi.MX.sym('x',self.nx)
        #u = casadi.MX.sym('u',self.nu)
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        delta=x[4]

        delta_dot = u[0]
        d = u[1]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t)

        t_dot = vx*casadi.cos(phi-phi_c+delta)/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c+delta)
        phi_dot = vx/(self.lf+self.lr) * casadi.tan(delta)
        v_dot = 5.5*d-0.1*vx
        
        dot_x = casadi.vertcat(
            t_dot,
            n_dot,
            phi_dot,
            v_dot,
            delta_dot
        )
        #ode = casadi.Function('ode',[x,u],[dot_x],['state','input'],['dxdt'])
        return dot_x

class BicycleDynamicsModelByParametricArc(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:
        super().__init__()
        self.lf = params['lf']
        self.lr = params['lr']
        self.m = params['m']
        self.Iz = params['Iz']

        self.nx = 6
        self.nu = 2
        self.track = track

        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model


    def update(self,x,u):        
        
        #x = [t,n,phi,vx,vy,omega]
        #u = [delta,d]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        delta = u[0]
        d = u[1]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t)
        
        #need work
        alphaf = -casadi.atan2(omega*self.lf + vy, vx+0.03) + delta
        alphar = casadi.atan2(omega*self.lr - vy,vx+0.03)
        #miou = casadi.MX.sym('miou')
        #fz = casadi.MX.sym('fz')
        Ff = self.front_tire_model.getForce(alphaf,vx,d)
        Fr = self.rear_tire_model.getForce(alphar,vx,d)
        
        Frx = Fr[0]
        Fry = Fr[1]
        Ffy = Ff[1]

        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        vx_dot = 1/self.m * (Frx - Ffy*casadi.sin(delta) + self.m*vy*omega)   #vxdot
        vy_dot = 1/self.m * (Fry + Ffy*casadi.cos(delta) - self.m*vx*omega)  #vydot
        omega_dot = 1/self.Iz * (Ffy*self.lf*casadi.cos(delta) - Fry*self.lr)       #omegadot
     
        dot_x = casadi.vertcat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot
        )
        return dot_x


class BicycleDynamicsModelWithSteerDot(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:
        super().__init__()
        self.lf = params['lf']
        self.lr = params['lr']
        self.m = params['m']
        self.Iz = params['Iz']

        self.nx = 8
        self.nu = 2
        self.track = track

        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

    def update(self,x,u):        
        
        #x = [t,n,phi,vx,vy,omega]
        #u = [delta,d]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        d = x[7]

        steer_dot = u[0]
        d_dot = u[1]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t)
        
        #need work
        alphaf = -casadi.atan2(omega*self.lf + vy, vx+0.03) + steer
        alphar = casadi.atan2(omega*self.lr - vy,vx+0.03)
        #miou = casadi.MX.sym('miou')
        #fz = casadi.MX.sym('fz')
        Ff = self.front_tire_model.getForce(alphaf,vx,d)
        Fr = self.rear_tire_model.getForce(alphar,vx,d)
        
        Frx = Fr[0]
        Fry = Fr[1]
        Ffy = Ff[1]

        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        vx_dot = 1/self.m * (Frx - Ffy*casadi.sin(steer) + self.m*vy*omega)   #vxdot
        vy_dot = 1/self.m * (Fry + Ffy*casadi.cos(steer) - self.m*vx*omega)  #vydot
        omega_dot = 1/self.Iz * (Ffy*self.lf*casadi.cos(steer) - Fry*self.lr)       #omegadot
     
        dot_x = casadi.vertcat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            d_dot
        )
        return dot_x



class RacecarDynamicsModel(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 6
        self.nx = 11
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model
        
        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.width = params['width']
        self.wheel_radius = params['wheel_radius']
        self.wheel_inertia = params['wheel_inertia']
        self.mass = params['m']
        self.Iz = params['Iz']
        self.a = 1
        self.b = 0.1
        self.kc = 0.2
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,throttle,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        wheel_omega = x[7:11]
        
        steer_dot = u[0]
        d = u[1]
        brake = u[2:]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        
        speed_at_wheel = casadi.veccat(casadi.fmax((vy+omega*self.lf)**2 + (vx-omega*self.width/2)**2,0.001)**0.5,
                                       casadi.fmax((vy+omega*self.lf)**2 + (vx+omega*self.width/2)**2,0.001)**0.5,
                                       casadi.fmax((vy-omega*self.lr)**2 + (vx-omega*self.width/2)**2,0.001)**0.5,
                                       casadi.fmax((vy-omega*self.lr)**2 + (vx+omega*self.width/2)**2,0.001)**0.5)
        
        #slipping ratio & angle
        #need work
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx-omega*self.width/2+0.01) + steer,
                              -casadi.atan2(omega*self.lf + vy, vx+omega*self.width/2+0.01) + steer,
                              casadi.atan2(omega*self.lr - vy,vx-omega*self.width/2+0.01),
                              casadi.atan2(omega*self.lr - vy,vx+omega*self.width/2+0.01))        
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
        lamb = -1 + self.wheel_radius*x[7:]/speed_at_wheel
        
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(4)*0.25*self.mass*9.81
        
        F_Fl = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        F_Fr = self.front_tire_model.getForce(lamb[1],alpha[1],Fz[1])
        F_Rl = self.rear_tire_model.getForce(lamb[2],alpha[2],Fz[2])
        F_Rr = self.rear_tire_model.getForce(lamb[3],alpha[3],Fz[3])
        
        F_Fx = F_Fl[0]+F_Fr[0]
        F_Fy = F_Fl[1]+F_Fr[1]
        F_Rx = F_Rl[0]+F_Rr[0]
        F_Ry = F_Rl[1]+F_Rr[1]
        
        Fx = casadi.veccat(F_Fl[0],F_Fr[0],F_Rl[0],F_Rr[0])
        Fy = casadi.veccat(F_Fl[1],F_Fr[1],F_Rl[1],F_Rr[1])

        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        vx_dot = 1/self.mass * (F_Fx*casadi.cos(steer) + F_Rx - F_Fy*casadi.sin(steer)) + vy*omega   #vxdot
        vy_dot = 1/self.mass * (F_Fx*casadi.sin(steer) + F_Ry + F_Fy*casadi.cos(steer)) - vx*omega  #vydot
        omega_dot = 1/self.Iz * ((F_Fx*casadi.sin(steer) + F_Fy*casadi.cos(steer))*self.lf
                    -F_Ry*self.lr 
                    + ((-Fx[0]+Fx[1])*casadi.cos(steer) + (-Fy[0]+Fy[1])*casadi.sin(steer) -Fx[2] +Fx[3] )*self.width/2 )      #omegadot
     
     
        v_wheel = (wheel_omega[0]+wheel_omega[1])/2*self.wheel_radius
        K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        
        #p = ((brake[1]+brake[0]+Fx[1]+Fx[0])-(brake[3]+brake[2]+Fx[3]+Fx[2])+2*K)/(4*K)
        p = 0.5
        front_wheel_omega_dot = (K*p - Fx[0:2] - brake[0:2])*self.wheel_radius/self.wheel_inertia
        rear_wheel_omega_dot = (K*(1-p) - Fx[2:] - brake[2:])*self.wheel_radius/self.wheel_inertia
        
        
         
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            front_wheel_omega_dot,
            rear_wheel_omega_dot
        )
        return dot_x
        

class RacecarDynamicsModelAlternative(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 4
        self.nx = 11
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model
        
        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.width = params['width']
        self.wheel_radius = params['wheel_radius']
        self.wheel_inertia = params['wheel_inertia']
        self.mass = params['m']
        self.Iz = params['Iz']
        self.a = 1
        self.b = 0.1
        self.kc = 0.2
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,throttle,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        wheel_omega = x[7:11]
        
        steer_dot = u[0]
        d = u[1]
        brake = u[2:]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        
        speed_at_wheel = casadi.veccat(casadi.fmax((vy+omega*self.lf)**2 + (vx-omega*self.width/2)**2,0.001)**0.5,
                                       casadi.fmax((vy+omega*self.lf)**2 + (vx+omega*self.width/2)**2,0.001)**0.5,
                                       casadi.fmax((vy-omega*self.lr)**2 + (vx-omega*self.width/2)**2,0.001)**0.5,
                                       casadi.fmax((vy-omega*self.lr)**2 + (vx+omega*self.width/2)**2,0.001)**0.5)
        
        #slipping ratio & angle
        #need work
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx-omega*self.width/2+0.01) + steer,
                              -casadi.atan2(omega*self.lf + vy, vx+omega*self.width/2+0.01) + steer,
                              casadi.atan2(omega*self.lr - vy,vx-omega*self.width/2+0.01),
                              casadi.atan2(omega*self.lr - vy,vx+omega*self.width/2+0.01))        
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
        lamb = -1 + self.wheel_radius*x[7:]/speed_at_wheel
        
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(4)*0.25*self.mass*9.81
        
        F_Fl = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        F_Fr = self.front_tire_model.getForce(lamb[1],alpha[1],Fz[1])
        F_Rl = self.rear_tire_model.getForce(lamb[2],alpha[2],Fz[2])
        F_Rr = self.rear_tire_model.getForce(lamb[3],alpha[3],Fz[3])
        
        F_Fx = F_Fl[0]+F_Fr[0]
        F_Fy = F_Fl[1]+F_Fr[1]
        F_Rx = F_Rl[0]+F_Rr[0]
        F_Ry = F_Rl[1]+F_Rr[1]
        
        Fx = casadi.veccat(F_Fl[0],F_Fr[0],F_Rl[0],F_Rr[0])
        Fy = casadi.veccat(F_Fl[1],F_Fr[1],F_Rl[1],F_Rr[1])
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        vx_dot = 1/self.mass * (F_Fx*casadi.cos(steer) + F_Rx - F_Fy*casadi.sin(steer)) + vy*omega   #vxdot
        vy_dot = 1/self.mass * (F_Fx*casadi.sin(steer) + F_Ry + F_Fy*casadi.cos(steer)) - vx*omega  #vydot
        omega_dot = 1/self.Iz * ((F_Fx*casadi.sin(steer) + F_Fy*casadi.cos(steer))*self.lf
                    -F_Ry*self.lr 
                    + ((-Fx[0]+Fx[1])*casadi.cos(steer) + (-Fy[0]+Fy[1])*casadi.sin(steer) -Fx[2] +Fx[3] )*self.width/2 )      #omegadot
     
     
        

        #t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        #n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        #phi_dot = omega
        
        #vx_dot = 1/self.mass * ((F_Fl[0]+F_Fr[0])*casadi.cos(steer) + (F_Rl[0]+F_Rr[0]) - (F_Fl[1]+F_Fr[1])*casadi.sin(steer)) + vy*omega   #vxdot
        #vy_dot = 1/self.mass * ((F_Fl[0]+F_Fr[0])*casadi.sin(steer) + (F_Rl[1]+F_Rr[1]) + (F_Fl[1]+F_Fr[1])*casadi.cos(steer)) - vx*omega  #vydot
        #omega_dot = 1/self.Iz * (((F_Fl[0]+F_Fr[0])*casadi.sin(steer) + (F_Fl[1]+F_Fr[1])*casadi.cos(steer))*self.lf
        #            -(F_Rl[1]+F_Rr[1])*self.lr 
        #            + ((-F_Fl[0]+F_Fr[0])*casadi.cos(steer) + (-F_Fl[1]+F_Fr[1])*casadi.sin(steer) -F_Rl[0] +F_Rr[0] )*self.width/2 )      #omegadot
     
     
        v_wheel = (wheel_omega[0]+wheel_omega[1])/2*self.wheel_radius
        K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        
        p = ((casadi.fabs(brake[0])+Fx[1]+Fx[0])-(casadi.fabs(brake[1])+Fx[3]+Fx[2])+2*K)/(4*K)
        wheel_omega_dot_fl = (K*p - F_Fl[0] - casadi.fmax(0,brake[0]))*self.wheel_radius/self.wheel_inertia
        wheel_omega_dot_fr = (K*p - F_Fr[0] + casadi.fmin(0,brake[0]))*self.wheel_radius/self.wheel_inertia
        wheel_omega_dot_rl = (K*(1-p) - F_Rl[0] - casadi.fmax(0,brake[1]))*self.wheel_radius/self.wheel_inertia
        wheel_omega_dot_rr = (K*(1-p) - F_Rr[0] + casadi.fmin(0,brake[1]))*self.wheel_radius/self.wheel_inertia
         
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            wheel_omega_dot_fl,
            wheel_omega_dot_fr,
            wheel_omega_dot_rl,
            wheel_omega_dot_rr
        )
        return dot_x
        

        
class BicycleDynamicsModelTwoWheelDrive(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 2
        self.nx = 9
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.wheel_radius = params['wheel_radius']
        self.wheel_inertia = params['wheel_inertia']
        self.mass = params['m']
        self.Iz = params['Iz']
        self.a = 1
        self.b = 0.1
        self.kc = 0.2
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,throttle,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        d = x[7]
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        wheel_omega = x[8]
        
        steer_dot = u[0]
        d_dot = u[1]
        #brake = u[2:]

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        
        speed_at_wheel = casadi.veccat(casadi.sqrt((vy+omega*self.lf)**2 + vx**2),                                       
                                       casadi.sqrt((vy-omega*self.lr)**2 + vx**2))
        
        #slipping ratio & angle
        #need work
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx+0.01) + steer,
                              casadi.atan2(omega*self.lr - vy,vx+0.01))        
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/(speed_at_wheel*casadi.cos(alpha)+10)        
        #lamb = casadi.DM.ones(2)*0.1
        #lamb = 0.1
        lamb = casadi.veccat(-1+ self.wheel_radius*wheel_omega/casadi.sqrt(casadi.fmax((vy+omega*self.lf)**2 + vx**2,0.001)),
                             -1+ self.wheel_radius*wheel_omega/casadi.sqrt(casadi.fmax((vy-omega*self.lr)**2 + vx**2,0.001)))
        lamb = casadi.fmax(lamb,-0.2)
        lamb = casadi.fmin(lamb,0.2)
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(2)*4*self.mass*9.81
        
        #Ff = self.front_tire_model.getForce(alpha[0],vx,d)
        #Fr = self.rear_tire_model.getForce(alpha[1],vx,d)
        Ff = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        #Ff[0] = (self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx  
        Fr = self.rear_tire_model.getForce(lamb[1],alpha[1],Fz[1])        
        #Fr[0] = 0
        
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot
        
        vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        v_wheel = wheel_omega*self.wheel_radius
        K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        
        wheel_omega_dot = (K - Ff[0]-Fr[0])*self.wheel_radius/self.wheel_inertia
         
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            d_dot,
            wheel_omega_dot
        )
        return dot_x
        
        
class BicycleDynamicsModelTwoWheelDriveWithBrake(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 4
        self.nx = 9
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.brake_torque = params['brake_torque']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.front_wheel_radius = params['front_wheel_radius']
        self.front_wheel_mass = params['front_wheel_mass']
        self.rear_wheel_radius = params['rear_wheel_radius']
        self.rear_wheel_mass = params['rear_wheel_mass']
        #self.wheel_inertia = params['wheel_inertia']
        self.front_wheel_inertia = self.front_wheel_radius*self.front_wheel_radius*self.front_wheel_mass
        self.rear_wheel_inertia = self.rear_wheel_radius*self.rear_wheel_radius*self.rear_wheel_mass
                
        self.mass = params['m']
        self.Iz = params['Iz']
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        #d = x[7]
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        front_wheel_omega = x[7]
        rear_wheel_omega = x[8]
        
        steer_dot = u[0]
        #d_dot = u[1]
        d = u[1]
        front_wheel_brake = u[2]
        rear_wheel_brake = u[3]        

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        
        #speed_at_wheel = casadi.veccat(casadi.sqrt((vy+omega*self.lf)**2 + vx**2),                                       
        #                               casadi.sqrt((vy-omega*self.lr)**2 + vx**2))
        
        #slipping ratio & angle
        #need work
        #alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx+0.01) + steer,
        #                      casadi.atan2(omega*self.lr - vy,vx+0.01))  
        
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf+vy, vx) + steer,
                              casadi.atan2(omega*self.lr-vy,vx)) 
        #alpha = casadi.veccat(-(omega*self.lf+vy)/(casadi.fmax(vx,0.1)) + steer,
        #                      (omega*self.lr-vy)/(casadi.fmax(vx,0.1)))             
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2)**0.5)
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2+0.00001)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2+0.00001)**0.5)

        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/(casadi.fmax(vx,0.1)),
        #                    -1+ self.wheel_radius*rear_wheel_omega/(casadi.fmax(vx,0.1)))
        #lamb = casadi.veccat(-1+ casadi.rdivide(self.wheel_radius*front_wheel_omega,vx),
        #                     -1+ casadi.rdivide(self.wheel_radius*rear_wheel_omega,vx))
        
        lamb = casadi.veccat(-1+ self.front_wheel_radius*front_wheel_omega/(vx*casadi.cos(steer)+vy*casadi.sin(steer)),
                             -1+ self.rear_wheel_radius*rear_wheel_omega/vx)

        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/vx,
        #                     -1+ self.wheel_radius*rear_wheel_omega/vx)
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(2)*self.mass*9.81/2
        
        #Ff = self.front_tire_model.getForce(alpha[0],vx,d)
        #Fr = self.rear_tire_model.getForce(alpha[1],vx,d)
        Ff = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        #Ff[0] = (self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx  
        Fr = self.rear_tire_model.getForce(lamb[1],alpha[1],Fz[1])        
        #Fr[0] = 0
        
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot
        
        vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        #v_wheel = (front_wheel_omega+rear_wheel_omega)/2*self.wheel_radius
        #K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        #K= self.Cm1*d
        K = self.Cm1 * d - self.Croll
        
        #front_wheel_omega_dot = (K -casadi.fabs(Ff[1])*self.wheel_radius*0.2 - Ff[0]*self.wheel_radius-self.brake_torque*front_wheel_brake)/self.wheel_inertia
        #rear_wheel_omega_dot = (K -casadi.fabs(Fr[1])*self.wheel_radius*0.2- Fr[0]*self.wheel_radius-self.brake_torque*rear_wheel_brake)/self.wheel_inertia
        front_wheel_omega_dot = (K - Ff[0]*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        rear_wheel_omega_dot = (K - Fr[0]*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia
        
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            #d_dot,
            front_wheel_omega_dot,
            rear_wheel_omega_dot
        )
        return dot_x

class BicycleDynamicsModelTwoWheelDriveWithBrakeNWH(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 4
        self.nx = 9
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.brake_torque = params['brake_torque']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.front_wheel_radius = params['front_wheel_radius']
        self.front_wheel_mass = params['front_wheel_mass']
        self.rear_wheel_radius = params['rear_wheel_radius']
        self.rear_wheel_mass = params['rear_wheel_mass']
        #self.wheel_inertia = params['wheel_inertia']
        self.front_wheel_inertia = self.front_wheel_radius*self.front_wheel_radius*self.front_wheel_mass
        self.rear_wheel_inertia = self.rear_wheel_radius*self.rear_wheel_radius*self.rear_wheel_mass
                
        self.mass = params['m']
        self.Iz = params['Iz']
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        #d = x[7]
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        front_wheel_omega = x[7]
        rear_wheel_omega = x[8]
        
        steer_dot = u[0]
        #d_dot = u[1]
        d = u[1]
        front_wheel_brake = u[2]
        rear_wheel_brake = u[3]     

        K = self.Cm1 * d - self.Croll   

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        
        #speed_at_wheel = casadi.veccat(casadi.sqrt((vy+omega*self.lf)**2 + vx**2),                                       
        #                               casadi.sqrt((vy-omega*self.lr)**2 + vx**2))
        
        #slipping ratio & angle
        #need work
        #alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx+0.01) + steer,
        #                      casadi.atan2(omega*self.lr - vy,vx+0.01))  
        
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf+vy, vx) + steer,
                              casadi.atan2(omega*self.lr-vy,vx)) 
        #alpha = casadi.veccat(-(omega*self.lf+vy)/(casadi.fmax(vx,0.1)) + steer,
        #                      (omega*self.lr-vy)/(casadi.fmax(vx,0.1)))             
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2)**0.5)
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2+0.00001)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2+0.00001)**0.5)

        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/(casadi.fmax(vx,0.1)),
        #                    -1+ self.wheel_radius*rear_wheel_omega/(casadi.fmax(vx,0.1)))
        #lamb = casadi.veccat(-1+ casadi.rdivide(self.wheel_radius*front_wheel_omega,vx),
        #                     -1+ casadi.rdivide(self.wheel_radius*rear_wheel_omega,vx))
        
        lamb = casadi.veccat(-1+ self.front_wheel_radius*front_wheel_omega/(vx*casadi.cos(steer)+vy*casadi.sin(steer)),
                             -1+ self.rear_wheel_radius*rear_wheel_omega/vx)

        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/vx,
        #                     -1+ self.wheel_radius*rear_wheel_omega/vx)
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(2)*self.mass*9.81/2
        
        #Ff = self.front_tire_model.getForce(alpha[0],vx,d)
        #Fr = self.rear_tire_model.getForce(alpha[1],vx,d)
        #Ff = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        #Ff[0] = (self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx  
        #Fr = self.rear_tire_model.getForce(lamb[1],alpha[1],Fz[1])        
        #Fr[0] = 0
        Ff_y = self.front_tire_model.getLateralForce(alpha[0],Fz[0])
        Fr_y = self.rear_tire_model.getLateralForce(alpha[1],Fz[1])
        
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        Ff_x_max = self.front_tire_model.getMaxLongitudinalForce(Fz[0])*self.front_wheel_radius
        Fr_x_max = self.rear_tire_model.getMaxLongitudinalForce(Fz[1])*self.rear_wheel_radius

        free_front_wheel_force = (K - self.brake_torque*front_wheel_brake)/self.front_wheel_radius
        free_rear_wheel_force = (K - self.brake_torque*rear_wheel_brake)/self.front_wheel_radius



        #free_front_friction_torque = casadi.fmin(free_front_wheel_torque,Ff_x_max)      
        #free_front_friction_torque = casadi.fmax(free_front_friction_torque,-Ff_x_max) 

        #free_front_friction_torque = casadi.sign(free_front_wheel_torque)*casadi.fmin(casadi.fabs(free_front_wheel_torque),Ff_x_max)

        #free_rear_friction_torque = casadi.fmin(free_rear_wheel_torque,Fr_x_max)      
        #free_rear_friction_torque = casadi.fmax(free_rear_wheel_torque,-Fr_x_max)  

        #Ff_x = casadi.if_else(casadi.fabs(lamb[0])<self.front_tire_model.getLambdaAtMaxForce(),free_front_wheel_torque/self.front_wheel_radius,self.front_tire_model.getLongitudinalForce(lamb[0],Fz[0]))
        #Fr_x = casadi.if_else(casadi.fabs(lamb[1])<self.rear_tire_model.getLambdaAtMaxForce(),free_rear_wheel_torque/self.rear_wheel_radius,self.rear_tire_model.getLongitudinalForce(lamb[1],Fz[1]))

        Ff_x = casadi.if_else(casadi.fabs(free_front_wheel_force)<Ff_x_max,free_front_wheel_force,self.front_tire_model.getLongitudinalForce(lamb[0],Fz[0]))
        Fr_x = casadi.if_else(casadi.fabs(free_rear_wheel_force)<Fr_x_max,free_rear_wheel_force,self.rear_tire_model.getLongitudinalForce(lamb[1],Fz[1]))


        #Ff_x = free_front_friction_torque/self.front_wheel_radius
        #Fr_x = free_rear_friction_torque/self.rear_wheel_radius



        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot

        vx_dot = 1/self.mass * (Fr_x + Ff_x*casadi.cos(steer) - Ff_y*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        vy_dot = 1/self.mass * (Fr_y + Ff_x*casadi.sin(steer) + Ff_y*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        omega_dot = 1/self.Iz * (Ff_y*self.lf*casadi.cos(steer) + Ff_x*self.lf*casadi.sin(steer) - Fr_y*self.lr) #omegadot
        
        #vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        #v_wheel = (front_wheel_omega+rear_wheel_omega)/2*self.wheel_radius
        #K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        #K= self.Cm1*d
        
        
        #front_wheel_omega_dot = (K -casadi.fabs(Ff[1])*self.wheel_radius*0.2 - Ff[0]*self.wheel_radius-self.brake_torque*front_wheel_brake)/self.wheel_inertia
        #rear_wheel_omega_dot = (K -casadi.fabs(Fr[1])*self.wheel_radius*0.2- Fr[0]*self.wheel_radius-self.brake_torque*rear_wheel_brake)/self.wheel_inertia
        #front_wheel_omega_dot = (K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        #rear_wheel_omega_dot = (K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia
        #front_wheel_omega_dot = (K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        #rear_wheel_omega_dot = (K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia

        #front_wheel_omega_dot = casadi.if_else(casadi.fabs(lamb[0])<self.front_tire_model.getLambdaAtMaxForce(),(vx_dot*casadi.cos(steer)+vy_dot*casadi.sin(steer))/self.front_wheel_radius,(K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia)
        #rear_wheel_omega_dot = casadi.if_else(casadi.fabs(lamb[1])<self.rear_tire_model.getLambdaAtMaxForce(),vx_dot/self.rear_wheel_radius,(K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia)

        front_wheel_omega_dot = casadi.if_else(casadi.fabs(free_front_wheel_force)<Ff_x_max,(vx_dot*casadi.cos(steer)+vy_dot*casadi.sin(steer))/self.front_wheel_radius,(K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia)
        rear_wheel_omega_dot = casadi.if_else(casadi.fabs(free_rear_wheel_force)<Fr_x_max,vx_dot/self.rear_wheel_radius,(K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia)

        
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            #d_dot,
            front_wheel_omega_dot,
            rear_wheel_omega_dot
        )
        return dot_x       

class BicycleDynamicsModelTwoWheelDriveWithBrakeNWHNoLong(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 4
        self.nx = 7
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.brake_torque = params['brake_torque']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.front_wheel_radius = params['front_wheel_radius']
        self.front_wheel_mass = params['front_wheel_mass']
        self.rear_wheel_radius = params['rear_wheel_radius']
        self.rear_wheel_mass = params['rear_wheel_mass']
        #self.wheel_inertia = params['wheel_inertia']
        self.front_wheel_inertia = self.front_wheel_radius*self.front_wheel_radius*self.front_wheel_mass
        self.rear_wheel_inertia = self.rear_wheel_radius*self.rear_wheel_radius*self.rear_wheel_mass
                
        self.mass = params['m']
        self.Iz = params['Iz']
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer]
        #u = [delta,d,front_brake,rear_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        #d = x[7]      
        
        
        steer_dot = u[0]
        #d_dot = u[1]
        d = u[1]
        front_wheel_brake = u[2]
        rear_wheel_brake = u[3]

        K = self.Cm1 * d - self.Croll   

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        

        
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf+vy, vx) + steer,
                              casadi.atan2(omega*self.lr-vy,vx))
        
        Fz = casadi.DM.ones(2)*self.mass*9.81/2
        
        Ff_y = self.front_tire_model.getLateralForce(alpha[0],Fz[0])
        Fr_y = self.rear_tire_model.getLateralForce(alpha[1],Fz[1])
        
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        Ff_x_max = self.front_tire_model.getMaxLongitudinalForce(Fz[0])*self.front_wheel_radius
        Fr_x_max = self.rear_tire_model.getMaxLongitudinalForce(Fz[1])*self.rear_wheel_radius

        Ff_x = (K - self.brake_torque*front_wheel_brake)/self.front_wheel_radius
        Fr_x = (K - self.brake_torque*rear_wheel_brake)/self.rear_wheel_radius


        #Ff_x = casadi.if_else(casadi.fabs(Ff_x)<Ff_x_max,Ff_x,casadi.sign(Ff_x)*Ff_x_max)
        #Fr_x = casadi.if_else(casadi.fabs(Fr_x)<Fr_x_max,Fr_x,casadi.sign(Fr_x)*Fr_x_max)

        #Ff_x = casadi.fmax(Ff_x,-Ff_x_max)
        #Ff_x = casadi.fmin(Ff_x,Ff_x_max)

        #Fr_x = casadi.fmax(Fr_x,-Fr_x_max)
        #Fr_x = casadi.fmin(Fr_x,Fr_x_max)

        #Ff_x = free_front_friction_torque/self.front_wheel_radius
        #Fr_x = free_rear_friction_torque/self.rear_wheel_radius



        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot

        
        vy_dot = 1/self.mass * (Fr_y + Ff_x*casadi.sin(steer) + Ff_y*casadi.cos(steer) - self.mass*vx*omega)  #vydot    
        vx_dot = 1/self.mass * (Fr_x + Ff_x*casadi.cos(steer) - Ff_y*casadi.sin(steer) + self.mass*vy*omega)-casadi.fabs(vy)*0.1  #vxdot    
        omega_dot = 1/self.Iz * (Ff_y*self.lf*casadi.cos(steer) + Ff_x*self.lf*casadi.sin(steer) - Fr_y*self.lr) #omegadot
        
        #vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        #v_wheel = (front_wheel_omega+rear_wheel_omega)/2*self.wheel_radius
        #K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        #K= self.Cm1*d
        
        
        #front_wheel_omega_dot = (K -casadi.fabs(Ff[1])*self.wheel_radius*0.2 - Ff[0]*self.wheel_radius-self.brake_torque*front_wheel_brake)/self.wheel_inertia
        #rear_wheel_omega_dot = (K -casadi.fabs(Fr[1])*self.wheel_radius*0.2- Fr[0]*self.wheel_radius-self.brake_torque*rear_wheel_brake)/self.wheel_inertia
        #front_wheel_omega_dot = (K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        #rear_wheel_omega_dot = (K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia
        #front_wheel_omega_dot = (K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        #rear_wheel_omega_dot = (K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia

        #front_wheel_omega_dot = casadi.if_else(casadi.fabs(lamb[0])<self.front_tire_model.getLambdaAtMaxForce(),(vx_dot*casadi.cos(steer)+vy_dot*casadi.sin(steer))/self.front_wheel_radius,(K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia)
        #rear_wheel_omega_dot = casadi.if_else(casadi.fabs(lamb[1])<self.rear_tire_model.getLambdaAtMaxForce(),vx_dot/self.rear_wheel_radius,(K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia)

        #front_wheel_omega_dot = casadi.if_else(casadi.fabs(free_front_wheel_force)<Ff_x_max,(vx_dot*casadi.cos(steer)+vy_dot*casadi.sin(steer))/self.front_wheel_radius,(K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia)
        #rear_wheel_omega_dot = casadi.if_else(casadi.fabs(free_rear_wheel_force)<Fr_x_max,vx_dot/self.rear_wheel_radius,(K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia)

        
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            #d_dot,
            #front_wheel_omega_dot,
            #rear_wheel_omega_dot
        )
        return dot_x    


class BicycleDynamicsModelTwoWheelDriveWithBrakeNWHFxInput(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 3
        self.nx = 7
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.brake_torque = params['brake_torque']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.front_wheel_radius = params['front_wheel_radius']
        self.front_wheel_mass = params['front_wheel_mass']
        self.rear_wheel_radius = params['rear_wheel_radius']
        self.rear_wheel_mass = params['rear_wheel_mass']
        #self.wheel_inertia = params['wheel_inertia']
        self.front_wheel_inertia = self.front_wheel_radius*self.front_wheel_radius*self.front_wheel_mass
        self.rear_wheel_inertia = self.rear_wheel_radius*self.rear_wheel_radius*self.rear_wheel_mass
                
        self.mass = params['m']
        self.Iz = params['Iz']
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer]
        #u = [delta,d,front_brake,rear_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        #d = x[7]      
        
        
        steer_dot = u[0]
        #d_dot = u[1]
        #d = u[1]
        #front_wheel_brake = u[2]
        #rear_wheel_brake = u[3]
        Ff_x = u[1]
        Fr_x = u[2]

        #K = self.Cm1 * d - self.Croll   

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        

        
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf+vy, vx) + steer,
                              casadi.atan2(omega*self.lr-vy,vx))
        
        Fz = casadi.DM.ones(2)*self.mass*9.81/2
        
        Ff_y = self.front_tire_model.getLateralForce(alpha[0],Fz[0])
        Fr_y = self.rear_tire_model.getLateralForce(alpha[1],Fz[1])
        
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        #Ff_x_max = self.front_tire_model.getMaxLongitudinalForce(Fz[0])*self.front_wheel_radius
        #Fr_x_max = self.rear_tire_model.getMaxLongitudinalForce(Fz[1])*self.rear_wheel_radius

        #Ff_x = (K - self.brake_torque*front_wheel_brake)/self.front_wheel_radius
        #Fr_x = (K - self.brake_torque*rear_wheel_brake)/self.rear_wheel_radius


        #Ff_x = casadi.if_else(casadi.fabs(Ff_x)<Ff_x_max,Ff_x,casadi.sign(Ff_x)*Ff_x_max)
        #Fr_x = casadi.if_else(casadi.fabs(Fr_x)<Fr_x_max,Fr_x,casadi.sign(Fr_x)*Fr_x_max)

        #Ff_x = casadi.fmax(Ff_x,-Ff_x_max)
        #Ff_x = casadi.fmin(Ff_x,Ff_x_max)

        #Fr_x = casadi.fmax(Fr_x,-Fr_x_max)
        #Fr_x = casadi.fmin(Fr_x,Fr_x_max)

        #Ff_x = free_front_friction_torque/self.front_wheel_radius
        #Fr_x = free_rear_friction_torque/self.rear_wheel_radius



        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot

        
        vy_dot = 1/self.mass * (Fr_y + Ff_x*casadi.sin(steer) + Ff_y*casadi.cos(steer) - self.mass*vx*omega)  #vydot    
        vx_dot = 1/self.mass * (Fr_x + Ff_x*casadi.cos(steer) - Ff_y*casadi.sin(steer) + self.mass*vy*omega)-casadi.fabs(vy)*0.1  #vxdot    
        omega_dot = 1/self.Iz * (Ff_y*self.lf*casadi.cos(steer) + Ff_x*self.lf*casadi.sin(steer) - Fr_y*self.lr) #omegadot
        
        #vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        #v_wheel = (front_wheel_omega+rear_wheel_omega)/2*self.wheel_radius
        #K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        #K= self.Cm1*d
        
        
        #front_wheel_omega_dot = (K -casadi.fabs(Ff[1])*self.wheel_radius*0.2 - Ff[0]*self.wheel_radius-self.brake_torque*front_wheel_brake)/self.wheel_inertia
        #rear_wheel_omega_dot = (K -casadi.fabs(Fr[1])*self.wheel_radius*0.2- Fr[0]*self.wheel_radius-self.brake_torque*rear_wheel_brake)/self.wheel_inertia
        #front_wheel_omega_dot = (K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        #rear_wheel_omega_dot = (K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia
        #front_wheel_omega_dot = (K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia
        #rear_wheel_omega_dot = (K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia

        #front_wheel_omega_dot = casadi.if_else(casadi.fabs(lamb[0])<self.front_tire_model.getLambdaAtMaxForce(),(vx_dot*casadi.cos(steer)+vy_dot*casadi.sin(steer))/self.front_wheel_radius,(K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia)
        #rear_wheel_omega_dot = casadi.if_else(casadi.fabs(lamb[1])<self.rear_tire_model.getLambdaAtMaxForce(),vx_dot/self.rear_wheel_radius,(K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia)

        #front_wheel_omega_dot = casadi.if_else(casadi.fabs(free_front_wheel_force)<Ff_x_max,(vx_dot*casadi.cos(steer)+vy_dot*casadi.sin(steer))/self.front_wheel_radius,(K - Ff_x*self.front_wheel_radius-self.brake_torque*front_wheel_brake)/self.front_wheel_inertia)
        #rear_wheel_omega_dot = casadi.if_else(casadi.fabs(free_rear_wheel_force)<Fr_x_max,vx_dot/self.rear_wheel_radius,(K - Fr_x*self.rear_wheel_radius-self.brake_torque*rear_wheel_brake)/self.rear_wheel_inertia)

        
        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot
            #d_dot,
            #front_wheel_omega_dot,
            #rear_wheel_omega_dot
        )
        return dot_x   

class BicycleDynamicsModelTwoWheelDriveWithBrakeUnity(DynamicsModel):
    def __init__(self, params,track,front_tire_model,rear_tire_model) -> None:        
        self.nu = 4
        self.nx = 9
        
        self.track = track
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.full_torque_over_all_wheels = params['full_torque_over_all_wheels']
        self.brake_torque = params['brake_torque']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.front_wheel_radius = params['front_wheel_radius']
        self.front_wheel_mass = params['front_wheel_mass']
        self.rear_wheel_radius = params['rear_wheel_radius']
        self.rear_wheel_mass = params['rear_wheel_mass']
        
        self.mass = params['m']
        self.Iz = params['inertia']
        
        self.downward_force = params['downward_force']
                
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        t = x[0]
        n = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        #d = x[7]
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        front_wheel_omega = x[7]
        rear_wheel_omega = x[8]
        
        steer_dot = u[0]
        #d_dot = u[1]
        d = u[1]
        front_wheel_brake = u[2]
        rear_wheel_brake = u[3]        

        kappa = self.track.f_kappa(t)
        phi_c = self.track.getPhiSym(t)
        tangent_vec = self.track.getTangentVec(t) 
        
        #speed_at_wheel = casadi.veccat(casadi.sqrt((vy+omega*self.lf)**2 + vx**2),                                       
        #                               casadi.sqrt((vy-omega*self.lr)**2 + vx**2))
        
        #slipping ratio & angle
        #need work
        #alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx+0.01) + steer,
        #                      casadi.atan2(omega*self.lr - vy,vx+0.01))  
        
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf+vy, vx) + steer,
                              casadi.atan2(omega*self.lr-vy,vx)) 
        #alpha = casadi.veccat(-(omega*self.lf+vy)/(casadi.fmax(vx,0.1)) + steer,
        #                      (omega*self.lr-vy)/(casadi.fmax(vx,0.1)))             
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2)**0.5)
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2+0.00001)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2+0.00001)**0.5)

        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/(casadi.fmax(vx,0.1)),
        #                    -1+ self.wheel_radius*rear_wheel_omega/(casadi.fmax(vx,0.1)))
        #lamb = casadi.veccat(-1+ casadi.rdivide(self.wheel_radius*front_wheel_omega,vx),
        #                     -1+ casadi.rdivide(self.wheel_radius*rear_wheel_omega,vx))
        
        lamb = casadi.veccat(-1+ self.front_wheel_radius*front_wheel_omega/(vx*casadi.cos(steer)+vy*casadi.sin(steer)),
                             -1+ self.rear_wheel_radius*rear_wheel_omega/vx)

        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/vx,
        #                     -1+ self.wheel_radius*rear_wheel_omega/vx)
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(2)*self.mass*9.81/2 + self.downward_force * vx /2
        
        #Ff = self.front_tire_model.getForce(alpha[0],vx,d)
        #Fr = self.rear_tire_model.getForce(alpha[1],vx,d)
        Ff = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        #Ff[0] = (self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx  
        Fr = self.rear_tire_model.getForce(lamb[1],alpha[1],Fz[1])        
        #Fr[0] = 0
        
        
        t_dot = (vx*casadi.cos(phi-phi_c)-vy*casadi.sin(phi-phi_c))/(casadi.norm_2(tangent_vec)*(1-n*kappa))
        n_dot = vx*casadi.sin(phi-phi_c)+vy*casadi.cos(phi-phi_c)        
        phi_dot = omega
        
        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot
        
        vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
     
        K = self.full_torque_over_all_wheels*d /2
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        #v_wheel = (front_wheel_omega+rear_wheel_omega)/2*self.wheel_radius
        #K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        #K= self.Cm1*d
        
        #front_wheel_omega_dot = (K -casadi.fabs(Ff[1])*self.wheel_radius*0.2 - Ff[0]*self.wheel_radius-self.brake_torque*front_wheel_brake)/self.wheel_inertia
        #rear_wheel_omega_dot = (K -casadi.fabs(Fr[1])*self.wheel_radius*0.2- Fr[0]*self.wheel_radius-self.brake_torque*rear_wheel_brake)/self.wheel_inertia
        #front_wheel_omega_dot = (K - Ff[0]*self.wheel_radius-self.brake_torque*front_wheel_brake)/self.wheel_inertia
        #rear_wheel_omega_dot = (K - Fr[0]*self.wheel_radius-self.brake_torque*rear_wheel_brake)/self.wheel_inertia
        front_wheel_omega_dot = Ff[0]/(2*casadi.pi*self.front_wheel_radius*self.front_wheel_mass) + K/(self.front_wheel_radius*self.front_wheel_mass)-self.brake_torque*front_wheel_brake*self.front_wheel_radius/self.front_wheel_mass
        rear_wheel_omega_dot = Fr[0]/(2*casadi.pi*self.rear_wheel_radius*self.rear_wheel_mass) + K/(self.rear_wheel_radius*self.rear_wheel_mass)-self.brake_torque*rear_wheel_brake*self.rear_wheel_radius/self.rear_wheel_mass

        dot_x = casadi.veccat(
            t_dot,
            n_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            #d_dot,
            front_wheel_omega_dot,
            rear_wheel_omega_dot
        )
        return dot_x

class BicycleDynamicsModelTwoWheelDriveWithBrakeXY(DynamicsModel):
    def __init__(self, params,front_tire_model,rear_tire_model) -> None:        
        self.nu = 4
        self.nx = 9
        
        
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.wheel_radius = params['wheel_radius']
        #self.wheel_inertia = params['wheel_inertia']
        
        key='wheel_radius'
        if key in params:
            self.wheel_inertia = params['wheel_inertia']
        else:
            self.wheel_inertia = params['wheel_mass']*self.wheel_radius*self.wheel_radius
        self.mass = params['m']
        self.Iz = params['Iz']
        self.a = 1
        self.b = 0.1
        self.kc = 0.2
        
        
        
    
    def update(self, x, u):
        #x = [t,n,phi,vx,vy,omega,steer,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d,front_left_brake,front_right_brake,rear_left_brake,rear_right_brake]
        px = x[0]
        py = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        #d = x[7]
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        front_wheel_omega = x[7]
        rear_wheel_omega = x[8]
        
        steer_dot = u[0]
        #d_dot = u[1]
        d = u[1]
        front_wheel_brake = u[2]
        rear_wheel_brake = u[3]        

                
        #speed_at_wheel = casadi.veccat(casadi.sqrt((vy+omega*self.lf)**2 + vx**2),                                       
        #                               casadi.sqrt((vy-omega*self.lr)**2 + vx**2))
        
        #slipping ratio & angle
        #need work
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx+1e-9) + steer,
                              casadi.atan2(omega*self.lr - vy,vx+1e-9))        
        
        lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/(casadi.fmax((vy+omega*self.lf)**2 + vx**2,1e-9))**0.5,
                             -1+ self.wheel_radius*rear_wheel_omega/(casadi.fmax((vy-omega*self.lr)**2 + vx**2,1e-9))**0.5)
        
        #lamb = casadi.veccat(-1+ self.wheel_radius*front_wheel_omega/((vy+omega*self.lf)**2 + vx**2)**0.5,
        #                     -1+ self.wheel_radius*rear_wheel_omega/((vy-omega*self.lr)**2 + vx**2)**0.5)
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(2)*self.mass*9.81/2
        
        #Ff = self.front_tire_model.getForce(alpha[0],vx,d)
        #Fr = self.rear_tire_model.getForce(alpha[1],vx,d)
        Ff = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        #Ff[0] = (self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx  
        Fr = self.rear_tire_model.getForce(lamb[1],alpha[1],Fz[1])        
        #Fr[0] = 0
        
        px_dot = vx*casadi.cos(phi)-vy*casadi.sin(phi)
        py_dot = vx*casadi.sin(phi)+vy*casadi.cos(phi)      
        #phi_dot = omega
        
        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot
        
        
        vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        v_wheel = (front_wheel_omega+rear_wheel_omega)/2*self.wheel_radius
        K = (self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel 
        
        front_wheel_omega_dot = (K - Ff[0]-front_wheel_brake)*self.wheel_radius/self.wheel_inertia
        rear_wheel_omega_dot = (K - Fr[0]-rear_wheel_brake)*self.wheel_radius/self.wheel_inertia
        
        dot_x = casadi.veccat(
            px_dot,
            py_dot,
            omega,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            #d_dot,
            front_wheel_omega_dot,
            rear_wheel_omega_dot
        )
        return dot_x                
        
class BicycleDynamicsModelTwoWheelDriveXY(DynamicsModel):
    def __init__(self, params,front_tire_model,rear_tire_model) -> None:        
        self.nu = 2
        self.nx = 9       
        
        self.front_tire_model = front_tire_model
        self.rear_tire_model = rear_tire_model

        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        
        self.lf = params['lf']
        self.lr = params['lr']
        self.wheel_radius = params['wheel_radius']
        key='wheel_radius'
        if key in params:
            self.wheel_inertia = params['wheel_inertia']
        else:
            self.wheel_inertia = params['wheel_mass']*self.wheel_radius*self.wheel_radius
        #self.wheel_inertia = params['wheel_inertia']
        self.mass = params['m']
        self.Iz = params['Iz']
        #self.a = 1
        #self.b = 0.1
        #self.kc = 0.2
        
        
        
    
    def update(self, x, u):
        #x = [px,py,phi,vx,vy,omega,steer,throttle,front_left_wheel_speed,front_right_wheel_speed,rear_left_wheel_speed,rear_right_wheel_speed]
        #u = [delta,d]
        px = x[0]
        py = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]
        steer = x[6]
        d = x[7]
        
        #wheel_omega = casadi.veccat(x[8]+x[9]/2,x[8]-x[9]/2,x[8]+x[10]/2,x[8]-x[10]/2)
        wheel_omega = x[8]
        
        steer_dot = u[0]
        d_dot = u[1]
                
        speed_at_wheel = casadi.veccat(casadi.sqrt((vy+omega*self.lf)**2 + vx**2),                                       
                                       casadi.sqrt((vy-omega*self.lr)**2 + vx**2))
        
        #slipping ratio & angle
        #need work
        alpha = casadi.veccat(-casadi.atan2(omega*self.lf + vy, vx+0.01) + steer,
                              casadi.atan2(omega*self.lr - vy,vx+0.01)) 
        alpha = casadi.fmax(alpha,-0.2)
        alpha = casadi.fmin(alpha,0.2)       
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/casadi.fmax(self.wheel_radius*wheel_omega,speed_at_wheel*casadi.cos(alpha))
        #lamb = (self.wheel_radius*wheel_omega-speed_at_wheel*casadi.cos(alpha))/(speed_at_wheel*casadi.cos(alpha)+10)        
        #lamb = casadi.DM.ones(2)*0.1
        #lamb = 0.1
        lamb = casadi.veccat(-1+ self.wheel_radius*wheel_omega/casadi.sqrt(casadi.fmax((vy+omega*self.lf)**2 + vx**2,0.001)),
                             -1+ self.wheel_radius*wheel_omega/casadi.sqrt(casadi.fmax((vy-omega*self.lr)**2 + vx**2,0.001)))
        #lamb = casadi.fmax(lamb,-0.3)
        #lamb = casadi.fmin(lamb,0.3)
        
        #Fz need work
        #Fz = casadi.veccat(0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81,0.25*self.mass*9.81)
        Fz = casadi.DM.ones(2)*4*self.mass*9.81
        
        #Ff = self.front_tire_model.getForce(alpha[0],vx,d)
        #Fr = self.rear_tire_model.getForce(alpha[1],vx,d)
        Ff = self.front_tire_model.getForce(lamb[0],alpha[0],Fz[0])
        #Ff[0] = (self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx  
        Fr = self.rear_tire_model.getForce(lamb[1],alpha[1],Fz[1])        
        #Fr[0] = 0
        
        
        px_dot = vx*casadi.cos(phi)-vy*casadi.sin(phi)
        py_dot = vx*casadi.sin(phi)+vy*casadi.cos(phi)        
        phi_dot = omega
        
        #vx_dot = 1/self.mass * (0 + K*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)   #vxdot
        #vy_dot = 1/self.mass * (Fr[1] + K*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        #omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + K*self.lf*casadi.sin(steer) - Fr[1]*self.lr)  #omegadot
        
        vx_dot = 1/self.mass * (Fr[0] + Ff[0]*casadi.cos(steer) - Ff[1]*casadi.sin(steer) + self.mass*vy*omega)  #vxdot
        vy_dot = 1/self.mass * (Fr[1] + Ff[0]*casadi.sin(steer) + Ff[1]*casadi.cos(steer) - self.mass*vx*omega)  #vydot        
        omega_dot = 1/self.Iz * (Ff[1]*self.lf*casadi.cos(steer) + Ff[0]*self.lf*casadi.sin(steer) - Fr[1]*self.lr) #omegadot
     
        #K = 0.25*self.kc*d/(self.a+self.b*wheel_omega)
        v_wheel = wheel_omega*self.wheel_radius
        K = casadi.fmax((self.Cm1-self.Cm2*v_wheel) * d - self.Croll -self.Cd*v_wheel*v_wheel,0)
        
        wheel_omega_dot = (K - Ff[0]-Fr[0])*self.wheel_radius/self.wheel_inertia
         
        dot_x = casadi.veccat(
            px_dot,
            py_dot,
            phi_dot,
            vx_dot,
            vy_dot,
            omega_dot,
            steer_dot,
            d_dot,
            wheel_omega_dot
        )
        return dot_x