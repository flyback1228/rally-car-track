import casadi

class SimpleTire:
    def __init__(self,params) -> None:
        #pajecka and motor coefficients
        self.k_long = params['k_long']
        self.k_lat = params['k_lat']       

    def getForce(self,lamb,alpha,Fz):
        return casadi.veccat(0.5*Fz*self.k_long*lamb,0.5*Fz*self.k_lat*alpha)

class SimpleElectricalDrivenTire:
    def __init__(self,params,drive_wheel) -> None:
        #pajecka and motor coefficients
        self.B = params['B']       
        self.C = params['C']
        self.Cm1 = params['Cm1']
        self.Cm2 = params['Cm2']
        self.Croll = params['Croll']
        self.Cd = params['Cd']
        self.D = params['D']
        self.drive_wheel = drive_wheel

    def getForce(self,alpha,vx,d):
        if(self.drive_wheel):
            return casadi.vertcat((self.Cm1-self.Cm2*vx) * d - self.Croll -self.Cd*vx*vx,
                                  self.D*casadi.sin(self.C*casadi.atan(self.B*alpha)))
        else:
            return casadi.vertcat(0,self.D*casadi.sin(self.C*casadi.atan(self.B*alpha)))

class SimplePacTireMode:
    def __init__(self,params)->None:
        self.B_long = params['B_long']       
        self.C_long = params['C_long']
        self.D_long = params['D_long']
        self.B_lat = params['B_lat']       
        self.C_lat = params['C_lat']
        self.D_lat = params['D_lat']

        self.lambda_m = casadi.tan(casadi.pi/(2*self.C_long))/self.B_long
        self.alpha_m = casadi.tan(casadi.pi/(2*self.C_lat))/self.B_lat
        pass

    def getForce(self,lamb,alpha,Fz):
        return casadi.veccat(Fz*self.D_long*casadi.sin(self.C_long*casadi.atan(self.B_long*lamb)),
                                Fz*self.D_lat*casadi.sin(self.C_lat*casadi.atan(self.B_lat*alpha)))

    def getLongitudinalForce(self,lamb,Fz):
        return Fz*self.D_long*casadi.sin(self.C_long*casadi.atan(self.B_long*lamb))
    
    def getLateralForce(self,alpha,Fz):
        return Fz*self.D_lat*casadi.sin(self.C_lat*casadi.atan(self.B_lat*alpha))

    def getMaxLongitudinalForce(self,Fz):
        return Fz*self.D_long

    def getMaxLateralForce(self,Fz):
        return Fz*self.D_lat

    def getAlphaAtMaxForce(self):
        return self.alpha_m

    def getLambdaAtMaxForce(self):
        return self.lambda_m

    #def getForce(self,lamb,alpha,Fz):
    #    return casadi.veccat(Fz*0.4*lamb,Fz*0.15*alpha)

    #def getForce(self,lamb,alpha,Fz):
    #    return self.D_lat*casadi.sin(self.C_lat*casadi.atan(self.B_lat*alpha))