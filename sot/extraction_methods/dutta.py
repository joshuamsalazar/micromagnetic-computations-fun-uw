import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#Constants and parameters
mu0 = 4*3.1415927*1e-7
hbar = 1.054571e-34
e = 1.602176634e-19
t_fm = 1.e-9
area = 7e-14 # M1 (6+1nm x 10\mu m)

# inputs 4sim
datafile="phi_v2w.dat"
je=1e10
Rahe = 1 #.083#0.174082
Rphe = 0.01 #.03#0.021193
Js = 1
H_ext=0.045/mu0
H_k = 0.0/mu0

I = je*area
Vahe = Rahe/I
Vphe = Rphe/I
pf=hbar/(2 * e * Js * t_fm )

#Fitting functions
def V_2w_model(phi, C_a, C_p, v_0):
    return C_a*np.cos(phi) + C_p*np.cos(phi)*np.cos(2.*phi) + v_0

def fit_C(phis, v2ws, H_ext): #   - Fit CA
    C_a, C_p, offset = curve_fit(V_2w_model, phis, v2ws)[0]
    hdlz=C_a*(-2*(H_ext-H_k))/Vahe 
    eta_dl=hdlz/(je*pf)
    hfly=C_p*(-H_ext)/Vphe       
    eta_fl=hfly/(je*pf)
    print("\t\tHext = ", 1000*H_ext*mu0, " [mT]")
    print("hdlz = %1.1e  \t eta_dl = %.5f \t C_a = %1.1e"%(hdlz, eta_dl/mu0, C_a))
    print("hfly = %1.1e  \t eta_fl = %.5f \t C_p = %1.1e"%(hfly, eta_fl/mu0, C_p))
    return C_a, C_p

Cs = []
fieldsCa, fieldsCp =  [[],[]]
for mT in np.linspace(30,80,6):#[10,30,50,70,90]:
    data = pd.read_csv("oop_Hx_mzp_HconsRotat"+str(int(mT))+"mT_j1.0e10.dat", sep="\t")
    H_ext = (0.001*mT)/mu0
    phis = np.array(data.iloc[:-3,10], dtype="float32")*np.pi/180
    v2ws = np.array(data.iloc[:-3,2], dtype="float32") 
    Ca, Cp = [fit_C(phis, v2ws, H_ext)[0], fit_C(phis, v2ws, H_ext)[1]]
    Cs.append( [Ca,Cp] )
    fieldsCa.append( 1/((H_ext-H_k)*mu0) )
    fieldsCp.append( 1/(H_ext*mu0) )
    #plt.plot(phis, v2ws, "C0.")
    #plt.plot(phis, Ca*np.cos(phis), 'C1.-' , label = "Ca" )
    #plt.plot(phis, Cp*np.cos(phis)*np.cos(2.*phis), 'C2.-' ,label = "Cp" )
    #plt.title("H_ext = "+ str(mT)+ "mT")
    #plt.legend()
    #plt.show()
    
fieldsCa, fieldsCp, Cs = (np.array(fieldsCa), np.array(fieldsCp), np.array(Cs))
Cas, Cps = [Cs[:,0], Cs[:,1]]
plt.plot(fieldsCa, Cas, "C0--", label = "C_a")
plt.plot(fieldsCp, Cps, "C1.", label = "C_p")
plt.xlabel( r'$1/H_{ext(x)}$ [1/T]') 
plt.ylabel('V_2w [V]')
plt.legend()
plt.show()

def f_linear(invHext, c, b):
    return c*invHext + b #fit to C_a*(1/Hext) + b 

slopeA, cov0 = curve_fit(f_linear, fieldsCa, Cas)[0]
slopeP, cov0 = curve_fit(f_linear, fieldsCp, Cps)[0]
print("cd    : %.5f"%(-slopeA*2/(Vahe*je*pf*mu0)/0.39027 ))
print("cf    : %.5f"%(-slopeP/(Vphe*je*pf*mu0) /0.18437))
print("cd    : %.5f"%(-slopeA*2/(Vahe*je*pf*mu0)))
print("cf    : %.5f"%(-slopeP/(Vphe*je*pf*mu0) ))
print("eta_DL: %.5f"%(-slopeA*2/(Vahe*je*pf*mu0) *2*4/5*100))
print("eta_FL: %.5f"%(-slopeP/(Vphe*je*pf*mu0) *5/9*100))
