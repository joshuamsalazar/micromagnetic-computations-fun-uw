import numpy as np
from scipy.integrate import ode
from scipy.optimize import curve_fit

Parameters = { #Convert to python class, but how to hash it? required for decorator @st.cache
    "gamma" : 2.2128e5,
    "alpha" : 1,
    "K1" : 12350,  
    "Js" : 1.5,
    "RAHE" : 0.17,
    "RPHE" : 0,
    "RAMR" : 0,
    "d" : 1e-9,    
    "frequency" : 0.1e9,
    "currentd" : 10 * 1e10,
    "hbar" : 1.054571e-34,
    "e" : 1.602176634e-19,
    "mu0" : 4 * 3.1415927 * 1e-7,
    "easy_axis" : np.array([0,0,1]),
    "p_axis" : np.array([0,-1,0]),
    "etadamp"    : 0.1,    
    "etafield"   : 0.1,               # etafield/etadamp=eta
    "hext" : np.array([0,0,0]),
    "omega" : 2 * np.pi * 0.1e9,
    "area" : 10e-6 * 7e-9}
je = 10

def f(t, m, p):
    j            = p["currentd"] * np.sin(2 * np.pi * p["frequency"] * t)
    prefactorpol = j * p["hbar"]/(2 * p["e"] * p["Js"] * p["d"])
    hani         = 2 * p["K1"]/p["Js"] * p["easy_axis"] * np.dot(p["easy_axis"],m)
    h            = p["hext"]+hani
    H            = - prefactorpol * (p["etadamp"] * np.cross(p["p_axis"],m) + p["etafield"] * p["p_axis"])
    mxh          = np.cross( m,  h-prefactorpol*( p["etadamp"] * np.cross(p["p_axis"],m) + p["etafield"] * p["p_axis"] )    ) #Corrected from Dieter
    mxmxh        = np.cross( m,  mxh) 
    rhs          = - p["gamma"]/(1+p["alpha"]**2) * mxh-p["gamma"] * p["alpha"]/(1+p["alpha"]**2) * mxmxh 
    return [rhs]

def fourier_model(t,v0,v1,v2):
    w = 2 * np.pi * 0.1e9 # PENDING: frequency input
    return v0 + v1*np.sin(w*t) + v2*np.cos(2*w*t)

def calc_equilibrium(m0_,t0_,t1_,dt_,paramters_):
    dt = dt_
    t1 = t1_
    magList = [[],[],[],[]] 
    r = ode(f).set_integrator('vode', method='bdf',atol=1e-14,nsteps =500000)
    r.set_initial_value(m0_, t0_).set_f_params(paramters_).set_jac_params(2.0)
    while r.successful() and r.t < t1:
        #To make sure the steps are equally spaced 
        #Hayashi et al. (2014), after eqn 45, suggests to divide one period into
        # 200 time steps to get accurate temporal variation of Hall voltages
        mag=r.integrate(r.t+dt)
        magList[0].append(r.t)
        magList[1].append(mag[0])
        magList[2].append(mag[1])
        magList[3].append(mag[2])
    return np.array(magList[0]), np.array(magList[1:])

def calc_w1andw2(m0_,t0_,t1_,dt_,params,customdir): 
    params["currentd"] = 0 #No current for initial relaxation
    timeRx, mRx = calc_equilibrium(m0_,t0_,t1_,dt_,params) #Computing equilibrium without current
    lastMRx = (mRx[0][-1], mRx[1][-1], mRx[2][-1])  #Last value of the equilibrium

    params["currentd"] = je * 1e10  #Turning on the current again
    time, m = calc_equilibrium(lastMRx,t0_,t1_,dt_,params)
    sinwt = np.sin( params["omega"] * time) 
    ac = params["currentd"] * sinwt #AC current

    #Vxy voltage: Following the convention of Dutta: PRB 103, 184416 (2021) V_xy = R_{AHE} * I * m_z  + 2 * R_{PHE} * I * m_x * m_y
    voltage = ac * params["area"] * (m[2] * params["RAHE"] +  2 * m[0]* m[1] * params["RPHE"] )  # V_xy = R_{AHE} * J_e * m_z * A 
    voltagexx = ac * m[0]**2 * params["RAMR"]
    [R0,R1w,R2w], _ = curve_fit(fourier_model, time, voltage)
    #[R0xx,R1wxx,R2wxx], _ = curve_fit(fourier_model, time, voltagexx)
    return(R1w, R2w, time, *m, timeRx, *mRx)