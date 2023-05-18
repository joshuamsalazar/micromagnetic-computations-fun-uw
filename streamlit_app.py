import streamlit as st
import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
from streamlit_app_page_functions import *
from scipy.optimize import curve_fit

header()

@st.cache_data
def text_to_vector(text):
    try:
        text = text.replace("(","")
        text = text.replace(")","")
        separated_text = np.array([float(s) for s in text.split(',')])
        if len(separated_text) != 3: separated_text = np.array([1,0,0])
    except:
        text = "1,0,0"
        separated_text = np.array([float(s) for s in text.split(',')])
        return separated_text
    else:
        separated_text=separated_text/np.linalg.norm(separated_text)
        return separated_text

with st.sidebar: #inputs
    customdir = st.text_input("Chose an external field sweep direction", "(1,0,0)")
    text_to_vector(customdir)
    hextdir = st.radio("Or any cartesian direction", ("x","y","z","custom"))
    form = st.form("Parameters")
    form.markdown("**Enter** your own custom values to run the model and **press** submit.")
    form.form_submit_button("Submit and run model.")
    alpha = float(form.text_input('Gilbert damping constant', 1))
    je = float(form.text_input('Current density j_e [10^10 A/m^2]', 10))
    K1 = float(form.text_input('Anisotropy constant K_1 [J/m^3]', 1.5 * 9100))
    Js = float(form.text_input('Saturation magnetization Js [T]', 0.65))
    RAHE = float(form.text_input('Anomalous Hall Effect coefficient', 0.65))
    d = float(form.text_input('FM layer thickness [nm]', (0.6+1.2+1.1) ))* 1e-9
    frequency = float(form.text_input('AC frequency [MHz]', 0.1e3))*1e6
    etadamp = float(form.text_input('Damping like torque term coefficient', 0.084))
    etafield = float(form.text_input('Field like torque term', 0.008))

timesteps = 200 #

Parameters = { #Convert to python class, but how to hash it? required for decorator @st.cache
    "gamma" : 2.2128e5,
    "alpha" : alpha,
    "K1" : K1,  
    "Js" : Js,
    "RAHE" : RAHE,
    "RPHE" : 0,
    "RAMR" : 0,
    "d" : d,    
    "frequency" : frequency,
    "currentd" : je * 1e10,
    "hbar" : 1.054571e-34,
    "e" : 1.602176634e-19,
    "mu0" : 4 * 3.1415927 * 1e-7,
    "easy_axis" : np.array([0,0,1]),
    "p_axis" : np.array([0,-1,0]),
    "etadamp"    : etadamp,    
    "etafield"   : etafield,               # etafield/etadamp=eta
    "hext" : np.array([0,0,0]),
    "omega" : 2 * np.pi * frequency,
    "area" : 1e-6 * 1e-6}

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

def calc_w1andw2(m0_,t0_,t1_,dt_,params): 
    params["currentd"] = 0
    timeRx, mRx = calc_equilibrium(m0_,t0_,t1_,dt_,params) #Computing equilibrium without current
    params["currentd"] = je * 1e10
    time, m = calc_equilibrium(mRx[-1],t0_,t1_,dt_,params)

    sinwt = np.sin( params["omega"] * time)
    ac = params["currentd"] * sinwt 

    #Computing the voltage from R_{AHE}
    voltage = ac * m[2] * params["RAHE"] * params["area"] # V_xy = R_{AHE} * J_e * m_z * A 
    voltagexx = ac * m[0]**2 * params["RAMR"]
    [R0,R1w,R2w], _ = curve_fit(fourier_model, time, voltage)
    #[R0xx,R1wxx,R2wxx], _ = curve_fit(fourier_model, time, voltagexx)
    return(R1w, R2w, time, *m, timeRx, *mRx)
    
paramters = Parameters
n = 21
phirange   = np.linspace(-np.pi/2, np.pi*3/2, num=n)
signalw  = []
signal2w  = []
timeEvol = []
timeEvolRx = []
Hx,Hy,Hz = [[],[],[]]
Mx,My,Mz = [[],[],[]]
m_eqx, m_eqy, m_eqz = [[],[],[]]
aheList, amrList, smrList = [[],[],[]]
fieldrangeT =[]
phirangeRad=[]

longitudinalSweep = True
rotationalSweep = False
hextamplitude = 0.1/paramters["mu0"]
fieldrange = np.linspace( -hextamplitude, hextamplitude, num = n )

@st.cache_data(persist=True)
def longSweep(t0_,t1_,dt_,params,hextdir):
    if longitudinalSweep:
        name = "_HSweep"
        for i in fieldrange:
            paramters["currentd"] = je * 1e10
            if hextdir == "x":
                paramters["hext"] = i*np.array([1,0,0])
            elif hextdir == "y":
                paramters["hext"] = i*np.array([0,1,0])
            elif hextdir == "z":
                paramters["hext"] = i*np.array([0,0,1])
            elif hextdir == "custom":
                customdir=text_to_vector(customdir)
                customdir/=np.linalg.norm(customdir)
                print("Custom direction: ", customdir)
                paramters["hext"] = i*customdir   #np.array([i[0],i[1],i[2]]) #Maybe Crashes the app!!! XXX 
            initm=[0,0,1]
            initm=np.array(initm)/np.linalg.norm(initm)
            R1w,R2w, t, mx,my,mz, tRx, hx,hy,hz = calc_w1andw2(m0_=initm,
                                                                            t0_=0,
                                                                            t1_=4/paramters["frequency"],
                                                                            dt_=1/(timesteps * paramters["frequency"]),
                                                                            params=paramters)
            #Storing each current-induced field and magnetization state for each ext field value
            timeEvol.append(t)
            timeEvolRx.append(tRx)
            Hx.append(hx)
            Hy.append(hy)
            Hz.append(hz)
            Mx.append(mx)
            My.append(my)
            Mz.append(mz)
            m_eqx.append(hx[-1])
            m_eqy.append(hy[-1])
            m_eqz.append(hz[-1])
            fieldrangeT.append(i * paramters["mu0"])
            signalw.append(R1w) 
            signal2w.append(R2w) 
            phirangeRad.append(0) #Get actual phi angle
            #AHE & AMR
            paramters["currentd"] = -paramters["currentd"]
            _, imagList  = calc_equilibrium(m0_=initm,t0_=0,t1_=4/paramters["frequency"],dt_=1/(timesteps * paramters["frequency"]), paramters_=paramters)
            
            aheList.append(mz[-1]-imagList[2][-1])
            amrList.append(mx[-1]*mx[-1])
            smrList.append(my[-1]*my[-1])
        
        #Live prompt
        #print(i, R1w, R2w, '\tHk,Hd', round(Hs[0]), round(Hs[1]), mx[-1], my[-1], mz[-1])
        return timeEvol, Hx,Hy,Hz, Mx,My,Mz, m_eqx, m_eqy, m_eqz, fieldrangeT, signalw, signal2w, aheList, amrList, smrList, timeEvolRx
signalw  = []
signal2w  = []
nsignal2w = []
lsignal2w = []
fsignal2w = []
timeEvol = []
Hx,Hy,Hz = [[],[],[]]
Mx,My,Mz = [[],[],[]]
m_eqx, m_eqy, m_eqz = [[],[],[]]
aheList, amrList = [[],[]]
fieldrangeT =[]
phirangeRad=[]

timeEvol, Hx,Hy,Hz, Mx,My,Mz, m_eqx, m_eqy, m_eqz, fieldrangeT, signalw, signal2w, aheList, amrList, smrList, timeEvolRx = longSweep(t0_=0,
                                                                            t1_=4/paramters["frequency"],
                                                                            dt_=1/(timesteps * paramters["frequency"]),
                                                                            params=paramters, hextdir=hextdir)


if rotationalSweep:
    name = "_HconsRotat"
    fieldrange = np.linspace(0,               0.8/paramters["mu0"],    num= int((n-1)/10) )
    for h in fieldrange:
        ipMagnitude = 0.05/paramters["mu0"]          # 0.05/paramters["mu0"] # in Tesla
        for i in phirange:
            paramters["currentd"] = je * 1e10
            paramters["hext"] = np.array([ np.cos(i) * ipMagnitude , np.sin(i) * ipMagnitude , h]) 
            initm=[0,0,-1]
            initm=np.array(initm)/np.linalg.norm(initm)
            R1w,R2w,hx,hy,hz,mx,my,mz, Hs, nR2w = calc_w1andw2(m0_=initm,t0_=0,t1_=1/paramters["frequency"],dt_=1/(timesteps * paramters["frequency"]), params=paramters)
            #Storing each current-induced field and magnetization state for each ext field value
            Hx.append(hx)
            Hy.append(hy)
            Hz.append(hz)
            Mx.append(mx)
            My.append(my)
            Mz.append(mz)
            phirangeRad.append(i*180/np.pi)
            fieldrangeT.append(h)
            signalw.append(R1w)
            signal2w.append(R2w)
            nsignal2w.append(nR2w)
            #Live prompt
            print( h, R1w, R2w, 'Pi:'+str(i%(2*np.pi)), '\tHk,Hd', round(Hs[0]), round(Hs[1]), mx, my, mz)
    
if st.checkbox("Show relaxation of magnetization", True):
    selected_field = st.select_slider('Slide the bar to check the trajectories for an specific field value [A/m]',
                    options = fieldrange.tolist())
    st.write("Field value equivalent to", str( round(selected_field*paramters["mu0"], 3) ), "[T]")

    s_index = fieldrange.tolist().index(selected_field)

    figtraj = graphm(timeEvol[s_index], Mx[s_index], My[s_index], Mz[s_index],
                      "time [ns]", r'$m_i$',  
                      "Evolution at " + str( round(selected_field*paramters["mu0"], 3) ) + "[T]")

    st.pyplot(figtraj) 

if st.checkbox("Show relaxation of magnetization DC", True):
    selected_fieldDC = st.select_slider('Slide the bar to check the DC trajectories for an specific field value [A/m]',
                    options = fieldrange.tolist())
    st.write("Field value equivalent to", str( round(selected_fieldDC*paramters["mu0"], 3) ), "[T]")

    s_index = fieldrange.tolist().index(selected_fieldDC)

    figtraj = graphm(timeEvolRx[s_index], Hx[s_index], Hy[s_index], Hz[s_index],
                      "time [ns]", r'$m_i$',  
                      "Evolution at " + str( round(selected_fieldDC*paramters["mu0"], 3) ) + "[T]")

    st.pyplot(figtraj) 

st.caption("Computing the harmonics")
print(np.array(Hx)[:,1].shape)
Hx=np.array(Hx)
Hy=np.array(Hy)
Hz=np.array(Hz)
figmag = graphm(fieldrangeT, Hx[:,-1], Hy[:,-1], Hz[:,-1], r'$\mu_0 H_x$ (T)', r'$m_i$',  "Equilibrium direction of m") #index denotes field sweep step

figv2w = graph(fieldrangeT, signal2w, r'$\mu_0 H_x$ (T)', r'$V_{2w} [V]$ ', "V2w", "Second harmonic voltage" )
figv1w = graph(fieldrangeT, signalw, r'$\mu_0 H_x$ (T)', r'$V_{w} [V]$ ', "Vw", "First harmonic voltage" )

figamr = graph(fieldrangeT, amrList, r'$\mu_0 H_x$ (T)', r'$m_x^2$', r'$m_x^2$','AMR effect')
figahe = graph(fieldrangeT, aheList, r'$\mu_0 H_x$ (T)', r'$m_{z,+j_e}-m_{z,-j_e}$', r'$m_{z,+j_e}-m_{z,ij_e}$','AHE effect')
figsmr = graph(fieldrangeT, smrList, r'$\mu_0 H_x$ (T)', r'$m_y^2$', r'$m_y^2$','SMR effect')

figmag = graphm(fieldrangeT, m_eqx, m_eqy, m_eqz, r'$\mu_0 H_x$ (T)', r'$m_i$',  "Equilibrium direction of m") #index denotes field sweep step
##plt.plot(fieldrangeT, lsignal2w, label = 'lock in r2w')
##plt.plot(fieldrangeT, fsignal2w, label = 'fft r2w')
##plt.plot(fieldrangeT, H,'r') 
##ax.set(xlabel=r'$\phi$ [grad]',ylabel = r'$m_{i}$ ') 

#st.pyplot(figv1w)
#st.pyplot(figv2w)

st.pyplot(figahe)
st.pyplot(figamr)
st.pyplot(figsmr)

st.write("It is important to highligh that by inducing an AC there is no an exact static point for equilibrium magnetization. However, when the system reaches equilibrium with respect to the AC current, the time averaged magnetization direction(check ref. [X] Phys. Rev. B 89, 144425 (2014)), which is equivalent to relaxing the system without current applied")

st.pyplot(figmag)

st.write(r"As can be noted in the magnetization dynamics for a given external field value, the system quickly gets its magnetization direction according to the applied AC current. However, if we just employ a single period for the time integration, the result of the Fourier integral may differ from the actual coefficient, as the first time steps do not have a pure wave behavior.")

st.write('If we just take in consideration the magnetization components to describe the AMR and AHE effects, the transfer curves are:')
st.write(r'Inside the simulation the voltage is computed as $V^{xy}(t)=J_x(t) m_z(t) R_{AHE} \sigma$, where $\sigma$ is the cross section area of the conducting element. In our case $\sigma=(2 \mu m \times 6 \text{nm})$ ')

st.write("Lastly, the resulting transfer curves using the Fourier series integral definition are: ")

st.write('The following page describes the details to consider to efficiently simulate a FM/HM interface. This model is based on the Landau-Lifshitz-Gilbert equation, and the equation is integrated using _scipy_ python libraries. Hence, the magnetization dynamics is computed  with this model, which also contains routines to calculate the first and second harmonics of the Anomalous Hall Voltage (from AH Effect). This interactve tool is designed to allow quick computations and detailed understanding of the considerations made to simulate such FM/HM interfaces. ')
st.write('The parameters used in the computation for the live plot results can be freely manipulated using the left sidebar (_available clicking in the arrowhead on the top left of this web app_). Feel free to perform computations with the desired values. ')

text_description()

#Pending code sections 
    #if st.checkbox("Show fields evolution", False):
    #    figfields = graphm(timeEvol[s_index], Hx[s_index], Hy[s_index], Hz[s_index],
    #                      "time [ns]", r'$m_i$',  
    #                      "Current induced fields at H_ext:" + str( round(selected_field*paramters["mu0"], 3) ) + "[T]")
    #
    #    st.pyplot(figfields)