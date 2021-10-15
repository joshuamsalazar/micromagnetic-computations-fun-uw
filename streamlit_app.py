import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from scipy.integrate import *
import scipy.optimize
import matplotlib.pyplot as plt
from functools import partial
import os, sys

st.sidebar.markdown("## Parameters used in the simulation")
st.sidebar.markdown("Enter your own custom values to run the model")

je = float(st.sidebar.text_input('Current density j_e [10^10 A/m^2]', 10))

periSampl = 1000 #

class Parameters:
    gamma = 2.2128e5
    alpha      = float(st.sidebar.text_input('Gilbert damping constant', 1))
    K1         = float(st.sidebar.text_input('Anisotropy constant K_1 [J/m^3]', 1.5 * 9100))   
    Js         = float(st.sidebar.text_input('Saturation magnetization Js [T]', 0.65))
    RAHE       = float(st.sidebar.text_input('Anomalous Hall effect coefficient', 0.65)) 
    d          = float(st.sidebar.text_input('FM layer thickness [nm]', (0.6+1.2+1.1) * 1e-9))       
    frequency  = float(st.sidebar.text_input('AC frequency [Hz]', 0.1e9)) 
    currentd   = je * 1e10
    hbar = 1.054571e-34
    e = 1.602176634e-19
    mu0 = 4 * 3.1415927 * 1e-7
    easy_axis = np.array([0,0,1])
    p_axis = np.array([0,-1,0])
    etadamp    = float(st.sidebar.text_input('Damping like torque term coefficient', 0.084))    
    etafield   = float(st.sidebar.text_input('Field like torque term', 0.008))               # etafield/etadamp=eta
    eta        = etafield/etadamp
    hext = np.array([1.0 * K1/Js,0,0])
    
def lockin(sig, t, f, ph):
    ref = np.cos(2 * 2*np.pi*f*t + ph/180.0*np.pi)
    #ref = np.sin(2*np.pi*f*t + ph/180.0*np.pi)
    comp = np.multiply(sig,ref)
    #print(t[-1]) #plot real part fft 
    return comp.mean()*2

def fft(sig, t, f):
    sample_dt = np.mean(np.diff(t))
    N = len(t)
    yfft = np.fft.rfft(sig)
    yfft_abs = np.abs(yfft) #!!!
    xfft = np.array(np.fft.rfftfreq(N, d=sample_dt))

    stride =max(int(2*f*0.1*sample_dt),2)
    idxF = np.argmin(np.abs(xfft-2*f))

    tmpmax = 0
    tmpj = 0
    for j in range(-stride, stride+1):
        if yfft_abs[idxF+j] > tmpmax:
            tmpmax = yfft_abs[idxF+j]
            tmpj = j

    idxF = idxF+tmpj
    return 2./N*(yfft.real[idxF])
    
def fields(t,m,p): 
    #Get the H^{DL} at (t, m, p)
    Hk = 2 * p.K1/p.Js
    Hd = p.etadamp * p.currentd * p.hbar/(2*p.e*p.Js*p.d)
    return (Hk, Hd)
    
def f(t, m, p):
    j            = p.currentd * np.cos(2 * 3.1415927 * p.frequency * t)
    prefactorpol = j * p.hbar/(2 * p.e * p.Js * p.d)
    hani         = 2 * p.K1/p.Js * p.easy_axis * np.dot(p.easy_axis,m)
    h            = p.hext+hani
    H            = - prefactorpol * (p.etadamp * np.cross(p.p_axis,m) + p.etafield * p.p_axis)
    mxh          = np.cross(     m,  h-prefactorpol*( p.etadamp * np.cross(p.p_axis,m) + p.etafield * p.p_axis )    ) #Corrected from Dieter
    mxmxh        = np.cross(     m,  mxh) 
    rhs          = - p.gamma/(1+p.alpha**2) * mxh-p.gamma * p.alpha/(1+p.alpha**2) * mxmxh  
    p.result.append([t,m[0],m[1],m[2],H[0],H[1],H[2]])
    return [rhs]
  
def calc_equilibrium(m0_,t0_,t1_,dt_,paramters_):
    t0 = t0_
    m0 = m0_
    dt = dt_
    r = ode(f).set_integrator('vode', method='bdf',atol=1e-14,nsteps =500000)
    r.set_initial_value(m0_, t0_).set_f_params(paramters_).set_jac_params(2.0)
    t1 = t1_
    #Creating a counter and an array to store the magnetization directions
    count = 0
    magList = [[],[],[],[]] 
    testSignal = []
    while r.successful() and r.t < t1: # and count < (periSampl + 1):  #OLD: XXX 
        #To make sure the steps are equally spaced 
        #Hayashi et al. (2014), after eqn 45, suggests to divide one period into
        # 200 time steps to get accurate temporal variation of Hall voltages
        mag=r.integrate(r.t+dt)
        magList[0].append(r.t)
        magList[1].append(mag[0])
        magList[2].append(mag[1])
        magList[3].append(mag[2])
        #testSignal.append(   23  * np.cos(2 * 2 * np.pi * paramters_.frequency * r.t) )
        #Computing the H^{DL} at each time step
        Hs = fields(r.t,mag,paramters_)
        count += 1
        #if count%100 == 0: print(count)
    magList = np.array(magList)
    #print(magList[0][0], magList[0][-1] )
    return(r.t,magList,Hs, testSignal)
  
def calc_w1andw2(m0_,t0_,t1_,dt_,paramters_): 
    paramters_.result = []
    t1,magList, Hs, testSignal    = calc_equilibrium(m0_,t0_,t1_,dt_,paramters_)
    npresults         = np.array(paramters_.result)
    time              = np.array( magList[0] )
    sinwt             = np.sin(     2 * 3.1415927 * paramters_.frequency * time)
    cos2wt            = np.cos( 2 * 2 * 3.1415927 * paramters_.frequency * time)
    current           = paramters_.currentd * np.cos(2 * 3.1415927 * paramters_.frequency * time)
    # time steps array creation
    z=0
    dt=[]
    dt.append(time[1]-time[0])
    for i in time: 
        if z>0:
            dt.append(time[z]-time[z-1])
        z=z+1
    dt=np.array(dt)
    #Computing the voltage from R_{AHE}
    voltage         = current * magList[3] * paramters_.RAHE  * (2e-6 * 6e-9)
    voltage         = voltage[periSampl:]
    current         = current[periSampl:]
    time            = time[periSampl:]
    sinwt           = sinwt[periSampl:]
    cos2wt          = cos2wt[periSampl:]
    dt              = dt[periSampl:]

    #nR2w            = np.sum(voltage/paramters_.currentd * cos2wt * dt)*(2/time[-1])
    R1w             = np.sum(voltage * sinwt  * dt)*(2 / (time[-1]*(3/4)) )
    R2w             = np.sum(voltage * cos2wt * dt)*(2 / (time[-1]*(3/4)) )
    #R2w             = np.sum(testSignal[periSampl:] * cos2wt * dt)*(2 / (time[-1]*(3/4)) )
    
    #R1w            = np.dot( voltage * dt,sinwt  )/( np.dot(sinwt * dt,sinwt) * paramters_.currentd)
    #nR2w            = np.dot( voltage * dt,cos2wt )/( np.dot(cos2wt * dt, cos2wt) * paramters_.currentd)
    
    fR2w           = fft( voltage, magList[0][periSampl:], paramters_.frequency)
    lR2w           = lockin( voltage, magList[0][periSampl:], paramters_.frequency, 0)
    
    #nR2w           = np.fft.fft(magList[3], 2)/2
    nR2w           = lockin( voltage/paramters_.currentd, magList[0][periSampl:], paramters_.frequency, 90)
    #Checking the magnetization time evolution at each external field value:
    
    #plt.plot(time, magList[1], label = 'mx')
    #plt.plot(time, magList[2], label = 'my')
    #plt.plot(time, magList[3][periSampl:], label = 'mz tree periods')
    #plt.plot(magList[0], magList[3], label = 'mz_full period')
    #plt.title("H_x = " + str(paramters_.hext[0]*paramters_.mu0) + "[T]" )
    #plt.legend()
    #plt.show()
    #plt.plot(time, mzlowfield(time, paramters_), label = 'test')
    #plt.plot(time, np.full(time.shape, sum(magList[1]) / len(magList[1]) ), label = 'mx')
    #plt.plot(time, np.full(time.shape, sum(magList[2]) / len(magList[2]) ), label = 'my')
    #plt.plot(time, np.full(time.shape, sum(magList[3]) / len(magList[3]) ), label = 'mz')
    #plt.plot(time, testSignal, label = 'cos(X)')
    #plt.plot(time, voltage, label = 'cos(X)')

    #Checking the current-induced fields time evolution at each external field value:
    
    #plt.plot(time, npresults[:,4], label = 'Hx')
    #plt.plot(time, npresults[:,5], label = 'Hy')
    #plt.plot(time, npresults[:,6], label = 'Hz')
    #plt.legend()
    #plt.show()
    
    #Final value of the current-induced field
    #H_eff = print(npresults[-1,4],npresults[-1,5],npresults[-1,6])
    #return(R1w,R2w,npresults[-1,4],npresults[-1,5],npresults[-1,6],npresults[-1,1],npresults[-1,2],npresults[-1,3], Hs, nR2w, lR2w, fR2w)
    return(R1w,R2w, 
           magList[0], # ZZZ re-write function to save memory (duplicated time array)
           npresults[:,4],npresults[:,5],npresults[:,6],
           magList[1], magList[2], magList[3],
           Hs, nR2w, lR2w, fR2w)
    
paramters = Parameters()
n = 21
phirange   = np.linspace(-np.pi/2,           np.pi*3/2,         num=n)
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
orgdensity = paramters.currentd

longitudinalSweep = True
rotationalSweep = False

if longitudinalSweep:
    name = "_HSweep"
    fieldrange = np.linspace(-0.1/paramters.mu0,     0.1/paramters.mu0,    num = n )
    for i in fieldrange:
        paramters.currentd = orgdensity
        paramters.hext = np.array([i,0,0])
        initm=[0,0,1]
        initm=np.array(initm)/np.linalg.norm(initm)
        R1w,R2w, t,hx,hy,hz, mx,my,mz, Hs, nR2w, lR2w, fR2w = calc_w1andw2(m0_=initm,
                                                                          t0_=0,
                                                                          t1_=4/paramters.frequency,
                                                                          dt_=1/(periSampl * paramters.frequency),
                                                                          paramters_=paramters)
        #Storing each current-induced field and magnetization state for each ext field value
        timeEvol.append(t)
        Hx.append(hx)
        Hy.append(hy)
        Hz.append(hz)
        Mx.append(mx)
        My.append(my)
        Mz.append(mz)
        m_eqx.append(mx[-1])
        m_eqy.append(my[-1])
        m_eqz.append(mz[-1])
        fieldrangeT.append(i * paramters.mu0)
        signalw.append(R1w)
        signal2w.append(R2w)
        nsignal2w.append(nR2w)
        lsignal2w.append(lR2w)
        fsignal2w.append(fR2w)
        phirangeRad.append(0)
        
        #AHE & AMR
        paramters.currentd = -paramters.currentd
        it1,imagList, iHs, itestSignal    = calc_equilibrium(m0_=initm,t0_=0,t1_=4/paramters.frequency,dt_=1/(periSampl * paramters.frequency), paramters_=paramters)
        
        aheList.append(mz[-1]-imagList[3][-1])
        amrList.append(mx[-1]*mx[-1])
        
        #Live prompt
        #print(i, R1w, R2w, '\tHk,Hd', round(Hs[0]), round(Hs[1]), mx[-1], my[-1], mz[-1])

if rotationalSweep:
    name = "_HconsRotat"
    fieldrange = np.linspace(0,               0.8/paramters.mu0,    num= int((n-1)/10) )
    for h in fieldrange:
        ipMagnitude = 0.05/paramters.mu0          # 0.05/paramters.mu0 # in Tesla
        for i in phirange:
            paramters.currentd = orgdensity
            paramters.hext = np.array([ np.cos(i) * ipMagnitude , np.sin(i) * ipMagnitude , h]) 
            initm=[0,0,-1]
            initm=np.array(initm)/np.linalg.norm(initm)
            R1w,R2w,hx,hy,hz,mx,my,mz, Hs, nR2w = calc_w1andw2(m0_=initm,t0_=0,t1_=1/paramters.frequency,dt_=1/(periSampl * paramters.frequency), paramters_=paramters)
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
    
def savedata(p, sig, fieldrangeT, name):
    #Storing the data into a dat file with the following strcture:
    #Delta denotes current-induced fields
    # ` denotes equilibium 
    # Current | H_ext | R2w | \Delta H_x | \Delta H_y | \Delta H_z | 7mz` | my` | mz` | Rw | 11 phi rad
    with open( "v2o_" + str(name) + "_j" + str(p.currentd/1e10) + "e10.dat", "w") as f:
        i = 0
        for sig in signal2w:
            f.write( str(p.currentd) + "\t"  + str(fieldrangeT[i]) + "\t" + str(sig) + "\t" 
                    + str(Hx[i]) + "\t" + str(Hy[i]) + "\t" + str(Hz[i]) +'\t' 
                    + str(Mx[i]) + "\t" + str(My[i]) + "\t" + str(Mz[i]) + '\t' + str(signalw[i]) + "\t" + str(phirangeRad[i])
                    + "\n")
            i += 1
        f.write("Hk\tHdamp\teta(f/d)\t t\t freq\n")
        f.write( str(Hs[0]) + '\t' + str(Hs[1]) + "\t" + str(p.etafield/p.etadamp) + "\t" + str(p.d) 
                + '\t' + str(p.frequency) + '\n')
        f.close()

def graph(x, y, xlab, ylab, pltlabel, plthead):
   fig, ax = plt.subplots()
   plt.plot(x, y, label = pltlabel)
   ax.set(xlabel = xlab, ylabel = ylab)
   plt.title(plthead)
   plt.legend()
   return fig

def graphm(t, mx, my, mz, xlab, ylab, plthead):
   fig, ax = plt.subplots()
   plt.plot(t, mx, label = r'$x$')
   plt.plot(t, my, label = r'$y$')
   plt.plot(t, mz, label = r'$z$')
   ax.set(xlabel = xlab, ylabel = ylab)
   plt.title(plthead)
   plt.legend()
   return fig

st.title('Magnetization dynamics for FM/HM interfaces, a single-spin model')
st.header('Online LLG integrator')
st.caption("Joshua Salazar, Sabri Koraltan, Harald Özelt, Dieter Suess")
st.caption("Physics of Functional Materials")
st.caption("University of Vienna")

st.write('The following page describes the details to consider to efficiently simulate a FM/HM interface. This model is based on the Landau-Lifshitz-Gilbert equation, and the equation is integrated using _scipy_ python libraries. Hence, the magnetization dynamics is computed  with this model, which also contains routines to calculate the first and second harmonics of the Anomalous Hall Voltage (from AH Effect). This interactve tool is designed to allow quick computations and detailed understanding of the considerations made to simulate such FM/HM interfaces. ')
st.write('The parameters used in the computation for the live plot results can be freely manipulated using the left sidebar (_available clicking in the arrowhead on the top left of this web app_). Feel free to perform computations with the desired values. ')

st.subheader('Theoretical description')

st.write('The system described by the model is a typical FM/HM interface. In our specific case, a Hall cross with a thin ferromagnetic layer displaying an out of plane magnetization (fig. 1).  ')
st.image("https://journals.aps.org/prb/article/10.1103/PhysRevB.89.144425/figures/1/medium",
        caption = "*Fig. 1* Hall bar structure. Adapted from Phys. Rev. B 89, 144425 (2014)",
        width   = 400 )
#($\eta_\text{DL}$ and $\eta_\text{FL}$)
st.write(r'The LLG equation employed in the model is in explicit form and takes the Slonczewsky spin-orbit-torque coefficients as input. It goes as follows:')
st.latex(r''' \frac{\partial \vec{m}}{\partial t} =
   \frac{\gamma}{1+\alpha^2} (\vec{m} \times \vec{H}_{\text{eff}}) - 
   \frac{\gamma \alpha}{1+\alpha^2} \:\vec{m} \times (\vec{m} \times \vec{H}_{\text{eff}})''')
st.write(r'Where $m$ represents the mgnetization unit vector, $\alpha$ the Gilbert damping constant, $\gamma$ the gyromagnetic ratio, and $\vec{H}_{\text{eff}}$ is the effective magnetic field. The effective magnetic field contains contributions of the applied external field, the effective anisotropy field, and the current induced fields via spin orbit torque effects. It reads as follows:')
st.latex(r''' \vec{ H }_{\text{eff}} =
\vec{ H }_{\text{ext}} + \vec{ H }_{\text{k}} + 
\vec{ H }^{\text{SOT}}_{\text{FL}} + 
\vec{ H }^{\text{SOT}}_{\text{DL}} \\ \:\\ \:\\
\vec{ H }_{\text{k}} = \frac{2\vec{K}_1}{Js}  \\ \:\\
\vec{ H }^{\text{SOT}}_{\text{FL}} = \eta_\text{FL} \frac{  j_e \hbar  }{ 2 e t \mu_0 M_s }\:\vec{m} \times (\vec{m} \times \vec{p}) \\ \:\\
\vec{ H }^{\text{SOT}}_{\text{DL}} = \eta_\text{DL} \frac{  j_e \hbar  }{ 2 e t \mu_0 M_s }\:(\vec{m} \times \vec{p})
''')


st.write(r"The $\vec{p}$ vector represents the spin polarization of electrons. For a current flowing along the x direction, the vector is $(0,-1,0)$. As the here simulated system presents out of plane magnetization along the +z axis, the $\vec{K}_1$ anisotropy constant is represented by $(0,0,K_1)$")
st.write("Therefore, this simplified model just describes out-of-plane systems with negligible Planar Hall Effect, compared to the Anomalous Hall Effect. It will get improved soon.")

st.caption("Performing the integration")

st.write("In order to accurately compute the first and second harmonic components of the Anomalous Hall Voltage, the period is, at least, split in 1000 equidistand time steps. This will ensure an accurate description of the time variation of the voltage induced by the AC current. Additionaly, it will improve the computation of the numerical Fourier integrals for getting the harmonic responses.")
st.write("Under AC, the voltage is made up by the following harmonics:")
st.latex(r''' V_{xy}(t) = V^{xy}_0 + V^{xy}_\omega\sin(\omega t) + V^{xy}_{2\omega}\cos(2\omega t) + ...''')
st.write("Those harmonic components can be isolated by applying the Fourier series coefficient integral definition, integrating over one full period.")
st.latex(r''' 
   V^{xy}_{\omega}=\frac{2}{T}\int_{T} V(t)\sin(\omega t)\text{dt} \\ \: \\
   V^{xy}_{2\omega}=\frac{2}{T}\int_{T} V(t)\cos(2\omega t)\text{dt} 
   ''')
st.write(r"As the system starts fully pointing in the z direction, it is important to simulate the electric current with a cosine wave $J_x=j_e \cos(\omega t)$. ")

if st.checkbox("Show relaxation of magnetization", True):
    selected_field = st.select_slider('Slide the bar to check the trajectories for an specific field value [A/m]',
                    options = fieldrange.tolist())
    st.write("Field value equivalent to", str( round(selected_field*paramters.mu0, 3) ), "[T]")

    s_index = fieldrange.tolist().index(selected_field)

    figtraj = graphm(timeEvol[s_index], Mx[s_index], My[s_index], Mz[s_index],
                      "time [ns]", r'$m_i$',  
                      "Evolution at " + str( round(selected_field*paramters.mu0, 3) ) + "[T]")

    st.pyplot(figtraj)

st.write(r"As can be noted in the magnetization dynamics for a given external field value, the system quickly gets its magnetization direction according to the applied AC current. However, if we just employ a single period for the time integration, the result of the Fourier integral may differ from the actual coefficient, as the first time steps do not have a pure wave behavior.") 

st.caption("Computing the harmonics")

st.write(r"Therefore, in order to accurately compute the integral, each time integration of the LLG equation, for each $H_{\text{ext,x}}$ value, is performed over 4 complete periods $t_f=4/f$. Then, for computing the Fourier integral, the initial period of the time integration of the LLG equation is ommited from the computation. Furthermore, to improve the accuracy of the calculated harmonic component of the voltage, the remaining three periods are integrated and the normalization factor of the Fourier integral is adjusted accordingly. Finally, the integral is numerically approximated by the following sum:")
st.latex(r''' 
V^{xy}_{ \omega} \approx \frac{2}{t_f(3/4)} \sum^{4000}_{i=1000} ({J_x}_i {m_z}_i R_{ \text{AHE} }) \sin(\omega t_i) (\Delta t)_i \\ \: \\
V^{xy}_{2\omega} \approx \frac{2}{t_f(3/4)} \sum^{4000}_{i=1000} ({J_x}_i {m_z}_i R_{ \text{AHE} }) \cos(2\omega t_i) (\Delta t)_i
''')
st.write(r'Where $i$ represents an index of the elements of the lists containing the values of each step of the simulation (_Note that one period has been split into 1000 equidistant steps_). Inside the simulation the voltage is computed as $V^{xy}(t)=J_x(t) m_z(t) R_{AHE} \sigma$, where $\sigma$ is the cross section area of the conducting element. In our case $\sigma=(2 \mu m \times 6 \text{nm})$ ')

st.write("Lastly, the resulting transfer curves using the Fourier series integral definition are: ")

figv2w = graph(fieldrangeT, signal2w, r'$\mu_0 H_x$ (T)', r'$V_{2w} [V]$ ', "V2w", "First harmonic voltage" )
figv1w = graph(fieldrangeT, signalw, r'$\mu_0 H_x$ (T)', r'$V_{w} [V]$ ', "V2w", "Second harmonic voltage" )

figamr = graph(fieldrangeT, amrList, r'$\mu_0 H_x$ (T)', r'$m_x^2$', r'$m_x^2$','AMR effect')
figahe = graph(fieldrangeT, aheList, r'$\mu_0 H_x$ (T)', r'$m_{z,+j_e}-m_{z,-j_e}$', r'$m_{z,+j_e}-m_{z,ij_e}$','AHE effect')

figmag = graphm(fieldrangeT, m_eqx, m_eqy, m_eqz, r'$\mu_0 H_x$ (T)', r'$m_i$',  "Equilibrium direction of m") #index denotes field sweep step
##plt.plot(fieldrangeT, lsignal2w, label = 'lock in r2w')
##plt.plot(fieldrangeT, fsignal2w, label = 'fft r2w')
##plt.plot(fieldrangeT, H,'r')
##ax.set(xlabel=r'$\phi$ [grad]',ylabel = r'$m_{i}$ ') 

st.pyplot(figv1w)
st.pyplot(figv2w)


st.write('If we just take in consideration the magnetization components to describe the AMR and AHE effects, the transfer curves are:')

st.pyplot(figahe)
st.pyplot(figamr)

st.write("It is important to highligh that by inducing an AC there is no an exact static point for equilibrium magnetization. However, when the system reaches equilibrium with respect to the AC current, the magnetization direction of the last time step of each period may be regarded as equilibrium magnetization (check ref. [X] Phys. Rev. B 89, 144425 (2014))")

st.pyplot(figmag)

#Pending code sections
    #if st.checkbox("Show fields evolution", False):
    #    figfields = graphm(timeEvol[s_index], Hx[s_index], Hy[s_index], Hz[s_index],
    #                      "time [ns]", r'$m_i$',  
    #                      "Current induced fields at H_ext:" + str( round(selected_field*paramters.mu0, 3) ) + "[T]")
    #
    #    st.pyplot(figfields)
