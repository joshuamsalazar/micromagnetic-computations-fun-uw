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
from bokeh.plotting import figure

je = st.selectbox("Current density A [10^01 A/m^2] ",
                     [1, 10, 100])
 
# print the selected hobby
st.write("Your current density is: ", je)

periSampl = 1000

class Parameters:
    gamma = 2.2128e5
    alpha = 1.0
    K1 = 1.5 * 9100   
    Js = 0.46
    RAHE = 0.65 
    d = (0.6+1.2+1.1) * 1e-9      
    frequency = 0.1e9
    currentd = je * 1e10
    hbar = 1.054571e-34
    e = 1.602176634e-19
    mu0 = 4 * 3.1415927 * 1e-7
    easy_axis = np.array([0,0,1])
    p_axis = np.array([0,-1,0])
    etadamp = 0.084    
    etafield = 0.008   # etafield/etadamp=eta
    eta = etafield/etadamp
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
    return(R1w,R2w,npresults[-1,4],npresults[-1,5],npresults[-1,6],npresults[-1,1],npresults[-1,2],npresults[-1,3], Hs, nR2w, lR2w, fR2w)

paramters = Parameters()
n = 101
phirange   = np.linspace(-np.pi/2,           np.pi*3/2,         num=n)
signalw  = []
signal2w  = []
nsignal2w = []
lsignal2w = []
fsignal2w = []
Hx,Hy,Hz = [[],[],[]]
Mx,My,Mz = [[],[],[]]
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
        R1w,R2w,hx,hy,hz,mx,my,mz, Hs, nR2w, lR2w, fR2w = calc_w1andw2(m0_=initm,t0_=0,t1_=4/paramters.frequency,dt_=1/(periSampl * paramters.frequency), paramters_=paramters)
        #Storing each current-induced field and magnetization state for each ext field value
        Hx.append(hx)
        Hy.append(hy)
        Hz.append(hz)
        Mx.append(mx)
        My.append(my)
        Mz.append(mz)
        fieldrangeT.append(i * paramters.mu0)
        signalw.append(R1w)
        signal2w.append(R2w)
        nsignal2w.append(nR2w)
        lsignal2w.append(lR2w)
        fsignal2w.append(fR2w)
        phirangeRad.append(0)
        #Live prompt
        print(i, R1w, R2w, '\tHk,Hd', round(Hs[0]), round(Hs[1]), mx, my, mz)

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

def showplot():
    #checking the 'equilibrium' magnetization directions
    #plt.plot(fieldrangeT, Mx,'b',label='m_x')
    #plt.plot(fieldrangeT, My,'g',label='m_y')
    
    #plt.plot(fieldrangeT, signalw, label = 'Vw')
    plt.plot(fieldrangeT, signal2w, label = 'V2w')
    #plt.plot(fieldrangeT, lsignal2w, label = 'lock in r2w')
    #plt.plot(fieldrangeT, fsignal2w, label = 'fft r2w')
    #plt.plot(fieldrangeT, Mz,'r', label='m_z')
    #plt.plot(fieldrangeT, np.array(signal2w) - np.array(nsignal2w), label = 'diff r2w')
    #plt.plot(fieldrangeT, H,'r')
    ax=plt.axes()
    plt.savefig('signal.png' )
    #ax.set(xlabel=r'$\phi$ [grad]',ylabel = r'$m_{i}$ ') 
    ax.set(xlabel = r'$\mu_0 H_x$ (T)',ylabel = r'$V_{2w} [V]$ ')
    plt.title("Current density " + str(sys.argv[1]) + "e10 [A/m2]" )
    plt.legend()
    plt.show()
    
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

def stplot(x, y, xlab, ylab, head, name):
    p = figure(
      title=head,
      x_axis_label=xlab,
      y_axis_label=ylab)
    
    p.line(x, y, legend_label = name, line_width=2)
    plot = st.bokeh_chart(p, use_container_width=True)
    return plot
  
  stplot(fieldrangeT, signal2w, "H_ext", "V2w", "Harmonics", "line")
    
