from scipy.integrate import *
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
import os, sys


periSampl = 1000

class Parameters:
    mu0 = 4 * 3.1415927 * 1e-7
    gamma = 2.2128e5
    alpha = 1#0.01
    Js = 1 #1.4
    K1  = (0.00/mu0)*Js/2 # (0.062/mu0)*Js/2=227889.5 [J/m^3] # old:-Js**2/(2*mu0) # (185296)
    K12 = 0#-159/10#                            #   K1/1000#-7320.113 
    RAHE = 1 #0.1 #1#1
    RPHE = 0.01 #0.09#0.021193
    RAMR = 1
    RANE = 0.000#1#1##.085
    d = 1e-9 #(0.6+1.2+1.1) * 1e-9      
    frequency = 0.1e9
    currentd = float(sys.argv[1]) * 1e10
    hbar = 1.054571e-34
    e = 1.602176634e-19
    mu0 = 4 * 3.1415927 * 1e-7
    easy_axis  = np.array([0,0,1])
    easy_axis2 = np.array([1,0,0])
    p_axis = np.array([0,-1,0])
    etadamp = 0.1 #-0.75   
    etafield = -0.02#.01 #-0.8 #0.1   # etafield/etadamp=eta
    eta = etafield/etadamp
    hext = np.array([1.0 * K1/Js,0,0])
    area = (10e-6 * 7e-9)
    result = []         
    tEvol = []          #Time evolution of: Time
    mEvol = []          #                   Magnetization direction
    mxhEvol = []        #                   Fieldlike term
    mxmxhEvol = []      #                   Dampinglike term
    HsotEvol = []  #                   Magnitude of DT & FT
    DHEvol = [] #                   Current induced fields \Delta H
    
#-------------------FFT functions-------------------#
 
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

def reset_results(paramters_):
    paramters_.result = []         
    paramters_.tEvol = []          #Time evolution of: Time
    paramters_.mEvol = []          #                   Magnetization direction
    paramters_.mxhEvol = []        #                   Fieldlike term
    paramters_.mxmxhEvol = []      #                   Dampinglike term
    paramters_.DHEvol = [] #
    
#---------------------helper function to get fields at any moment---------#
    
def jacmini(t,m1,p):
 m=m1/np.linalg.norm(m1)
 j            = p.currentd * np.sin(2 * 3.1415927 * p.frequency * t)
 prefactorpol = j * p.hbar/(2 * p.e * p.Js * p.d)
 hani         = 2 * p.K1/p.Js * p.easy_axis * np.dot(p.easy_axis,m)
 hani2        = 2 * p.K12/p.Js * p.easy_axis2 * np.dot(p.easy_axis2,m)
 h            = p.hext+hani+hani2
 mxh          = np.cross(m, h+prefactorpol*( p.alpha*p.etadamp - p.etafield)*p.p_axis ) #Corrected from Dieter
 mxmxh        = np.cross(m, np.cross(m, h+prefactorpol*(-1/p.alpha*p.etadamp-p.etafield)*p.p_axis)  )
 return mxmxh

def fmini(t,m1,p):
 return(np.linalg.norm(jacmini(t,m1,p)))

def fac(t, m, p):
    m=m/np.linalg.norm(m)
    j            = p.currentd * np.sin(2 * 3.1415927 * p.frequency * t)
    prefactorpol = j * p.hbar/(2 * p.e * p.Js * p.d)
    hani         = 2 * p.K1/p.Js * p.easy_axis * np.dot(p.easy_axis,m)
    hani2        = 2 * p.K12/p.Js * p.easy_axis2 * np.dot(p.easy_axis2,m)
    h            = p.hext+hani+hani2
    H            = -prefactorpol * (p.etadamp*np.cross(p.p_axis,m) - p.etafield*p.p_axis)
    #heff=h-prefactorpol*(np.cross(m,p.p_axis)+p.eta*p.p_axis)
    heff=h-prefactorpol*(p.etadamp*np.cross(m,p.p_axis) + p.etafield*p.p_axis)
    mxh=np.cross(m, heff)
    mxmxh=np.cross(m, mxh)
    #mxh          = np.cross(m, h+prefactorpol*( p.alpha*p.etadamp - p.etafield)*p.p_axis )              #Corrected from Dieter
    #mxmxh        = np.cross(m, np.cross(m, h+prefactorpol*(-1/p.alpha*p.etadamp-p.etafield)*p.p_axis)  )     
    rhs=-p.gamma/(1+p.alpha**2)*mxh-p.gamma*p.alpha/(1+p.alpha**2)*mxmxh      
    p.tEvol.append(t)
    p.mEvol.append(m)
    p.mxhEvol.append(-p.gamma/(1+p.alpha**2)*mxh)
    p.mxmxhEvol.append(-p.gamma*p.alpha/(1+p.alpha**2)*mxmxh)
    p.DHEvol.append(H)
    return [rhs]

def vxx(t,v0,v1,v2):
    w = 2 * np.pi * 0.1e9
    return v0 + v1*np.sin(w*t) + v2*np.cos(2*w*t)


def calc_equilibrium(m0_,t0_,t1_,dt_,paramters_):
    t0 = t0_
    m0 = m0_
    dt = dt_
    r = ode(fac).set_integrator('vode', method='bdf',atol=1e-14,nsteps =500000)
    r.set_initial_value(m0_, t0_).set_f_params(paramters_).set_jac_params(2.0)
    t1 = t1_
    #Creating a counter and an array to store the magnetization directions
    magList = [[],[],[],[]] 
    old = [[],[],[],[],[],[],[]]        #old: t, mx, my, mz, mxh, mxmxh, rhs
    count = 0
    while r.successful() and r.t < t1: # and count < (periSampl + 1):  #OLD: XXX 
        #To make sure the steps are equally spaced 
        #Hayashi et al. (2014), after eqn 45, suggests to divide one period into
        # 200 time steps to get accurate temporal variation of Hall voltages
        mag=r.integrate(r.t+dt)
        magList[0].append(r.t)
        magList[1].append(mag[0])
        magList[2].append(mag[1])
        magList[3].append(mag[2])
        #old[5] = 0#np.amax(np.linalg.norm(paramters_.mxmxhEvol,axis=1))
        #if count%5000 == 0: print(len(paramters_.tEvol),len(paramters_.mxmxhEvol), old[5], count)
        #print(old[5])
        #if oldparamters_.tEvol[-1] < old[5]:
        #count+=1
    return np.array(magList)

def calc_w1andw2(m0_,t0_,t1_,dt_,paramters_): 
    def show_relaxation(mPlt,mdcPlt,DHPlt,mxhPlt,mxmxhPlt,rhsPlt):                                  #Plotting function
        ax=plt.axes()
        if mPlt == True:
            plt.plot(magList[0], magList[1] ,"C0-.",  linewidth=3, label = 'Mx')
            plt.plot(magList[0], magList[2] ,"C1-.",  linewidth=3, label = 'My')
            plt.plot(magList[0], magList[3] ,"C2-.",  linewidth=3, label = 'Mz')
        if mdcPlt == True:
            plt.plot(tdc, mdc[:,0], "C0", label = 'Mx j=0')
            plt.plot(tdc, mdc[:,1], "C1", label = 'My j=0')
            plt.plot(tdc, mdc[:,2], "C2", label = 'Mz j=0')
        if DHPlt == True:
            plt.plot(t, DH[:,0], "C0--", label = 'hx')
            plt.plot(t, DH[:,1], "C1--", label = 'hy')
            plt.plot(t, DH[:,2], "C2--", label = 'hz')
        if mxhPlt == True:
            plt.plot(t, mxh[:,0]/np.amax(np.abs(mxh[:,0])), "C0--", label = 'Mxhx')
            plt.plot(t, mxh[:,1]/np.amax(np.abs(mxh[:,1])), "C1--", label = 'Mxhy')
            plt.plot(t, mxh[:,2]/np.amax(np.abs(mxh[:,2])), "C2--", label = 'Mxhz')
        if mxmxhPlt == True:
            plt.plot(t, mxmxh[:,0]/np.amax(np.abs(mxmxh[:,0])), "C0-.", label = 'Mxhx')
            plt.plot(t, mxmxh[:,1]/np.amax(np.abs(mxmxh[:,1])), "C1-.", label = 'Mxhy')
            plt.plot(t, mxmxh[:,2]/np.amax(np.abs(mxmxh[:,2])), "C2-.", label = 'Mxhz')
        if rhsPlt == True:
            plt.plot(t, mxh[:,0]/np.amax(np.abs(mxh[:,0]))+mxmxh[:,0]/np.amax(np.abs(mxmxh[:,0])), "C0--", label = 'dm/dt')
            plt.plot(t, mxh[:,1]/np.amax(np.abs(mxh[:,1]))+mxmxh[:,1]/np.amax(np.abs(mxmxh[:,1])), "C1--", label = 'dm/dt')
            plt.plot(t, mxh[:,2]/np.amax(np.abs(mxh[:,2]))+mxmxh[:,2]/np.amax(np.abs(mxmxh[:,2])), "C2--", label = 'dm/dt')
        plt.plot(magList[0], current/orgdensity ,"C3--",  linewidth=3, label = 'Je')
        plt.plot(magList[0], current*magList[3]/orgdensity+magList[3] ,"C4--",  linewidth=3, label = 'Vxy(t)')
        ax.set(xlabel = r'$\mu_0 H_ext$ [T] ',ylabel = '')
        #(along z, tilted 5 deg. in x)
        plt.title("|H_ext| = " + str(round(paramters_.hext[2]*paramters_.mu0,4)) + "[T]" ) #M_i')#r'$V_{2w} [V]$ 
        plt.legend()
        plt.show()
#--------------------------------------------------FT from here-----------------------------------------------------------#        
    reset_results(paramters_)                                                             #Deleting previous results
    paramters_.currentd = orgdensity
    paramters_.currentd = 0                                                                #Solving eq. magnetization wo/AC
    magListdc = calc_equilibrium(m0_,t0_,t1_,dt_,paramters_)
    tdc                 = np.array(paramters_.tEvol)
    mdc                 = np.array(paramters_.mEvol)
    mxhdc               = np.array(paramters_.mxhEvol)
    mxmxhdc             = np.array(paramters_.mxmxhEvol)
    DHdc                = np.array(paramters_.DHEvol)
    paramters_.currentd = orgdensity                                                        #Returning to the original current
    #input("LLG wo/AC solved, press enter to continue")

    reset_results(paramters_)
    magList = calc_equilibrium(magListdc[1:,-1],t0_,t1_,dt_,paramters_)                                #Solving the LLG with AC current
    t                 = np.array(paramters_.tEvol)
    m                 = np.array(paramters_.mEvol)
    mxh               = np.array(paramters_.mxhEvol)
    mxmxh             = np.array(paramters_.mxmxhEvol)
    DH                = np.array(paramters_.DHEvol)

    time              = magList[0]                                                          #Equally spaced time vector
    sinwt             = np.sin(     2 * 3.1415927 * paramters_.frequency * time)            #Sinw finction to project into
    cos2wt            = np.cos( 2 * 2 * 3.1415927 * paramters_.frequency * time)            #Cos2w finction to project into
    current           = orgdensity * np.sin(2 * 3.1415927 * paramters_.frequency * time)    #AC current
    z=0                                                                                     #Time discretization
    dt=[]
    dt.append(time[1]-time[0])
    for i in time: 
        if z>0: dt.append(time[z]-time[z-1])
        z=z+1
    dt=np.array(dt)
    
    #Computing the voltage from R_{XY}
    voltage         = current*paramters_.area*(magList[3]*paramters_.RAHE 
                                               + magList[1]*magList[2]*paramters_.RPHE) + magList[3]*paramters_.RANE
    voltagexx       = current*paramters_.area*(magList[1]**2)*paramters_.RAMR 
    
    fitxy, cov = curve_fit(vxx, time, voltage)
    fit, cov = curve_fit(vxx, time, voltagexx)
    #print(fit)
    #ax=plt.axes()
    #plt.plot(time, vxx(time, fit[0], fit[1], fit[2]), "C1--", label = 'fit')
    #plt.plot(time, voltagexx, "C2--", label = 'vxx')
    #ax.set(xlabel = 'time',ylabel = 'Vxx')
    #plt.title("H_z = " + str(round(paramters_.hext[2]*paramters_.mu0,4)) + "[T]" ) #M_i')#r'$V_{2w} [V]$ 
    #plt.legend()
    #plt.show()
    
    #voltage         = voltage[periSampl*3:]
    #voltagexx       = voltagexx[periSampl*3:]
    #current         = current[periSampl*3:]
    #time            = time[periSampl*3:]
    #sinwt           = sinwt[periSampl*3:]
    #cos2wt          = cos2wt[periSampl*3:]
    #dt              = dt[periSampl*3:]

    R1w             = fitxy[1]
    R2w             = fitxy[2]
    #R1w             = np.sum(voltage * sinwt  * dt)*(2 / (time[-1]*(1/4)) )
    #R2w             = np.sum(voltage * cos2wt * dt)*(2 / (time[-1]*(1/4)) )
    #R1wxx           = np.sum(voltagexx * sinwt  * dt)*(2 / (time[-1]*(1/4)) )
    R1wxx           = fit[1]
    R2wxx           = fit[2]
    #R2wxx           = np.sum(voltagexx * cos2wt * dt)*(2 / (time[-1]*(1/4)) )
    #R1w            = np.dot( voltage * dt,sinwt  )/( np.dot(sinwt * dt,sinwt) * paramters_.currentd)
    #nR2w           = np.dot( voltage * dt,cos2wt )/( np.dot(cos2wt * dt, cos2wt) * paramters_.currentd)
    fR2w           = 0#fft( voltagexx, magList[0][periSampl*3:], paramters_.frequency)
    lR2w           = 0#lockin( voltagexx, magList[0][periSampl*3:], paramters_.frequency, 0)
    nR2w           = 0#lockin( voltagexx/paramters_.currentd, magList[0][periSampl*3:], paramters_.frequency, 90)


    show_relaxation(mPlt=True,mdcPlt=True,DHPlt=False,mxhPlt=False,mxmxhPlt=False,rhsPlt=False)

    return(R1w,R2w, mdc[-1,0], mdc[-1,1], mdc[-1,2], nR2w, lR2w, fR2w, R1wxx, R2wxx)

paramters = Parameters()
n = 31
phirange   = np.linspace(-np.pi/2,           np.pi*3/2,         num=n)
signalw  = []
signal2w  = []
signalwxx  = []
signal2wxx  = []
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
th = 5*np.pi/180            #External Field titled direction 
ph = 0
if longitudinalSweep:
    name = "Sweep" + sys.argv[2]
    fieldrange = np.linspace(-0.02/paramters.mu0,     0.02/paramters.mu0,    num = n )
                                 #np.linspace(0.08/paramters.mu0,     -0.08/paramters.mu0,    num = n )))
    for i in fieldrange:
        paramters.currentd = orgdensity
        paramters.hext = np.array([i,0,0])
        initm=np.array([0,0,-1])
        R1w,R2w,mx,my,mz, nR2w, lR2w, fR2w, R1wxx, R2wxx = calc_w1andw2(m0_=initm,t0_=0,t1_=5/paramters.frequency,dt_=1/(periSampl * paramters.frequency), paramters_=paramters)
        #Storing each current-induced field and magnetization state for each ext field value
        Hx.append(0)
        Hy.append(0)
        Hz.append(0)
        Mx.append(mx)
        My.append(my)
        Mz.append(mz)
        fieldrangeT.append(i * paramters.mu0)
        signalw.append(R1w)
        signal2w.append(R2w)
        signalwxx.append(R1wxx)
        signal2wxx.append(R2wxx)
        nsignal2w.append(nR2w)
        lsignal2w.append(lR2w)
        fsignal2w.append(fR2w)
        phirangeRad.append(0)
        print("Hext & |Hext| [T]:", paramters.hext*paramters.mu0, paramters.mu0*(paramters.hext[0]**2 
                                                                                 + paramters.hext[1]**2
                                                                                 + paramters.hext[2]**2)**0.5)

if rotationalSweep:
    name = "_HconsRotat" + sys.argv[2]
    fieldrange = [0/paramters.mu0] #np.linspace(0.045/paramters.mu0,               0.2/paramters.mu0,    num=3)#num= int((n-1)/10) )
    for h in fieldrange:
        ipMagnitude = h          # 0.05*paramters.mu0 # in Tesla
        for i in phirange:
            paramters.currentd = orgdensity
            paramters.hext = np.array([ np.cos(i) * ipMagnitude , np.sin(i) * ipMagnitude , 0]) 
            initm=[0,0,1]
            initm=np.array(initm)/np.linalg.norm(initm)
            #R1w,R2w,hx,hy,hz,mx,my,mz, Hs, nR2w = calc_w1andw2(m0_=initm,t0_=0,t1_=1/paramters.frequency,dt_=1/(periSampl * paramters.frequency), paramters_=paramters)
            #Storing each current-induced field and magnetization state for each ext field value
            R1w,R2w,mx,my,mz, nR2w, lR2w, fR2w, R1wxx, R2wxx = calc_w1andw2(m0_=initm,t0_=0,t1_=6/paramters.frequency,dt_=1/(periSampl * paramters.frequency), paramters_=paramters)
            #Storing each current-induced field and magnetization state for each ext field value
            Hx.append(0)
            Hy.append(0)
            Hz.append(0)
            Mx.append(mx)
            My.append(my)
            Mz.append(mz)
            fieldrangeT.append(h)
            signalw.append(R1w)
            signal2w.append(R2w)
            signalwxx.append(R1wxx)
            signal2wxx.append(R2wxx)
            nsignal2w.append(nR2w)
            lsignal2w.append(lR2w)
            fsignal2w.append(fR2w)
            phirangeRad.append(i*180/np.pi)
            #Live prompt
            print("Hext & |Hext| [T]:", paramters.hext*paramters.mu0, 'Phi:'+str(i*180/np.pi) , R2w)            
            #print( h, R1w, R2w, 'Pi:'+str(i%(2*np.pi)), '\tHk,Hd', round(Hs[0]), round(Hs[1]), mx, my, mz)

def showplot():
    #checking the 'equilibrium' magnetization directions
    #plt.plot(fieldrangeT, Mx,'b',label='m_x')
    #plt.plot(fieldrangeT, My,'g',label='m_y')
    #plt.plot(fieldrangeT, Mz,'r',label='m_z')
    #plt.plot(fieldrangeT, Hx,'b',label=r'$\Delta H_x$')
    #plt.plot(fieldrangeT, Hy,'g',label=r'$\Delta H_y$')
    #plt.plot(fieldrangeT, Hz,'r',label=r'$\Delta H_z$')
    plt.plot(phirange, signal2w)
    #plt.plot(fieldrangeT, signalw, label = 'Vw')
    #plt.plot(fieldrangeT, signal2w, label = 'V2w (Fourier integral)')
    #plt.plot(fieldrangeT, lsignal2w, label = 'V2w (Lock-in fx)')
    #plt.plot(fieldrangeT, fsignal2w, label = 'V2w (np.fft)')
    #plt.plot(fieldrangeT, signalwxx, label = 'Vwxx')
    #plt.plot(fieldrangeT, signal2wxx, label = 'V2wxx')
    #plt.plot(fieldrangeT, Mz,'r', label='m_z')
    #plt.plot(fieldrangeT, np.array(signal2w) - np.array(nsignal2w), label = 'diff r2w')
    #plt.plot(fieldrangeT, H,'r')
    ax=plt.axes()
    plt.savefig('signal.png' )
    #ax.set(xlabel=r'$\phi$ [grad]',ylabel = r'$m_{i}$ ') 
    ax.set(xlabel = r'$\mu_0 H_{ext(x)}$ (T)',ylabel = '')#r'$V_{2w} [V]$ ')
    plt.title("Current density " + str(sys.argv[1]) + "e10 [A/m2]" )
    plt.legend()
    plt.show()
    
def savedata(p, sig, fieldrangeT, name):
    #Storing the data into a dat file with the following strcture:
    #Delta denotes current-induced fields
    # ` denotes equilibium 
    # Current | H_ext | R2w |
    # \Delta H_x | \Delta H_y | \Delta H_z |
    # 7mx` | my` | mz` | Rw | 11 phi rad
    # 12 r1wxx r2wxx
    with open( str(name) + "_j" + str(p.currentd/1e10) + "e10.dat", "w") as f:
        i = 0
        for sig in signal2w:
            f.write( str(p.currentd) + "\t"  + str(fieldrangeT[i]) + "\t" + str(sig) + "\t" 
                    + str(Hx[i]) + "\t" + str(Hy[i]) + "\t" + str(Hz[i]) +'\t' 
                    + str(Mx[i]) + "\t" + str(My[i]) + "\t" + str(Mz[i]) + '\t' + str(signalw[i]) + "\t" 
                    + str(phirangeRad[i]) + "\t" + str(signalwxx[i]) + "\t" + str(signal2wxx[i])
                    + "\n")
            i += 1
        f.write("Hk1 \t Hk12 \t Hdamp " +
                "\t etaD \t etaF \t t \t freq \t Js " + 
                "\t Rahe \t Rphe \t Ramr \t cross_section\n")
        f.write( str(2 * p.K1/p.Js) + '\t' + str(2 * p.K12/p.Js) + '\t' 
                + str(p.etadamp * p.currentd * p.hbar/(2*p.e*p.Js*p.d)) + "\t" 
                + str(p.etadamp) + "\t" + str(p.etafield) + "\t" + str(p.d) + "\t"
                + str(p.frequency) + "\t" + str(p.Js) + "\t" 
                + str(p.RAHE) + "\t" + str(p.RPHE) + "\t" + str(p.RAMR) + "\t" + str(p.area)
                + '\n')
        f.close()

savedata(paramters, signal2w, fieldrangeT, "oop_Hx_mzp" + name)
#showplot()

os.system("cp " + sys.argv[0] + " "
          + "v2o_oop_Hx_mzp" + name 
          + "_j" + str(float(sys.argv[1])/1.0) + "e10_"
          + sys.argv[0])
