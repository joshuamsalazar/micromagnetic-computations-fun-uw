from scipy.integrate import ode
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os, sys

class Parameters:
 gamma = 2.2128e5
 alpha = 1.0
 K1 = 1.5*9100   
 Js = 0.46
 RAHE = 0.65 
 d = (0.6+1.2+1.1)*1e-9      
 frequency = 0.1e9
 currentd = float(sys.argv[1])*1e10
 hbar = 1.054571e-34
 e = 1.602176634e-19
 mu0 = 4*3.1415927*1e-7
 easy_axis = np.array([0,0,1])
 p_axis = np.array([0,-1,0])
 etadamp = 0.084    # is equal to spin Hall angle
 etafield = 0.008   # etafield=eta*etadamp  
 hext = np.array([1.0*K1/Js,0,0])

def jacmini(p,m1):
 m=m1/np.linalg.norm(m1)
 prefactorpol = p.currentd*p.hbar/(2*p.e*p.Js*p.d)   #Zhu, Daoqian, and Weisheng Zhao. "Threshold Current Density for Perpendicular Magnetization Switching Through Spin-Orbit Torque." Physical Review Applied 13.4 (2020): 044078.
 hani=2*p.K1/p.Js*p.easy_axis*np.dot(p.easy_axis,m)
 h=p.hext+hani
 mxh = np.cross(m, h-prefactorpol*(p.etadamp*np.cross(m,p.p_axis)+p.etafield*p.p_axis))
 mxmxh = np.cross(m, mxh)
 return(mxmxh)
 
def fields(t,m,p):
    #Hk = 2*p.K1/p.Js 
    #Hk12 = 2*p.K12/p.Js
    Hk = 2*p.K1/p.Js                                        #p.easy_axis*np.dot(p.easy_axis,m)     #, np.dot(p.easy_axis,m),     '\t', p.m)
    #Hk12 = 2*p.K12/p.Js*p.mu0                                     #p.easy_axis2*np.dot(p.easy_axis2,m)  #, np.dot(p.easy_axis2,m),    '\t', p.m)
    Hd = p.etadamp*p.currentd*p.hbar/(2*p.e*p.Js*p.d)
    return (Hk, Hd)#Hk12[0] ,Hd)

def f(t, m, p):
 j=p.currentd*np.cos(2*3.1415927*p.frequency*t)
 prefactorpol = j*p.hbar/(2*p.e*p.Js*p.d)   #Zhu, Daoqian, and Weisheng Zhao. "Threshold Current Density for Perpendicular Magnetization Switching Through Spin-Orbit Torque." Physical Review Applied 13.4 (2020): 044078.
 hani=2*p.K1/p.Js*p.easy_axis*np.dot(p.easy_axis,m)
 h=p.hext+hani
# This is an equivalent formulation as discribed in the review article of Claas
# mxh = np.cross(m, h+prefactorpol*(p.alpha*p.etadamp-p.etafield)*p.p_axis)
# mxmxh = np.cross(m, np.cross(m, h+prefactorpol*(-1/p.alpha*p.etadamp-p.etafield)*p.p_axis))
# rhs = -p.gamma/(1+p.alpha**2)*mxh-p.gamma*p.alpha/(1+p.alpha**2)*mxmxh
 H=-prefactorpol*(p.etadamp*np.cross(p.p_axis,m)+p.etafield*p.p_axis)
 mxh = np.cross(m, h-prefactorpol*(p.etadamp*np.cross(p.p_axis,m)+p.etafield*p.p_axis))
 
 mxmxh = np.cross(m, mxh) 
 rhs = -p.gamma/(1+p.alpha**2)*mxh-p.gamma*p.alpha/(1+p.alpha**2)*mxmxh    #Field and damping like torque term
 #rhs = -p.gamma*p.alpha/(1+p.alpha**2)*mxmxh                                 #only damping torque
 #print(t,m[0],m[1],m[2])
 p.result.append([t,m[0],m[1],m[2],H[0],H[1],H[2]])
 return [rhs]
#def jac(t, y, arg1): 
# return [[1j*arg1, 1], [0, -arg1*2*y[1]]]

def calc_equilibrium(m0_,t0_,t1_,dt_,paramters_):
 t0 = t0_
 m0 = m0_
 dt = dt_
#prefactorpol = currentd*hbar/(2*e*Js)
# https://www.iue.tuwien.ac.at/phd/makarov/dissertationch5.html
 r = ode(f).set_integrator('vode', method='bdf',atol=1e-14,nsteps =500000)
 r.set_initial_value(m0_, t0_).set_f_params(paramters_).set_jac_params(2.0)
 t1 = t1_
 while r.successful() and r.t < t1:    
    mag=r.integrate(r.t+dt)
    Hs=fields(r.t,mag,paramters_)
 #print(mag,'\t', fields(r.t,mag,paramters_) )
 return(r.t,mag,Hs)

def calc_w1andw2(m0_,t0_,t1_,dt_,paramters_): 
 paramters_.result = []
 t1,mag1, Hs = calc_equilibrium(m0_,t0_,t1_,dt_,paramters_)
 npresults=np.array(paramters_.result, dtype=np.float128)
 time = npresults[:,0]
 coswt=np.cos( 2*3.1415927*paramters_.frequency*time)
 sin2wt=np.sin( 2*2*3.1415927*paramters_.frequency*time)
 current=paramters_.currentd*np.cos(2*3.1415927*paramters_.frequency*time)
 #print(current)
 z=0
 dt=[]
 dt.append(time[1]-time[0])
 for i in time: 
  if z>0:
   dt.append(time[z]-time[z-1])
  z=z+1
 dt=np.array(dt, dtype=np.float128)
 #print(dt)
 mz=npresults[:,3]
 #print(mz)
 #print(np.dot(sin2wt*dt,coswt)/np.dot(sin2wt*dt,sin2wt))
 voltage=current*npresults[:,3]*paramters_.RAHE
 #voltage = mz
 
 #print(voltage)
 #plt.plot(time, npresults[:,3])
 #plt.plot(time, (voltage-1.0)*10,'g')
 #plt.plot(time, coswt,'b')
 
 #plt.plot(time, coswt,'r')
 #plt.plot(fieldrangeT, signal2,'r')
 plt.show()
 #exit(1)
 
 R1w=np.dot( voltage*dt,coswt)/(np.dot(coswt*dt,coswt)*paramters_.currentd)
 R2w=np.dot( voltage*dt,sin2wt)/(np.dot(sin2wt*dt,sin2wt)*paramters_.currentd)
 #plt.plot(time, npresults[:,1], label = 'mx')
 #plt.plot(time, npresults[:,2], label = 'my')
 #plt.plot(time, npresults[:,3], label = 'mz')
 #plt.legend()
 #plt.show()
 #plt.plot(time, npresults[:,4], label = 'Hx')
 #plt.plot(time, npresults[:,5], label = 'Hy')
 #plt.plot(time, npresults[:,6], label = 'Hz')
 #plt.legend()
 #plt.show()
 #H_eff = print(npresults[-1,4],npresults[-1,5],npresults[-1,6])
 return(R1w,R2w,npresults[-1,4],npresults[-1,5],npresults[-1,6],npresults[-1,1],npresults[-1,2],npresults[-1,3], Hs)
 
# print(npresults[:,0])
 
paramters = Parameters()
fieldrange = np.linspace(-0.1/paramters.mu0, 0.1/paramters.mu0, num=209)
signalw  = []
signal2w  = []
Hx,Hy,Hz = [[],[],[]]
Mx,My,Mz = [[],[],[]]
fieldrangeT =[]
orgdensity = paramters.currentd
for i in fieldrange:
 paramters.hext = np.array([i,0,0])
 paramters.currentd = orgdensity
 initm=[0,0,1]
 initm=np.array(initm)/np.linalg.norm(initm)
 R1w,R2w,hx,hy,hz,mx,my,mz, Hs = calc_w1andw2(m0_=initm,t0_=0,t1_=30/paramters.frequency,dt_=30/paramters.frequency,paramters_=paramters)
 print(i,R1w,R2w, '\tHk,Hd', round(Hs[0]), round(Hs[1]), mx, my, mz)
 signalw.append(R1w)
 signal2w.append(R2w)
 fieldrangeT.append(i*paramters.mu0)
 Hx.append(hx)
 Hy.append(hy)
 Hz.append(hz)
 Mx.append(mx)
 My.append(my)
 Mz.append(mz)
 #if np.fabs(fmini(paramters,mag1))>1: 
 # print("Not converged or self oscilation\n")
 # exit(1)

def showplot():
    plt.plot(fieldrangeT, signal2w)
    plt.plot(fieldrangeT, Mx,'b')
    plt.plot(fieldrangeT, My,'g')
    plt.plot(fieldrangeT, Mz,'r')
    #plt.plot(fieldrangeT, H,'r')
    ax=plt.axes()
    ax.set(xlabel=r'$\mu_0 H_x$ (T)',ylabel=r'$R_{2w}$ ')
    plt.savefig('signal.png')
    plt.show()
    
def savedata(p, sig, fieldrangeT, name):
    with open( "v2o_" + str(name) + "_j" + str(p.currentd/1e10) + "e10.dat", "w") as f:
        i = 0
        for sig in signal2w:
            f.write( str(p.currentd) + "\t"  + str(fieldrangeT[i]) + "\t" + str(sig) + "\t" 
                    + str(Hx[i]) + "\t" + str(Hy[i]) + "\t" + str(Hz[i]) +'\t' 
                    + str(Mx[i]) + "\t" + str(My[i]) + "\t" + str(Mz[i]) + '\t' + str(signalw[i]) + "\t"
                    + "\n")
            i += 1
        #f.write("H_{k,1} \t H_{k,12} \t H^{DL}_{SOT} \t \eta \n")
        #f.write( str(Hs[0]) + '\t' + str(Hs[1]) +  '\t' + str(Hs[2]) + "\t" + str(p.etafield/p.etadamp) + "\t" + str(p.d) )
        f.close()

savedata(paramters, signal2w, fieldrangeT, "test")
showplot()
