from scipy.integrate import ode
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os, sys

class Parameters:
 mu0=4*3.1415927*1e-7
 gamma=2.2128e5
 alpha=0.1
 Js=1.13
 K1=(0.03/mu0)*Js/2              # K1= -0.5*Js**2/(mu0) # K12=K1/500
 hbar=1.054571e-34
 e=1.602176634e-19
 easy_axis=np.array([0,0,1])
 easy_axis2=np.array([0,0,1])
 d=1e-9
 currentd=0#5e12
 linearrange=-0.05                #Tesla
 K12=(0.0012/mu0)*Js/2            #linearrange*Js/(2*mu0)
 p_axis=np.array([0,-1,0])
 #etadamp=0.09                  # is equal to spin Hall angle
 #etafield=0.17                 # etafield=eta*etadamp 
 eta=1                          # eta= eta_field/eta_damp
 hext=[-0.02/mu0,0,0]    
 tEvol=[]                         #Time evolution of: Time
 mEvol=[]                         #                   Magnetization direction
 HeffEvol=[]
 mxhEvol=[]                       #                   Fieldlike term
 mxmxhEvol=[]                     #                   Dampinglike term
 
def jacmini(p,m1):
 m=m1#/np.linalg.norm(m1)
 prefactorpol=p.currentd*p.hbar/(2*p.e*p.Js*p.d)
 hani1=2*p.K1/p.Js*p.easy_axis*np.dot(p.easy_axis,m)
 hani2=2*p.K12/p.Js*p.easy_axis2*np.dot(p.easy_axis2,m)
 h=p.hext+hani1+hani2
 #heff=h-prefactorpol*(p.etadamp*np.cross(p.p_axis,m)+p.etafield*p.p_axis)
 #heff=h+prefactorpol*(p.etadamp*np.cross(m,p.p_axis)+p.etafield*p.p_axis)
 heff=h+prefactorpol*(np.cross(m,p.p_axis)+p.eta*p.p_axis)
 heff/=np.linalg.norm(heff)
 mxh=np.cross(m, heff)
 return(mxh)
 
def fmini(p,m1):
 return(np.linalg.norm(jacmini(p,m1)))

def f(t, m, p):
 m=m/np.linalg.norm(m)
 prefactorpol=p.currentd*p.hbar/(2*p.e*p.Js*p.d) 
 hani1=2*p.K1/p.Js*p.easy_axis*np.dot(p.easy_axis,m)
 hani2=2*p.K12/p.Js*p.easy_axis2*np.dot(p.easy_axis2,m)
 h=p.hext+hani1+hani2
 heff=h+prefactorpol*(np.cross(m,p.p_axis)+p.eta*p.p_axis)
 mxh=np.cross(m, heff)
 mxmxh=np.cross(m, mxh)
 rhs=-p.gamma/(1+p.alpha**2)*mxh-p.gamma*p.alpha/(1+p.alpha**2)*mxmxh 
 p.tEvol.append(t)
 p.mEvol.append(m)
 p.HeffEvol.append(heff)
 p.mxhEvol.append(-p.gamma/(1+p.alpha**2)*mxh)
 p.mxmxhEvol.append(-p.gamma*p.alpha/(1+p.alpha**2)*mxmxh)
 return [rhs]


def reset_results(paramters_):
    paramters_.result=[]         
    paramters_.tEvol=[]          #Time evolution of: Time
    paramters_.mEvol=[]          #                   Magnetization direction
    paramters_.HeffEvol=[]
    paramters_.mxhEvol=[]        #                   Fieldlike term
    paramters_.mxmxhEvol=[]      #                   Dampinglike term

def calc_equilibrium(m0_,t0_,t1_,dt_,paramters_):
 t0=t0_
 m0=m0_
 dt=dt_
#prefactorpol=currentd*hbar/(2*e*Js)
# https://www.iue.tuwien.ac.at/phd/makarov/dissertationch5.html
 r=ode(f).set_integrator('vode', method='bdf',atol=1e-14)
# r.set_initial_value(m0_, t0_).set_f_params(paramters_).set_jac_params(2.0)
 r.set_initial_value(m0_, t0_).set_f_params(paramters_)
 t1=t1_
 count=0
 #resall=m0
 while r.successful() and r.t < t1:   
    count += 1
    mag=r.integrate(r.t+dt)
    #print(type(mag))
    #print(r.t+dt, mag)
 #input(mag)
 #print(resall)
 return(r.t,mag)

def show_relaxation(mPlt,mdcPlt,HeffPlt,mxhPlt,mxmxhPlt,rhsPlt):                    #Plotting function
    ax=plt.axes()
    #if mPlt == True:
    #    plt.plot(magList[0], magList[1] ,"C0.",  linewidth=3, label='Mx')
    #    plt.plot(magList[0], magList[2] ,"C1.",  linewidth=3, label='My')
    #    plt.plot(magList[0], magList[3] ,"C2.",  linewidth=3, label='Mz')
    if mdcPlt == True:
        plt.plot(t, m[:,0], "C0.", label='Mx ')
        plt.plot(t, m[:,1], "C1.", label='My ')
        plt.plot(t, m[:,2], "C2.", label='Mz ')
    if HeffPlt == True:
        plt.plot(t, Heff[:,0]/np.amax(Heff[:,0]), "C0-", label='hx')
        plt.plot(t, Heff[:,1]/np.amax(Heff[:,1]), "C1-", label='hy')
        plt.plot(t, Heff[:,2]/np.amax(Heff[:,2]), "C2-", label='hz')
    if mxhPlt == True:
        plt.plot(t, mxh[:,0]/np.amax(mxh[:,0]), "C0-.", label='Mxhx')
        plt.plot(t, mxh[:,1]/np.amax(mxh[:,1]), "C1-.", label='Mxhy')
        plt.plot(t, mxh[:,2]/np.amax(mxh[:,2]), "C2-.", label='Mxhz') #np.linalg.norm(mxh, axis=1)
    if mxmxhPlt == True:
        plt.plot(t, mxmxh[:,0]/np.amax(np.abs(mxmxh[:,0])), "C0--", label='MxMxhx')
        plt.plot(t, mxmxh[:,1]/np.amax(np.abs(mxmxh[:,1])), "C1--", label='MxMxhy')
        plt.plot(t, mxmxh[:,2]/np.amax(np.abs(mxmxh[:,2])), "C2--", label='MxMxhz')
    if rhsPlt == True:
        plt.plot(t, mxh[:,0]/np.amax(np.abs(mxh[:,0]))+mxmxh[:,0]/np.amax(np.abs(mxmxh[:,0])), "C0-", label='dm/dt')
        plt.plot(t, mxh[:,1]/np.amax(np.abs(mxh[:,1]))+mxmxh[:,1]/np.amax(np.abs(mxmxh[:,1])), "C1-", label='dm/dt')
        plt.plot(t, mxh[:,2]/np.amax(np.abs(mxh[:,2]))+mxmxh[:,2]/np.amax(np.abs(mxmxh[:,2])), "C2-", label='dm/dt')
    #plt.plot(magList[0], (np.sin(2 * 3.1415927 * paramters_.frequency * magList[0])) ,"C3--",  linewidth=3, label='Je')
    ax.set(xlabel=r't [s] ',ylabel='')
    plt.title("|H_ext|=%.4f [T]     eta=%.3f    |m_fxh|=%g \n" % (paramters.hext[0]*paramters.mu0, paramters.eta, norm2)) #M_i')#r'$V_{2w} [V]$ 
    plt.legend()
    plt.show()


paramters=Parameters()
p=paramters
if True: #Show plot
    initm=[0,0,1]
    t2, mag2=calc_equilibrium(m0_=initm,t0_=0,t1_=15e-9,dt_=1e-9,paramters_=paramters)
    t                =np.array(paramters.tEvol)
    m                =np.array(paramters.mEvol)
    Heff             =np.array(paramters.HeffEvol)
    mxh              =np.array(paramters.mxhEvol)
    mxmxh            =np.array(paramters.mxmxhEvol)
    norm2=np.fabs(fmini(paramters,mag2))
    print("|H| [T]: %.3f \t eta: %.2f \t norm: %g \t j_e= %1.1e [10^10A/m^2] \t Hk= %.3f [T] \t alph: %.1f \t K12: %.3f" 
    % (p.hext[0]*p.mu0, p.eta, norm2, p.currentd, 
        2*p.K1/p.Js*p.mu0, p.alpha, 2*p.K12/p.Js*p.mu0) )
    #print("\n |H| [T]: %.4f \t eta=%.4f \t |m_fxh|=%g \n" % (paramters.hext[0]*paramters.mu0, p.eta, norm2))
    show_relaxation(mPlt=False,mdcPlt=True,HeffPlt=False,mxhPlt=False,mxmxhPlt=False,rhsPlt=False)
    exit()
    
####################################################################################################

a=0.02/(p.mu0)
eta=10
fieldrange=np.linspace(-a, a, num=4) # np.linspace(-a, a, num=13)
etarange=np.linspace(-eta, eta, num=4)
fieldrangeT =[]
orgdensity=paramters.currentd
data2d=[]
X, Y=np.meshgrid(fieldrange*(p.mu0),etarange)

k1Range=[-13488.38122778, 13488.38122778]#np.linspace( K1, -K1, num=3)# -K1*3=60 mT [float(sys.argv[1])]#
k12Range=np.linspace( -0.02*(p.Js*(0.060/p.mu0)/2), 0.02*(p.Js*(0.060/p.mu0)/2), num=2)
currRange=np.linspace( 5e10, 5e12, num=2)#currRange=np.linspace( 1e10, 1e13, num=6)
alphaRange= [0.1,0.2] #,0.8,1][float(sys.argv[2])]#
c_total= len(k1Range)*len(k12Range)*len(currRange)*len(alphaRange)*len(fieldrange)*len(etarange)
c=0
#os.system("rm grid_k1%1.1e_alph%1.1e.dat" % (2*p.K1*p.Js*p.mu0, m) )
for n in k12Range:
 paramters.K12=n
 for m in alphaRange:
  paramters.alpha=m
  for k in k1Range:
   paramters.K1=k
   for l in currRange:
    paramters.currentd=l
    for j in etarange:
     paramters.eta=j
     #signal =[]  
     for i in fieldrange:
      paramters.hext=np.array([i,0.00/(p.mu0),0.00/(p.mu0)])
      initm=[1,0,0]
      t2,mag2=calc_equilibrium(m0_=initm,t0_=0,t1_=150e-9,dt_=1e-9,paramters_=paramters)
      norm2=np.fabs(fmini(paramters,mag2))
      c+=1
      #print("###############################################################")
      print("|H|[T]: %.3f \t eta: %.2f \t norm: %g \t j_e= %1.1e [A/m^2] \t Hk= %.3f [T] \t alph: %.1f \t Hk12[T]: %.3f %1.1f/%1.1f" 
            % (i*p.mu0, p.eta, norm2, l, 
               2*p.K1/p.Js*p.mu0, p.alpha, 2*p.K12/p.Js*p.mu0, c, c_total) )
      os.system("echo '%g \t %g \t %g \t %g \t %g \t %g \t %g' >> grid_k1%1.1e_alph%1.1e.dat" 
                % (2*p.K12/p.Js*p.mu0, p.alpha, 
                   2*p.K1/p.Js*p.mu0, p.currentd,
                   i*p.mu0, p.eta, norm2, 2*p.K1/p.Js*p.mu0, m) ) 
      reset_results(p)
  
      #os.system("echo '%g \t %g \t %g \t %g \t %g \t %g \t %g' >> grid_full.dat" 
      #          % (2*p.K12*p.Js*p.mu0, p.alpha, 2*p.K1*p.Js*p.mu0, p.currentd, i*p.mu0, p.eta, norm2) )  
      #os.system("echo '%g \t %g \t %g \t %g \t %g' >> grid_hk%1.1e_je%1.1e.dat" 
      #          % (2*p.K1*p.Js*p.mu0, p.currentd, i*p.mu0, p.eta, norm2, p.K1*p.mu0/2, p.currentd) )
      
  #if norm2>1e-3: print("norm2 Not converged or self oscilation\n")
  #  signal.append(norm2)
  #  fieldrangeT.append(i*(p.mu0))
  # data2d.append(signal)

#contour=plt.contour(X, Y, data2d)
#plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
#contour_filled=plt.contourf(X, Y, data2d)
#plt.colorbar(contour_filled)
#plt.title(r'$\alpha=%g$  $j_e =$ %2.1e' % (p.alpha, p.currentd) )
#plt.xlabel('Bx (T)')
#plt.ylabel(r'$\eta$(field/damp)')
#plt.savefig(sys.argv[0] + '.png', dpi=300)
#plt.show()
#os.system("cp " + str(sys.argv[0]) + "_alph" + str(sys.argv[1]) + ".py.bck")
