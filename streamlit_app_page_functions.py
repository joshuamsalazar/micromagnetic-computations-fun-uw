import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
from streamlit_app_functions.theoretical_description import text as text_theoretical_description
print("Page functions loaded.")

def header():
    #st.title('Magnetization dynamics for FM/HM interfaces, a single-spin model')
    st.header('Online LLG integrator - External Field sweeps')
    st.caption("Joshua Salazar, S. Koraltan, C. Abert, P. Flauger, M. Agrawal, S. Zeilinger, A. Satz, C. Schmitt, G. Jakob, R. Gupta, M. Kläui, H. Brückl, J. Güttinger and Dieter Suess")
    st.caption("University of Vienna - Physics of Functional Materials")
    st.subheader("Please WAIT until the simulation finishes before submiting new values")

def text_description():
    st.subheader('Theoretical description')
    text_theoretical_description()

#Functions for plots
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
   ax.set_ylim([-1.05,1.05])
   plt.title(plthead)
   plt.legend()
   return fig

#Old functions for fourier transform and lock-in transform. Deprecated.
def lockin(sig, t, f, ph):
    '''Lock in function for a signal sig(t) at frequency f with phase ph'''
    ref = np.cos(2 * 2*np.pi*f*t + ph/180.0*np.pi)
    #ref = np.sin(2*np.pi*f*t + ph/180.0*np.pi)
    comp = np.multiply(sig,ref)
    #print(t[-1]) #plot real part fft 
    return comp.mean()*2

def fft(sig, t, f):
    '''FFT function for a signal sig(t) at frequency f. Deprecated. Use sin-cos fit instead.'''
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

#fR2w           = fft( voltage, magList[0][periSampl:], params["frequency"])
#lR2w           = lockin( voltage, magList[0][periSampl:], params["frequency"], 0)
#nR2w           = lockin( voltage/params["currentd"], magList[0][periSampl:], params["frequency"], 90)

def fields(t,m,p): 
    #Get the H^{DL} at (t, m, p)
    Hk = 2 * p["K1"]/p["Js"]
    Hd = p["etadamp"] * p["currentd"] * p["hbar"]/(2*p["e"]*p["Js"]*p["d"])
    return (Hk, Hd)
    
def savedata(p, sig, fieldrangeT, name):
    #Storing the data into a dat file with the following strcture:
    #Delta denotes current-induced fields
    # ` denotes equilibium  
    # Current | H_ext | R2w | \Delta H_x | \Delta H_y | \Delta H_z | 7mz` | my` | mz` | Rw | 11 phi rad
    with open( "v2o_" + str(name) + "_j" + str(p["currentd"]/1e10) + "e10.dat", "w") as f:
        i = 0
        for sig in signal2w:
            f.write( str(p["currentd"]) + "\t"  + str(fieldrangeT[i]) + "\t" + str(sig) + "\t" 
                    + str(Hx[i]) + "\t" + str(Hy[i]) + "\t" + str(Hz[i]) +'\t' 
                    + str(Mx[i]) + "\t" + str(My[i]) + "\t" + str(Mz[i]) + '\t' + str(signalw[i]) + "\t" + str(phirangeRad[i])
                    + "\n")
            i += 1
        f.write("Hk\tHdamp\teta(f/d)\t t\t freq\n")
        f.write( str(Hs[0]) + '\t' + str(Hs[1]) + "\t" + str(p["etafield"]/p["etadamp"]) + "\t" + str(p["d"]) 
                + '\t' + str(p["frequency"]) + '\n')
        f.close()
