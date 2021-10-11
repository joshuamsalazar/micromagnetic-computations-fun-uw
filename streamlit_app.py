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
