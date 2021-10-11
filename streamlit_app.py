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

hobby = st.selectbox("Hobbies: ",
                     ['Dancing', 'Reading', 'Sports'])
 
# print the selected hobby
st.write("Your hobby is: ", hobby)

periSampl = 1000

class Parameters:
    gamma = 2.2128e5
    alpha = 1.0
    K1 = 1.5 * 9100   
    Js = 0.46
    RAHE = 0.65 
    d = (0.6+1.2+1.1) * 1e-9      
    frequency = 0.1e9
    currentd = float(sys.argv[1]) * 1e10
    hbar = 1.054571e-34
    e = 1.602176634e-19
    mu0 = 4 * 3.1415927 * 1e-7
    easy_axis = np.array([0,0,1])
    p_axis = np.array([0,-1,0])
    etadamp = 0.084    
    etafield = 0.008   # etafield/etadamp=eta
    eta = etafield/etadamp
    hext = np.array([1.0 * K1/Js,0,0])
