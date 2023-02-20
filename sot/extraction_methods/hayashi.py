import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.misc import derivative

##### **Hayashi method**
##### Based on the method described on 10.1103/PhysRevB.89.144425.

#Variables
pwd = "./"
expX, expY, expZ = (6 -2,7 -2,8 -2)
dirLong = 8 -2
dirTrans = 6 -2
colVw = 9 -2
colV2w = 11 -2
listVolt=("0.4","0.6","0.8","1.0")
listDirection=("trans","long")
mu0=4*np.pi*1e-7
tM2=0.9e-9
JsM2=1.55
RaheM2=0.8
RpheM2=0.02
hbar = 1.054571e-34
e = 1.602176634e-19
pf=hbar/(2*e*JsM2*tM2)
areaM2=((6+0.9)*1e-9 * 2e-6)

#Fitting functions
def linealFx(Hext, a ,b): 
    return a*Hext + b

def parabFx(Hext, a ,b, c):
    return a*(Hext**2) + b*Hext + c

def vxy(t,v0,v1,v2):
    return v0 + v1*np.sin(w*t) + v2*np.cos(2*w*t)

#Class describing the data of a 2omega scan from the experimental set
class HextSweep():
    
    def __init__(self, file, direction= "trans", mz = '+', volt = "1.0", color = "C0" ):
        self.direction = direction # trans or long
        self.file = file #filename
        self.mz = mz # mz+ or mz-
        self.volt = volt #Applied voltage in str()
        self.df = pd.read_csv(pwd + self.file, skiprows=0, sep='\t')

    def Current(self):
        return self.df.iloc[:,13].mean()/50


dictH = {i: #Dictionary storing each file and using atributes as keys
             {j: {'mz+': None, 'mz-' : None} for j in ('trans', 'long')}
             for i in ("0.4","0.6","0.8","1.0")}

dictEtas = {i: #Dictionary storing each file and using atributes as keys
             {j: {'mz+': None, 'mz-' : None} for j in ('field', 'damp')}
             for i in ("0.4","0.6","0.8","1.0")}

def loadRahulData():
    prefix="V/2wScan_M2device_" #Constant strings on filenames
    midfix="V_Vxy_avg10_87deg_"
    sufix = "M_H0.0E+0T_phi-3°_theta 0°_1.50sWait.dat"
    
    for dire in ("trans","long"): #Loading the data

        if dire == "trans":
            midfix="V_Vxy_avg10_-3deg_"
            sufix   = "M_H0.0E+0T_phi-3°_theta 0°_1.50sWait.dat"
        else: 
            midfix="V_Vxy_avg10_87deg_"
            sufix = "M_H0.0E+0T_phi87°_theta 0°_1.50sWait.dat"     

        for volt in listVolt:
            dictH[volt][dire]['mz-'] = HextSweep(file = volt+prefix+volt+midfix+"-"+sufix,
                                            volt=volt,
                                            mz='-',
                                            direction=dire)
            dictH[volt][dire]['mz+'] = HextSweep(file = volt+prefix+volt+midfix+"+"+sufix,
                                            volt=volt,
                                            mz='+',
                                            direction=dire)
#Dictionary storing the current [A]
loadRahulData()
dictCurr = { i : dictH[i]["trans"]["mz+"].Current() for i in listVolt}

resEtas = {i : ["DLP", "FLP", "DLM", "FLM"] for i in listVolt}

######################  1.0  ################
for volt in listVolt:
    plt.subplot(2,2,1)
    plt.title("vw xy @ %s V"%(volt))
    x=-dictH[volt]["trans"]["mz+"].df.iloc[63:123,dirTrans]
    y=-dictH[volt]["trans"]["mz+"].df.iloc[63:123,colVw]
    vwtransMzM=np.polyfit(x, y, 2)
    plt.plot(x, y, "C0.", label="vw mz-")
    plt.plot(x, x**2*vwtransMzM[0] + x*vwtransMzM[1] + vwtransMzM[2], "C0", label="vw mz-")

    x=-dictH[volt]["trans"]["mz-"].df.iloc[63:123,dirTrans]
    y=-dictH[volt]["trans"]["mz-"].df.iloc[63:123,colVw]
    vwtransMzP=np.polyfit(x, y, 2)
    plt.plot(x, y, "C1.", label="vw mz+")
    plt.plot(x, x**2*vwtransMzP[0] + x*vwtransMzP[1] + vwtransMzP[2], "C1", label="vw mz+")
    plt.ylabel("[V]")
    plt.xlabel("m0 Htrans [T] (m+ x curr+)")
    plt.legend()

    plt.subplot(2,2,2)
    plt.title("vw xy @ %s V"%(volt))
    x=-dictH[volt]["trans"]["mz+"].df.iloc[63:123,dirTrans]
    y=-dictH[volt]["trans"]["mz+"].df.iloc[63:123,colV2w]
    v2wtransMzM=np.polyfit(x, y, 1)
    plt.plot(x, y, "C0.", label="v2w mz-")
    plt.plot(x, x*v2wtransMzM[0] + v2wtransMzM[1] , "C0", label="v2w mz-")

    x=-dictH[volt]["trans"]["mz-"].df.iloc[63:123,dirTrans]
    y=-dictH[volt]["trans"]["mz-"].df.iloc[63:123,colV2w]
    v2wtransMzP=np.polyfit(x, y, 1)
    plt.plot(x, y, "C1.", label="vw mz+")
    plt.plot(x, x*v2wtransMzP[0] + v2wtransMzP[1], "C1", label="vw mz+")
    plt.xlabel("m0 Htrans [T] (m+ x curr+)")
    plt.legend()

    plt.subplot(2,2,3)
    x=+dictH[volt]["long"]["mz+"].df.iloc[63:123,dirLong]
    y=-dictH[volt]["long"]["mz+"].df.iloc[63:123,colVw]
    vwlongMzM=np.polyfit(x, y, 2)
    plt.plot(x, y, "C0.", label="v2w mz-")
    plt.plot(x, x**2*vwlongMzM[0] + x*vwlongMzM[1] + vwlongMzM[2], "C0", label="vw mz-")

    x=+dictH[volt]["long"]["mz-"].df.iloc[63:123,dirLong]
    y=-dictH[volt]["long"]["mz-"].df.iloc[63:123,colVw]
    vwlongMzP=np.polyfit(x, y, 2)
    plt.plot(x, y, "C1.", label="vw mz+")
    plt.plot(x, x**2*vwlongMzP[0] + x*vwlongMzP[1] + vwlongMzP[2], "C1", label="vw mz+")
    plt.ylabel("[V]")
    plt.xlabel("m0 Hlong [T] (curr+)")
    plt.legend()

    plt.subplot(2,2,4)
    x=+dictH[volt]["long"]["mz+"].df.iloc[63:123,dirLong]
    y=-dictH[volt]["long"]["mz+"].df.iloc[63:123,colV2w]
    v2wlongMzM=np.polyfit(x, y, 1)
    plt.plot(x, y, "C0.", label="v2w mz-")
    plt.plot(x, x*v2wlongMzM[0] + v2wlongMzM[1] , "C0", label="v2w mz-")

    x=+dictH[volt]["long"]["mz-"].df.iloc[63:123,dirLong]
    y=-dictH[volt]["long"]["mz-"].df.iloc[63:123,colV2w]
    v2wlongMzP=np.polyfit(x, y, 1)
    plt.plot(x, y, "C1.", label="vw mz+")
    plt.plot(x, x*v2wlongMzP[0] + v2wlongMzP[1], "C1", label="v2w mz+")
    plt.xlabel("m0 Hlong [T] (curr+)")
    plt.legend()

    ksi=RpheM2/RaheM2
    CurvVw = 2*vwtransMzP[0]
    SlopeV2w = v2wtransMzP[0]
    bFieldP = SlopeV2w/CurvVw
    CurvVw = 2*vwlongMzP[0]
    SlopeV2w = v2wlongMzP[0]
    bDampP = SlopeV2w/CurvVw
    CurvVw = 2*vwtransMzM[0]
    SlopeV2w = v2wtransMzM[0]
    bFieldM = SlopeV2w/CurvVw
    CurvVw = 2*vwlongMzM[0]
    SlopeV2w = v2wlongMzM[0]
    bDampM = SlopeV2w/CurvVw
    resSOTDampP = -2*((bDampP+2*ksi*bFieldP)/(1-4*ksi**2))
    resSOTFieldP = -2*((bFieldP+2*ksi*bDampP)/(1-4*ksi**2))
    resSOTDampM = -2*((bDampM-2*ksi*bFieldM)/(1-4*ksi**2))
    resSOTFieldM = -2*((bFieldM-2*ksi*bDampM)/(1-4*ksi**2))
    resSOT10 = np.array([resSOTDampP, resSOTFieldP, resSOTDampM, resSOTFieldM]) #Returns the HSOT_eff values [T]
    resEtas[volt] = np.array([resSOTDampP, resSOTFieldP, resSOTDampM, resSOTFieldM])/(mu0 * pf * dictCurr[volt] / areaM2)
    plt.tight_layout()
    btrans=(v2wtransMzP[0]/(vwtransMzP[0]))
    blong=(v2wlongMzP[0]/(vwlongMzP[0]))
    resEtas[volt][0]=blong/( mu0 * pf * dictCurr[volt] / areaM2)
    resEtas[volt][1]=btrans/(-mu0 * pf * dictCurr[volt] / areaM2)
    btransM=(v2wtransMzM[0]/(vwtransMzM[0]))
    blongM=(v2wlongMzM[0]/(vwlongMzM[0]))
    resEtas[volt][2]=-blongM/( mu0 * pf * dictCurr[volt] / areaM2)
    resEtas[volt][3]=btransM/(-mu0 * pf * dictCurr[volt] / areaM2)
    print("Current [mA]: %.2f [1e10 A/m^2]: %.2f \t Voltage applied %s V"%(dictCurr[volt]*1000, dictCurr[volt]/areaM2/1e10, volt))
    print("########################################", volt, "###########################")
    print("mz+:\t\t  Slope \t  CurvVw \tmz-: \t    Slope \t   CurvVw ")
    print("Long\t\t  %.1e \t %.1e \t\t| %.1e \t %.1e"%(v2wlongMzP[0], 2*vwlongMzP[0], v2wlongMzM[0], 2*vwlongMzM[0]))
    print("Ratio S/C\t\t\t %.1e   \t\t| %.1e "%(v2wlongMzP[0]/(2*vwlongMzP[0]), v2wlongMzM[0]/(2*vwlongMzM[0])))
    print("mz+:\t\t  Slope \t  CurvVw \tmz-: \t    Slope \t   CurvVw ")
    print("Trans\t\t  %.1e \t %.1e \t\t| %.1e \t %.1e"%(v2wtransMzP[0], 2*vwtransMzP[0], v2wtransMzM[0], 2*vwtransMzM[0]))
    print("Ratio S/C\t\t\t %.1e   \t\t| %.1e "%(v2wtransMzP[0]/(2*vwtransMzP[0]), v2wtransMzM[0]/(2*vwtransMzM[0])))
    print("\t\tExtracted parameters\t\t\t\t mz-:")
    print("HSOT_eff DL [mT]: %.3f \t etaDL  : %.3f   \t\t|   %.3f "%(resSOTDampP*1000, resEtas[volt][0], resEtas[volt][2]))
    print("HSOT_eff FL [mT]: %.3f \t etaFL  : %.3f   \t\t|   %.3f "%(resSOTFieldP*1000, resEtas[volt][1], resEtas[volt][3]))
    plt.show()

############################### Results #############################
resEtaDLP = np.array([resEtas["0.4"][0], resEtas["0.6"][0], resEtas["0.8"][0], resEtas["1.0"][0]])
resEtaFLP = np.array([resEtas["0.4"][1], resEtas["0.6"][1], resEtas["0.8"][1], resEtas["1.0"][1]])
resEtaDLM = np.array([resEtas["0.4"][2], resEtas["0.6"][2], resEtas["0.8"][2], resEtas["1.0"][2]])
resEtaFLM = np.array([resEtas["0.4"][3], resEtas["0.6"][3], resEtas["0.8"][3], resEtas["1.0"][3]])

plt.plot(np.fromiter(dictCurr.values(), dtype=float)/areaM2, resEtaDLM, "C0--", label = "Eta DL mz-")
plt.plot(np.fromiter(dictCurr.values(), dtype=float)/areaM2, resEtaDLP, "C1--", label = "Eta DL mz+")
plt.legend()
plt.ylabel("eta ")
plt.plot(np.fromiter(dictCurr.values(), dtype=float)/areaM2, resEtaFLM, "C0.-", label = "Eta FL mz-")
plt.plot(np.fromiter(dictCurr.values(), dtype=float)/areaM2, resEtaFLP, "C1.-",label = "Eta FL mz+")
plt.legend()
plt.xlabel("je [1e10 A/m^2]")
plt.savefig("m2_0.9nm_etas.png")
plt.show()
