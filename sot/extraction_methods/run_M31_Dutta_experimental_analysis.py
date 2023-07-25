import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

dict_name = str(sys.argv[1])

mu0 = 4*3.1415927*1e-7
hbar = 1.054571e-34
e = 1.602176634e-19
Js = 1.13
d = 1.e-9
resistor = 50.
area = 7e-14

def U2w_model(x,CA,CP,offset):
    return CA*np.cos(x) + CP*np.cos(x)*np.cos(2.*x) + offset

def dutta_model(x, Hdl, Hfl, offset, H_ext, R_AHE, R_PHE, Hk):
    return -1/2.*(Hdl/(H_ext+Hk))*R_AHE*I*np.cos(x) - ((Hfl+H_oe)/H_ext)*R_PHE*I*np.cos(x)*np.cos(2.*x) + offset

def dutta_model_Hk(x, Hdl, Hfl, Hk, offset, H_ext, R_AHE, R_PHE):
    return -1/2.*(Hdl/(H_ext+Hk))*R_AHE*I*np.cos(x) - ((Hfl+H_oe)/H_ext)*R_PHE*I*np.cos(x)*np.cos(2.*x) + offset

def plotDynamics(filename,t,j,mx,my,mz):
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].plot(t,mx,label="mx")
    axs[0].plot(t,my,label="my")
    axs[0].plot(t,mz,label="mz")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("m")
    axs[0].legend()
    axs[1].plot(t,j*(mz + mx*my))
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("HE")
    fig.tight_layout()
    fig.savefig(filename+".pdf")

H_fl_vals = []
H_dl_vals = []
CA_vals = []
CP_vals = []
eta_fl_vals = []
eta_dl_vals = []
h_mean_vals = []
j_vals = []

V_vals = [1.0,1.5,2.0,3.0,4.0]#,5.0]
R_AHE_vals = [0.174, 0.174, 0.174, 0.174, 0.158]#, 0.083]
R_PHE_vals = np.array([0.021, 0.020, 0.019, 0.017, 0.013])*2#, 0.03]
Js_vals = np.array([1.54, 1.53, 1.53, 1.53, 1.53])#, 0.95])
H_perp_vals = np.array([0.062, 0.061, 0.055, 0.049, 0.038])/(mu0)#, -0.044])/(mu0)
H_oe_vals = np.array([22.5734,33.8884,45.2747,68.3019,91.8104])#*100

for V,H_perp,R_AHE,R_PHE,Js,H_oe in zip(V_vals,H_perp_vals, R_AHE_vals, R_PHE_vals, Js_vals, H_oe_vals):
    H_fl_vals.append([]) 
    H_dl_vals.append([])
    CA_vals.append([])
    CP_vals.append([])
    eta_fl_vals.append([])
    eta_dl_vals.append([])
    h_mean_vals.append([])
    j_vals.append([])

    for htext in ["11", "13", "17", "19"]:
        filename = "vector-scanT300K phe "+str(V)+"V_72steps_H0."+htext+"T_phi_-90.0to270.0_theta_0.0to0.0_v0.dat"
        file = dict_name+filename

        h_vals = []
        H_vals = []
        phi_vals = []
        Ux_vals = []
        Uy_vals = []
        with open(file) as f:
            lines = f.readlines()
            for line in lines[1:]:
                split_line = line.split("\t")

                h_vals.append([float(split_line[4]), float(split_line[5]), float(split_line[6]) ])
                H_vals.append(float(split_line[7]))
                phi_vals.append(float(split_line[8]))
                Ux_vals.append([float(split_line[10]), float(split_line[12]), float(split_line[14]), float(split_line[16])])
                Uy_vals.append([float(split_line[11]), float(split_line[13]), float(split_line[15]), float(split_line[17])])

        h_vals = np.array(h_vals)
        H_vals = np.array(H_vals)/mu0
        phi_vals = np.pi*np.array(phi_vals)/180.
        Ux_vals = np.array(Ux_vals)
        Uy_vals = np.array(Uy_vals)
        
        UR = np.mean(Ux_vals[:,2])
        I = UR/resistor
        print(F"############### {V} V ### I {I*1000:.2} mA Hext {htext} mT ")
        j = I/area

        V_AHE = R_AHE * I
        V_PHE = R_PHE * I

        factor = 0.5*j*hbar/(e*Js*d)

        y_vals = Ux_vals[:,1]
        H_ext = np.mean(H_vals)
        params = curve_fit(U2w_model, phi_vals, y_vals)[0]
        CA = params[0]
        CP = params[1]

        H_aniso = -H_perp
        
        H_fl = -H_ext*CP/V_PHE - H_oe 
        H_dl = -2.*CA*(H_ext+H_aniso)/V_AHE
        
        #                                                      def dutta_model_Hk(x, Hdl, Hfl, Hk, offset, H_ext, R_AHE, R_PHE):
        paramsNew = curve_fit(lambda x, Hdl, Hfl, H_aniso, offset: dutta_model_Hk(x, Hdl, Hfl, H_aniso, offset, H_ext, R_AHE, R_PHE), phi_vals, y_vals)[0]
        H_dlNEW = paramsNew[0]
        H_flNEW = paramsNew[1]
        H_kNEW= paramsNew[2]
        CANEW=-1/2.*(H_dlNEW/(H_ext+H_aniso))*R_AHE*I
        CPNEW=- ((H_flNEW+H_oe)/H_ext)*R_PHE*I
        
        print(" T(Rxx):\t CA = %1.1e \t H_dl = %1.2f (mT) \t etaD = %.2f \t Hk(Rxx)= %1.2f mT"%(CA, H_dl*mu0*1000, H_dl/factor, -H_aniso*mu0*1000))
        print("*3ple fit:\t CA = %1.1e \t H_dl = %1.2f (mT) \t etaD = %.2f \t Hk fit = %1.2f mT"%(CANEW, H_dlNEW*mu0*1000, H_dlNEW/factor, H_kNEW*mu0*1000))
        print(" T(Rxx):\t CP = %1.1e \t H_fl = %1.2f (mT) \t etaF = %.2f \t Hk(Rxx)= %1.2f mT"%(CP, H_fl*mu0*1000, H_fl/factor, -H_aniso*mu0*1000))
        print("*3ple fit:\t CP = %1.1e \t H_fl = %1.2f (mT) \t etaF = %.2f \t Hk fit = %1.2f mT"%(CPNEW, H_flNEW*mu0*1000, H_flNEW/factor, H_kNEW*mu0*1000))
     
        j_vals[-1].append(j)
        h_mean_vals[-1].append(H_ext)
        H_fl_vals[-1].append(H_fl)
        H_dl_vals[-1].append(H_dl)
        eta_dl_vals[-1].append(H_dl/factor)
        eta_fl_vals[-1].append(H_fl/factor)
        CA_vals[-1].append(CA)
        CP_vals[-1].append(CP)

        text_H_fl = "H_fl = %.4e T" % (mu0*H_fl)
        text_H_dl = "H_dl = %.4e T" % (mu0*H_dl)
        text_eta_fl = "\n$\eta$_fl = %.4e" % (H_fl/factor)
        text_eta_dl = "\n$\eta$_dl = %.4e" % (H_dl/factor)

plt.close()
fig, axs = plt.subplots(2, gridspec_kw = {'wspace':0, 'hspace':0})
for dl_vals in np.transpose(eta_dl_vals):
    axs[0].plot(V_vals, dl_vals, marker="o", label="dl")
for fl_vals in np.transpose(eta_fl_vals):
    axs[1].plot(V_vals, fl_vals, marker="o", label="fl")
for ax in axs:
    ax.legend()
plt.show()

plt.close()
fig, axs = plt.subplots(2, gridspec_kw = {'wspace':0, 'hspace':0})
for dl_vals in np.transpose(eta_dl_vals):
    axs[0].plot(V_vals, np.array(H_dl_vals)*mu0*1000, marker="o", label="dl")
for fl_vals in np.transpose(eta_fl_vals):
    axs[1].plot(V_vals, np.array(H_fl_vals)*mu0*1000, marker="o", label="fl")
plt.show()

print("Hd in mt")
print(np.array(H_dl_vals)*mu0*1000)
print("Hf in mt")
print(np.array(H_fl_vals)*mu0*1000)