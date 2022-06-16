#!/usr/bin/python3
import arrayfire as af
from magnumaf import *
from scipy.constants import mu_0 as mu0
import os, sys

args = parse()
filepath = args.outdir

af.set_device(int(2))                                                                                                   #!!!
af.info()

#####CHANGELOG#####
# 17 -> 18: stronger field t the end to recover +z magntization
# x18 -> f01: full cross dynamics
# f03 -> _2: relax function saves the state.t to restart simulations
#     -> _3: restarting a relaxed state out. And recovering adequate time
#     -> _4: letting a full relaxation run to compare if the results are the same (keeping state.t)
# f03_4 -> f04: full sized cross, added parameters to use an external script
# f04         : removed parameters, created xrun01.sh to stre output separatedly
# f07         : added routine external field sweeps, similar syntax of data files to open in Mathematica
# v3M1full    : Starting from homogeneous state and checking if currents cause domain formation
# v4testM1    : Testing current increase (7.8e10 not enough to induce ip magnetization)
# v5fullM1    : INcreasing current further
###################

####README##########
'''
Through out the script, there are many variables that
need to be set. I tried to make as many comments as possible,
and I believe that it should be understandable for everyone.

Please go through the code completely once, to understand
its structure.

Eventually I will move all the input parameters to
the beginning of the script, or I will make them to be read
from an external .txt file.

Harald also added the option to simulate sensor transfer curves
using static LLG field relaxation method.
For Sensor transfer curves use this option
For any questions, don't hesitate to contact me.

Best wishes,
Sabri

'''

###################################################
###################################################
###Auxiliary Functions for mat.pars and geometry###
###################################################
###################################################
def calcK1(Hk, Js):
    Keff = Js*Hk/2./mu0
    K = Keff + 0.5*mu0*(Js/mu0)**2.
    return K

def calcAex(Js_T):
    A0 = 20e-12#Joule/meter
    J0 = 1.2#Tesla
    Aex_T = A0*(Js_T/J0)**1.7 
    return Aex_T

def cross_geometry(nx : int, ny: int, nz : int, width_cross_x : float, width_cross_y : float, make_3d = True, region = 0):
    thickness_x = int(width_cross_x/dx)
    thickness_y = int(width_cross_y/dy)
    
    cross = af.constant(0, nx, ny, nz, 1,  dtype=af.Dtype.f64)
    x_strt = nx/2 - thickness_x/2
    x_stop = nx/2 + thickness_x/2
    y_strt = ny/2 - thickness_y/2
    y_stop = ny/2 + thickness_y/2
    if region == 0:        #Hall Cross Geometry
        cross[x_strt : x_stop, :] = 1
        cross[:, y_strt:y_stop] = 1
    elif region == 1:      #Only stripe along X -> relevant for Rxx Measurements
        cross[:, y_strt:y_stop] = 1
    elif region == 2:	   #Only stripe along Y -> relevant for Ryy Measurements
        cross[x_strt : x_stop, :] = 1
    elif region == 3:      #Only central region -> relevant for Ryx Measurements
        cross[x_strt : x_stop, y_strt : y_stop ] = 1
    if make_3d:
        return af.tile(cross, 1, 1, 1, 3)
    else:
        return af.tile(cross, 1, 1, 1, 1)

def get_random_m0(nx, ny, nz, mask, seed=None):
    if seed: np.random.seed(seed)
    m0 = np.random.normal(0,1,(nx,ny,nz,3))
    m0 = m0/np.linalg.norm(m0,axis=3,keepdims=True)
    m0 = af.interop.np_to_af_array(m0)
    return m0 * mask

def get_hext_arr(h_ext, h_axis):
    return h_ext * h_axis 

###################################################
###################################################
#####Auxiliary Arrays for material parameters######
###################################################
###################################################

#####
## 
os.system("echo '###T (°C),0.92nm,1.02nm,1.14nm,1.25nm\n25.222,266.82,61,-280.59,-683.83\n100.404,201.7,44.26,-261.62,-618.85\n150.121,171.27,25.14,-261.06,-584.84\n199.838,98.75,-25.53,-262.43,-504.25\n250.04,53.24,-44.27,-239.83,-501.08' > Hk.v.T.dat")
#####
Hkdata = np.loadtxt("Hk.v.T.dat", delimiter = ",")
Tlist = Hkdata[:,0]
HkMatrix = Hkdata[:, 1:]*1e-3
tlist = np.array([0.92e-9, 1.02e-9, 1.14e-9, 1.25e-9])#Units of m.
Jslist = np.array([1.15, 1.13, 1.06, 0.94, 0.76])#Mslist for T = 25, 100, 150, 200, 250
I0list = np.array([0.1, 1.9, 2.5, 3, 3.351])*1e-3
####################################################
####################################################
####################################################
####################################################

#This is a general simulation script for magnum.af
#with the goal to unite all physically reasonable simulations
#in one code.

#In the following, initial parameters have to be chosen
#according to which the simulations will be performed.

# Choose here what to simulate                                             
dyn_hyst = False # dynamic hysteresis
stat_hyst = False # semi-dynamic (pointwise relaxation) hysteresis
transfer_curve = False # tranfer curve by relaxation at different external fields
stc_hyst = False # static hysteresis  
sweep = True #External field sweep in one direction 
#!!!

#First, choose one of the ticknesses available to simulate.
#This choice returns the material parameters for different Temperatures
#which are obtained from experimental data.
#Then choose a temperature.
tindex = 0    # Choose index for thickness, 0:= t = 0.92nm etc.
tempindex = 0 # 250°C Choose index for temperature, 0:= T = 25 deg.C.
Jdir = '+x' # Choose current direction: +x, -x, +y, -y

#Here we set the dimensions of Hall Cross. 
#First choose the length and width along x.
#width cross x is the width of stripe along x,
#meaning it sets the thicnkess along y

tiny_scale = 1

# Physical dimensions in [m]
width_cross_x = 10000e-9*tiny_scale#2000e-9
width_cross_y = 10000e-9*tiny_scale#2000e-9
length_cross_x = 10000e-9*tiny_scale#8000e-9
length_cross_y = 10000e-9*tiny_scale#8000e-9
thickness_Ta = 6e-9#6.e-9

x, y, z = np.max([length_cross_x, width_cross_y]), np.max([length_cross_y, width_cross_x]), tlist[tindex]
# Discretization
# Always use dz = z and nz = 1
# for in plane structures where single state domain is expected larger cell sizes can be chosen
# recommended maximal cell size = 10e-9.
# recommended regular cell size = 5e-9.
# recommended minimum cell size = 3e-9.
#WARNING: choose cell size to obtain results in feasible times
#WARNING: pay attention to possible large sizes of vtk files for too small cell sizes
#WARNING: choose cell sizes so, that nx, ny are even numbers (performance boost due to FFT)
dx,dy,dz = 10e-9, 10e-9, z
nx, ny, nz = int(x/dx), int(y/dy), int(1)

#Now we have the relevant parameters and auxiliary functions and arrays
#We can now use the chosen dimensions to calculate the material parameters
#width_crosses = np.array([width_cross_x, width_cross_x, width_cross_y, width_cross_y])
#Arealist = width_crosses*(tlist[tindex]*thickness_Ta) # calculates the area of crossection of the cros
#Note that Tantaulum with thickness 6 nm lays beneath the CoFeB layer. This must be regarded too.

Acrosssection = width_cross_x if 'x' in Jdir else width_cross_y
Acrosssection *= (tlist[tindex] + thickness_Ta)
Jlist = I0list/Acrosssection

#calculate all relevant material parameters for micromagnetics
Hksystem = 0.061 #HkMatrix[tempindex,tindex]
tsystem = 1e-9#tlist[tindex]
Tsystem = Tlist[tempindex]
Jssystem = 1.13 #Jslist[tempindex]
Jesystem = 1e11 #*tiny_scale**2 #Jlist[tempindex]*tiny_scale
K1system = calcK1(Hksystem, Jssystem)
Aexsystem = calcAex(Jssystem)

print("t = %g [m]"%(tsystem))
print("T = %g [C]"%(Tsystem))
print("Hk = %E [mT]"%(Hksystem))
print("Ms = %E [T]"%(Jssystem/mu0))
print("Ku = %E [J/m^3]"%(K1system))
print("Ax = %E []"%(Aexsystem))
print("Area= %E [m^2]\t je[A] = %E"%(Acrosssection,I0list[tempindex]))
print("Je = %E [A/m^2] in %s dir"%(Jesystem, Jdir))

#use help function cross_geometry to create the geometries
#note cross - rxx, ryy, rxy 1d are required to obtain the
#average magnetization in this regions
#One can calculate experimentally relevant results with post processing
#R_AHE = use avg. m in Rxy and plot mx, my, mz
#R_AMR = use avg. m in Rxx, or Ryy and plot mx^2, my^2, mz^2
cross3d = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y)
cross1d = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = False)
crossrxx = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = False, region = 1)
crossryy = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = False, region = 2)
crossrxy = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = False, region = 3)
crossrxx3d = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = True, region = 1)
crossryy3d = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = True, region = 2)
crossrxy3d = cross_geometry(nx, ny, nz, width_cross_x, width_cross_y, make_3d = True, region = 3)
#For test purposes one can always check with paraview if the regions are as desired
testing = False
#testing = True
if testing: 
    Util.write_vti(cross3d, dx, dy, dz, filepath + "Region_Hallcross")
    Util.write_vti(crossrxx, dx, dy, dz, filepath + "Region_rxx")
    Util.write_vti(crossryy, dx, dy, dz, filepath + "Region_ryy")
    Util.write_vti(crossrxy, dx, dy, dz, filepath + "Region_rxy")
    exit()

#set spin orbit torque parameters
#damping and field like coefficients
#based on Jdir we set here the polarization
#and use the geometries to mask the current flow in the desired direction
eta_damp = -0.045
eta_field = -0.06
def get_p_array(Jdir, crossrxx3d=crossrxx3d, crossryy3d=crossryy3d):
    if Jdir == '+x':
        print("using je along +x, p_y = -1")
        #Current flows in +x Direction
        #this results in a polarization p = [0, -1, 0]
        #we multiply with cross_rxx_3d and generate current flow only in x direction
        parray = crossrxx3d * Magnetization.homogeneous(nx, ny, nz, [0., -1., 0.])   
    elif Jdir == '-x':
        print("using je along -x, p_y = +1")
        #Current flows in -x Direction
        #this results in a polarization p = [0, +1, 0]
        #we multiply with cross_rxx_3d and generate current flow only in x direction
        parray = crossrxx3d * Magnetization.homogeneous(nx, ny, nz, [0., 1., 0.])   
    elif Jdir == '+y':
        #Current flows in +y Direction
        #this results in a polarization p = [-1, 0, 0]
        #we multiply with cross_ryy_3d and generate current flow only in y direction
        parray = crossryy3d * Magnetization.homogeneous(nx, ny, nz, [-1., 0., 0.])   
    elif Jdir == '-y':
        #Current flows in -y Direction
        #this results in a polarization p = [+1, 0, 0]
        #we multiply with cross_ryy_3d and generate current flow only in y direction
        parray = crossryy3d * Magnetization.homogeneous(nx, ny, nz, [1., 0., 0.])   
    return parray
parray = get_p_array('+x')

#Use cross1d to mask the regions where magnetic parameters should be applied
Ms_array = cross1d * Jssystem / Constants.mu0
A_array = cross1d * Aexsystem
K_array = cross1d * K1system
Kaxis_array = cross3d * Magnetization.homogeneous(nx, ny, nz, [0, 0, 1])

# Initial magnetization configuration
# For most cases it is recommended to start from a random magnetic configuration
# we use a seeding method to be able to start from same configuration and make systematic comparisons between differnt physical phenomena
# One can also initiate a homogenous magnetization configuration

random = False
homogenous = True
m0list = [0., 0., 1.]
artificial_DW = False
mrandomseed = 101
if random:
    m0 = get_random_m0(nx, ny, nz, cross3d, seed=mrandomseed)
    Util.write_vti(m0, dx, dy, dz, filepath + "m0_seed_%d"%mrandomseed)
elif homogenous:
    m0 = cross3d * Magnetization.homogeneous(nx, ny, nz, m0list)    
#elif artificial_DW: # this is only for Keff > 0.0
    #m0 = cross3d * Magnetization.homogeneous(nx, ny, nz, m0list)    
    #m0[:, :ny/2, :, 2] += -1.0
    #m0[:, ny/2:, :, 2] += 1.0

# We have set everything up sofar. Now, we choose the micromagnetic energetic contributions
# We will then relax the structure first
# Once the structure is relaxed, we can start the dynamic hysteresis simulations

mesh  = Mesh(nx, ny, nz, dx, dy, dz)
state = State(mesh, Ms = Ms_array, m = m0)

#define energy contributions
#Use SparseExchangeField because of Jump Conditions due to Masking
exchange = SparseExchangeField(A_array, mesh)
demag = DemagField(mesh, verbose = True, caching = True, nthreads = 12)
aniso = UniaxialAnisotropyField(K_array, Kaxis_array)
sot = SpinTransferTorqueField(parray, eta_damp, eta_field, Jesystem, z)

llgterms = [exchange, demag, aniso, sot]
llg = LLGIntegrator(alpha = 1.0, terms = llgterms)

def relax_m(relax_time):
    cnt = 0
    stream = open(filepath +"m_relax.dat", "w")
    stream.write("#t(s), mx,my,mz, mx_rxx, my_rxx, mz_rxx, mx_rxy, my_rxy, mz_rxy, mx_ryy, my_ryy, mz_ryy\n")
    state.t = 0.0
    while state.t <= relax_time:
        mx,my,mz = Util.spacial_mean_in_region(state.m, cross1d)
        mx_rxx, my_rxx, mz_rxx = Util.spacial_mean_in_region(state.m, crossrxx)
        mx_rxy, my_rxy, mz_rxy = Util.spacial_mean_in_region(state.m, crossrxy)
        mx_ryy, my_ryy, mz_ryy = Util.spacial_mean_in_region(state.m, crossryy)
        stream.write("%g %g %g %g %g %g %g %g %g %g %g %g %g\n"%(state.t,
            mx,my,mz,
            mx_rxx, my_rxx, mz_rxx,
            mx_rxy, my_rxy, mz_rxy,
            mx_ryy, my_ryy, mz_ryy,
        ))
        print("t = %g \t m_i = [%g %g %g] \t step %g \t status: init.rx "%( round(state.t, 12), 
                                                                                 mx, my, mz, llg.accumulated_steps) 
        + str(0) + "/" + str(0) )
        llg.step(state)
        cnt += 1
    state.write_vti(filepath + "m_relax_%g_%d"%(state.t, llg.accumulated_steps) )
    stream.close()

if dyn_hyst:
    relax_m(10e-9)
    state.t = 0.0
    #Magnetization state was relaxed, now we continue with the dynamic hystereses simulations
    #Note that one shall not increase the field not faster than 200 mT/microsecond
    #Ideally 100 mT/microsecond, but 200mT/microsecond should be fine.
    #See Abert et al. J. Appl. Phys. 116, 123908 (2014);
    #We use an the auxiliary interpolator function from numpy
    #We first create the field profile for the homogenous external field

    Hmax=0.025/Constants.mu0 # Maximal Field Value is chosen to be 50 mT
    period = 250e-9 # period is the time the field needs to go form -bmax to bmax
    simtime = 2.5*period # total simulation time, sim end with the hysteresis
    print("Fieldrate: {:.1f} mT/µs".format(2*Hmax*Constants.mu0*1e3/(simtime1*1e6)))

    Hlist = np.array([0.0,  Hmax, -Hmax, Hmax])
    timelist = np.array([0.0, period/2., 1.5*period, 2.5*period])
    Hext = np.interp(state.t, timelist, Hlist)
    Hextdir = np.array([1.0, 0.0, 0.0])
    zeeswitch = af.constant(0.0, nx, ny, nz, 3, dtype=af.Dtype.f64)
    zee = ExternalField(zeeswitch)
    zee.set_homogeneous_field(0, 0, 0)
    llg = LLGIntegrator(alpha=0.1, terms=llgterms)
    llg.add_terms(zee)
    stream = open(filepath +"m_hyst.dat", "w")
    state.t = 0.0
    index=int(0)
    scalars_every = int(100)
    fields_every = int(10000)
    while state.t < simtime:
        Hext = np.interp(state.t, timelist, Hlist)
        Hextarray = get_hext_arr(Hext, Hextdir)
        zee.set_homogeneous_field(Hextarray[0], Hextarray[1], Hextarray[2])
        if int(llg.accumulated_steps%scalars_every) == int(0):
            mx,my,mz = Util.spacial_mean_in_region(state.m, cross1d)
            mx_rxx, my_rxx, mz_rxx = Util.spacial_mean_in_region(state.m, crossrxx)
            mx_rxy, my_rxy, mz_rxy = Util.spacial_mean_in_region(state.m, crossrxy)
            mx_ryy, my_ryy, mz_ryy = Util.spacial_mean_in_region(state.m, crossryy)
            stream.write("%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n"%(state.t,
                Hextarray[0], Hextarray[1], Hextarray[2],
                mx,my,mz,
                mx_rxx, my_rxx, mz_rxx,
                mx_rxy, my_rxy, mz_rxy,
                mx_ryy, my_ryy, mz_ryy,
            ))
            print("%g %g %g %g"%(state.t,mx,my,mz))
        if int(llg.accumulated_steps%fields_every) == int(0):
            state.write_vti(filepath + "hyst_m_at_%g_mT_stepno_%d"%(Hext*mu0*1e3, llg.accumulated_steps))
        llg.step(state)
        index+=int(1)
    stream.close()

if transfer_curve:
    # Here a magnetization reversal is computed but only at equidistant points with relaxation
    Bmax = 10e-3 # Maximal Field Value in Tesla
    B_step = 1.0e-3 # 
    b_list = np.arange(-Bmax, Bmax+B_step, B_step)
    Hextdir = np.array([1.0, 0.0, 0.0])
    relax_time = 50e-9
    mrandomseed = 77424
    def relax_trans(h, Jdir, relax_time, mrandomseed):
        m0 = get_random_m0(nx, ny, nz, cross3d, mrandomseed)
        parray = get_p_array(Jdir)
        state = State(mesh, Ms = Ms_array, m = m0)
        zeeswitch = af.constant(0.0, nx, ny, nz, 3, dtype=af.Dtype.f64)
        zee = ExternalField(zeeswitch)
        Hextarray = get_hext_arr(h, Hextdir)
        zee.set_homogeneous_field(Hextarray[0], Hextarray[1], Hextarray[2])
        llg = LLGIntegrator(alpha = 1., terms = llgterms)
        llg.add_terms(zee)
        filename = filepath + "relax_J_%s_%g_mT.dat"%(Jdir, h*Constants.mu0*1e3)
        state.t = 0.0
        scalars_every = 10
        with open(filename, 'w') as stream:
            while state.t <= relax_time+1e-11:
                mx, my, mz = Util.spacial_mean_in_region(state.m, cross1d)
                mx_rxx, my_rxx, mz_rxx = Util.spacial_mean_in_region(state.m, crossrxx)
                mx_rxy, my_rxy, mz_rxy = Util.spacial_mean_in_region(state.m, crossrxy)
                mx_ryy, my_ryy, mz_ryy = Util.spacial_mean_in_region(state.m, crossryy)
                stream.write("%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n"
                    %(state.t, Hextarray[0], Hextarray[1], Hextarray[2], 
                        mx, my, mz,
                        mx_rxx, my_rxx, mz_rxx,
                        mx_rxy, my_rxy, mz_rxy,
                        mx_ryy, my_ryy, mz_ryy,
                        ))
                print("%g %g %g %g"%(state.t,mx,my,mz))
                llg.step(state)
            state.write_vti(filepath + "tc_J_%s_m_at_%g_mT"%(Jdir, h*Constants.mu0*1e3))
    for Jdir in ['+x', '-x']:
        for b in b_list:
            relax_trans(b/Constants.mu0, Jdir, 50e-9, mrandomseed)

def sweep(vecHextdir, Hmax, steps, relax_time, Jesystem, action, mi=[0.,0.,1.]):
    '''
    Takes a direction for the sweep in np.array form. i.e Hextdir = np.array([1.0, 0.0, 0.0]) and the max value of the field
    '''
    #relax_m(4e-9)  #Initial relaxation from random state
    #state.t = 0.0   
    listHsweep = np.linspace(-Hmax, Hmax, steps)
    
    Rw = 0
    R2w = 0
    Rwxx = 0
    R2wxx = 0
    
    #zeeswitch = af.constant(0.0, nx, ny, nz, 3, dtype=af.Dtype.f64) #Setup Zeeman interaction within the external field
    #zee = ExternalField(zeeswitch)
    #llg = LLGIntegrator(alpha = 1., terms = llgterms)
    #llg.add_terms(zee)
    
    stream = open(filepath + "%s_%s_%d.dat"%(sys.argv[0], action, Jesystem), "w")
    scalars_every = int(50)
    fields_every = int(1000)

    for h in listHsweep:
        arrayHext = get_hext_arr(h, vecHextdir) #Setting up the external field
        zeeswitch = af.constant(0.0, nx, ny, nz, 3, dtype=af.Dtype.f64)
        zee = ExternalField(zeeswitch)
        zee.set_homogeneous_field(arrayHext[0], arrayHext[1], arrayHext[2])
        steps_sine = 11 #Number of sampling points within a sine cycle: N
        output_name = (filepath, action, h*Constants.mu0*1000)
        os.system("rm %sac_%s_%.2fmT.dat"%output_name)
        os.system("echo 't \t Hext [mT]\t mx my mz \t step \t je [A/m^2]\t sin(w): ' >> %sac_%s_%.2fmT.dat" % output_name)
        os.system("rm %sac_%s_%.2fmT.rx"%output_name)
        os.system("echo 't \t mx my mz \t je [A/m^2] \t' >> %sac_%s_%.2fmT.rx" % output_name)
        for i in np.sin(np.arange(steps_sine+1)/steps_sine*2*np.pi):
            
            #m0 = get_random_m0(nx, ny, nz, cross3d, mrandomseed) #Setting up random state and updated terms
            m0 = cross3d * Magnetization.homogeneous(nx, ny, nz, mi) #It is previously saturated
            #Jdir = '+x' if i >= 0  else '-x'
            #parray = get_p_array(Jdir)
            state = State(mesh, Ms = Ms_array, m = m0)
            sot = SpinTransferTorqueField(parray, eta_damp, eta_field, Jesystem*i, z) #Absolute value to sinewave avoid - * - 
            llgterms = [exchange, demag, aniso, sot, zee]
            llg = LLGIntegrator(alpha = 1. , terms = llgterms)
            state.t = 0.0 
            
            while state.t < relax_time: #Evolving/relaxing the structure 
                
                if int(llg.accumulated_steps%scalars_every) == int(0): #Generating on-screen output 
                    mx,my,mz = Util.spacial_mean_in_region(state.m, cross1d)
                    #HSotx, HSoty, HSotz = Util.spacial_mean_in_region(llg.H_in_Apm(), cross1d)
                    mx_rxx, my_rxx, mz_rxx = Util.spacial_mean_in_region(state.m, crossrxx)
                    mx_rxy, my_rxy, mz_rxy = Util.spacial_mean_in_region(state.m, crossrxy)
                    mx_ryy, my_ryy, mz_ryy = Util.spacial_mean_in_region(state.m, crossryy)
                    output_live = ( round(state.t, 12), h*Constants.mu0*1000, mx, my, mz, llg.accumulated_steps, Jesystem*i, i )
                    print("t = %e \t Hext = %g [mT]\t m_i = [%g %g %g] \t step %g \t je = %g \t sin(n/N): %g" % output_live)
                    output_rx = ( round(state.t, 12), mx, my, mz, Jesystem*i, filepath, action, h*Constants.mu0*1000 )
                    os.system("echo '%e %g %g %g %g ' >> %sac_%s_%.2fmT.rx" % output_rx)
                if int(llg.accumulated_steps%fields_every) == int(0) and False: #Generating vector output for Paraview
                    output_namevti = (filepath, action, h*Constants.mu0*1000, i, llg.accumulated_steps)
                    state.write_vti("%sm_%s_%.2fmT_sin%.2f_stepno_%d"% output_namevti)
                llg.step(state) #Evolving
            output_namevti = (filepath, action, h*Constants.mu0*1000, i, llg.accumulated_steps)
            state.write_vti("%sm_%s_%.2fmT_sin%.2f_stepno_%d"% output_namevti)
            output_dat = output_live + output_name #Extract the final step on a ac_XXXmT.dat file
            os.system("echo ' %g \t %.2f \t  %g %g %g  \t  %g \t  %g \t  %g' >> %sac_%s_%.2fmT.dat" % output_dat) 
            
        stream.write("%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n"%(Jesystem*i,
            h, R2w, 0, 0, 0,
            mx,my,mz,
            Rw, 0, Rwxx, R2wxx,
            mx_rxx, my_rxx, mz_rxx,
            mx_rxy, my_rxy, mz_rxy,
            mx_ryy, my_ryy, mz_ryy
        )) #Output to file with postprocessing-notebook syntax
    
    stream.close()

        
#sweep(np.array([1,0,0]), 0.1/Constants.mu0, 3, 2.5e-9, Jesystem)
#sweep(np.array([1,0,0]), 0.075/Constants.mu0, 7, 4e-9, Jesystem)
sweep(np.array([1,0,0]), 0.020/Constants.mu0, 11, 5e-9, Jesystem, action='sweep_m+', mi=[0.,0.,1.])   
sweep(np.array([1,0,0]), 0.020/Constants.mu0, 11, 5e-9, Jesystem, action='sweep_m-', mi=[0.,0.,-1.])
#relax_m(10e-9)
   
