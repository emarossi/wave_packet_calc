print('loading modules')

import multiprocessing as mp
import cmath,math,time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools
import numexpr as ne
#import cupy as cp
import sys

print('calling python script with')
print(f'file {sys.argv[1]}')
print(f'pol_p {sys.argv[2]} {sys.argv[3]} {sys.argv[4]}')
print(f'pol_d {sys.argv[5]} {sys.argv[6]} {sys.argv[7]}')
print(f'pump_carrier {sys.argv[8]}')
print(f'dump_carrier {sys.argv[9]}')
print(f'bandwidth {sys.argv[10]}')
print(f'grid dimension (#points) {sys.argv[11]}')
print(f'storing result in filename: {sys.argv[12]}')

###############################
#DEFINITION OF INPUT VARIABLES
###############################

file = sys.argv[1]


pol_p = np.array([0,0,0])
pol_d = np.array([0,0,0])

pol_p[0] = sys.argv[2]
pol_p[1] = sys.argv[3]
pol_p[2] = sys.argv[4]
pol_d[0] = sys.argv[5]
pol_d[1] = sys.argv[6]
pol_d[2] = sys.argv[7]

if not pol_p.any() and not pol_d.any():
	print('random over 100000 samples')
	pol_r = np.mean(np.random.uniform(low=-1, high=1, size=(3,100000),axis=1)) #size = (3,#num_samples)
	pol_r_n = pol_r/np.sqrt(np.sum(pol_r**2))
	pol_p = pol_r_n
	pol_d = pol_r_n
	print(pol_r_n,np.sqrt(np.sum(pol_r_n**2)))
else:
	print('NOT random!!!')

print(f'pol_p check {pol_p}')
print(f'pol_d check {pol_d}')

energy_1auE_eV = sp.constants.physical_constants['hartree-electron volt relationship'][0]

pump_carrier = float(sys.argv[8])/energy_1auE_eV #229.73/     #input values in eV->converted to a.u.
dump_carrier = float(sys.argv[9])/energy_1auE_eV #229.73/energy_1auE_eV
bandwidth = float(sys.argv[10])/energy_1auE_eV # 8/energy_1auE_eV            #i

dim = int(sys.argv[11])

outputfilename = sys.argv[12]

path_2P = ''
file_2P = file
          


"""
Module with class TicToc to replicate the functionality of MATLAB's tic and toc.

Documentation: https://pypi.python.org/pypi/pytictoc
"""

__author__       = 'Eric Fields'
__version__      = '1.5.3'
__version_date__ = '2 August 2023'


from timeit import default_timer

class TicToc(object):
    
    """
    Replicate the functionality of MATLAB's tic and toc.
    
    #Methods
    TicToc.tic()       #start or re-start the timer
    TicToc.toc()       #print elapsed time since timer start
    TicToc.tocvalue()  #return floating point value of elapsed time since timer start
    
    #Attributes
    TicToc.start     #Time from timeit.default_timer() when t.tic() was last called
    TicToc.end       #Time from timeit.default_timer() when t.toc() or t.tocvalue() was last called
    TicToc.elapsed   #t.end - t.start; i.e., time elapsed from t.start when t.toc() or t.tocvalue() was last called
    """
    
    def __init__(self):
        """Create instance of TicToc class."""
        self.start   = float('nan')
        self.end     = float('nan')
        self.elapsed = float('nan')
        
    def tic(self):
        """Start the timer."""
        self.start = default_timer()
        
    def toc(self, msg='Elapsed time is', restart=False):
        """
        Report time elapsed since last call to tic().
        
        Optional arguments:
            msg     - String to replace default message of 'Elapsed time is'
            restart - Boolean specifying whether to restart the timer
        """
        self.end     = default_timer()
        self.elapsed = self.end - self.start
        print('%s %f seconds.' % (msg, self.elapsed))
        if restart:
            self.start = default_timer()
        
    def tocvalue(self, restart=False):
        """
        Return time elapsed since last call to tic().
        
        Optional argument:
            restart - Boolean specifying whether to restart the timer
        """
        self.end     = default_timer()
        self.elapsed = self.end - self.start
        if restart:
            self.start = default_timer()
        return self.elapsed
    
    def __enter__(self):
        """Start the timer when using TicToc in a context manager."""
        self.start = default_timer()
    
    def __exit__(self, *args):
        """On exit, pring time elapsed since entering context manager."""
        self.end = default_timer()
        self.elapsed = self.end - self.start
        print('Elapsed time is %f seconds.' % self.elapsed)
        
timer = TicToc()


print('loading constants')

#######################
#READ-OUT QCHEM OUTPUT
#######################

#The code in this section reads the output from Qchem. The following variables are read:
#1-photon properties: GS->VE (A), GS->CE (B) and CE->CE (C), transition energies and transition dipole moments.
#2-photon properties: GS->GS (REXS) and GS->VE (RIXS) transition moments and relative frequency grids.

# load constants
#Definition of unit conversion constants

time_1aut_s = sp.constants.physical_constants['atomic unit of time'][0]
planck_eV = sp.constants.physical_constants['reduced Planck constant in eV s'][0]
epsilon_0_au = sp.constants.epsilon_0/sp.constants.physical_constants['atomic unit of permittivity'][0]

def RIXS_TM_mod(row):
    
    if 'A --> B' in row or 'B --> A' in row:
        temp_a = row.split('     ')[1].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
        temp_b = row.split('     ')[2].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
        temp_c = row.split('     ')[3].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
        
    else:
        temp_a = row.split('     ')[2].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
        temp_b = row.split('     ')[3].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
        temp_c = row.split('     ')[4].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
    
    if '+-' in temp_a:
        temp_a = temp_a.replace('+-','-')

    if '+-' in temp_b:
        temp_b = temp_b.replace('+-','-')

    if '+-' in temp_c:
        temp_c = temp_c.replace('+-','-')
            
    return([complex(temp_a),complex(temp_b),complex(temp_c)])

def REXS_TM_mod(row):
    
    temp_a = row.split('     ')[2].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
    temp_b = row.split('     ')[3].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
    temp_c = row.split('     ')[4].replace(' i (','').replace(')','').replace(' ','') + 'j'.strip()
    
    if '+-' in temp_a:
        temp_a = temp_a.replace('+-','-')

    if '+-' in temp_b:
        temp_b = temp_b.replace('+-','-')

    if '+-' in temp_c:
        temp_c = temp_c.replace('+-','-')
        
    return([complex(temp_a),complex(temp_b),complex(temp_c)])

def dipole_moment_processing(vector_string):

    #This function imports the three components of the transition dipole and assembles them in a numpy array
    #The dot product between the polarization vector and the transition dipole vector is given in output.
    #The absolute value of the dot product is considered, since the sign is random
    
    x = float(vector_string.split('(')[1].split(',')[0].replace('X','').strip())
    y = float(vector_string.split('(')[1].split(',')[1].replace('Y','').strip())
    z = float(vector_string.split('(')[1].split(',')[2].replace('Z','').replace(')','').strip())

    return(list([x,y,z]))



#Importing all the data from Qchem's output files

label_list = ['GS']

num_val_states = 1    #ground + #valence-excited states considered in the transition
num_core_states = 0

#1-photon: checkpoint variables
block_A = False
block_B = False
block_C = False
block_AB = False

#1-photon: list initialisation
A_en = []
A_dip_AB = []; A_dip_BA = []
B_en = []
B_dip_AB = []; B_dip_BA = []
C_en = []
C_dip_AB = []; C_dip_BA = []

#2-photon: checkpoint variables
val_out = False  #Output of data relative to a couple of states
RIXS_out = False   #Output of RIXS tensor (relative to a couple of frequency points)
REXS_out = False   #Output of REXS tensor (relative to an omega_p point)

#2-photon: list initialisation
index_ab = [(0,0)]  #list of tuples with indeces of A and B states involved in A->B. Initialised with (0,0) for REXS.
AB = []; BA = []; AA = []  #lists of RIXS AB, BA and REXS AA tensors
pump_l = []; dump_l = []   #lists storing the pump and dump grid parameters

with open(path_2P+file_2P,'r') as f:    #Sulfur L1-Edge
# with open(path_2P+'OCS-SRIXS-OK12V-40r-0_027s_HF.out','r') as f:     #Oxygen K-Edge

    for count, line in enumerate(f):
        
        #Number and labels of the states involved
        
        if 'EOMEE transition' in line and 'CVS' not in line:
            label_list.append('VE-'+line.split(' ')[3].strip())
            num_val_states += 1
            
        elif 'CVS-EOMEE transition' in line:
            label_list.append('CE-'+line.split(' ')[3].strip())
            num_core_states += 1

        #####################
        #1-photon properties#
        #####################
        
        #BLOCKS A+B: Finding out where the blocks of the matrix output are printed out
        if ('State A: eomee_ccsd/rhfref/singlets:' in line and 'cvs' not in line) or 'State A: ccsd:' in line:
            index_a_temp = int(line.split(':')[2].split('/')[0])
            block_AB = True
        
        #BLOCK A: GS+valence_excited <-> GS+valence_excited; transition energies and dipole moments
        
        #Switch
        if block_AB == True and 'State B: eomee_ccsd/rhfref/singlets:' in line:
            index_ab.append((index_a_temp,int(line.split(':')[2].split('/')[0])))
            index_A = count
            block_A = True
            
        #Data aquisition
        if block_A == True and count in range(index_A+1,index_A+5):
            
            if count == index_A+1:
                A_en.append(float(line.split('=')[2].strip().replace('eV','').strip()))
                
            if count == index_A+3:
                A_dip_AB.append(dipole_moment_processing(line))
                
            if count == index_A+4:
                A_dip_BA.append(dipole_moment_processing(line))
                block_A = False
                block_AB = False
        
        # #BLOCK B: GS+valence_excited <-> core_excited; transition energies and dipole moments
        
        #Switch
        if block_AB == True and 'State B: cvs_eomee_ccsd/rhfref/singlets:' in line:
            
            index_B = count
            block_B = True
            block_AB = False
            
        #Data aquisition
        if block_B == True and count in range(index_B+1,index_B+5):
            
            if count == index_B+1:
                B_en.append(float(line.split('=')[2].strip().replace('eV','').strip()))
                
            if count == index_B+3:
                B_dip_AB.append(dipole_moment_processing(line))
                
            if count == index_B+4:
                B_dip_BA.append(dipole_moment_processing(line))
                block_B = False
                block_AB = False
        
        # #BLOCK C: core_excited <-> core_excited; transition energies and dipole moments
        
        #Switch
        if 'State A: cvs_eomee_ccsd/rhfref/singlets:' in line:
            index_C = count
            block_C = True
        #Data aquisition
        if block_C == True and count in range(index_C+2,index_C+6):
                      
            if count == index_C+2:
                C_en.append(float(line.split('=')[2].strip().replace('eV','').strip()))
              
            if count == index_C+4:
                C_dip_AB.append(dipole_moment_processing(line))
              
            if count == index_C+5:
                C_dip_BA.append(dipole_moment_processing(line))
                block_C = False
        
        #####################
        #2-photon properties#
        #####################
        
        #REXS OUTPUT
        
        if 'CCSD REXS Moments M_IJ, I,J=X,Y,Z (a.u.):' in line:
            REXS_out = True
            index_REXS = count
            
        if 'REXS Scattering Strength Tensor S (a.u.):' in line:
            REXS_out = False
        
        if REXS_out == True and count in range(index_REXS+1,index_REXS+4):
            AA.append(REXS_TM_mod(line))
        
#         #VALENCE STATE TRANSITION PROPERTIES
#         #Storing the couple of A,B indeces 
        
#         if 'State A: eomee_ccsd/rhfref/singlets:' in line or 'State A: ccsd:' in line:
#             index_a_temp = int(line.split(':')[2].split('/')[0])
        
#         if 'State B: eomee_ccsd/rhfref/singlets:' in line:
#             index_ab.append((index_a_temp,int(line.split(':')[2].split('/')[0])))
#             val_out = True
#             index_state = count
            
        #RIXS OUTPUT
            
        if 'RIXS Moments M_IJ (A-->B), I,J=X,Y,Z (a.u.):' in line:
            RIXS_out = True
            index_RIXS = count
            
        if 'RIXS Scattering Strength Tensor S (a.u.):' in line:
            RIXS_out = False
            
        if RIXS_out == True and count in range(index_RIXS+1,index_RIXS+4):
            AB.append(RIXS_TM_mod(line))
            
        if RIXS_out == True and count in range(index_RIXS+5,index_RIXS+8):
            BA.append(RIXS_TM_mod(line))
            
        #GRID OUTPUT
            
        if 'Absorbed photon' in line and len(index_ab) == 2:
            pump_l.append(float(line.split('=')[1].replace('a.u.','').strip()))
            
        if 'Emitted photon' in line and len(index_ab) == 2:
            dump_l.append(float(line.split('=')[1].replace('a.u.','').strip()))

###########################################################
#1-PHOTON - ASSEMBLING TRANSITION ENERGY AND DIPOLE ARRAYS
###########################################################

#A BLOCK: shape=(#valence_excited,#valence_excited)

A_dip_array = np.zeros((num_val_states,num_val_states,3))
A_en_array = np.zeros((num_val_states,num_val_states))

#For calculations considering all intrastate transitions
# A_dip_array[np.triu_indices(num_val_states,k=1)]=(np.asmatrix(A_dip_AB)+np.asmatrix(A_dip_BA))/2  #for calculations with all intrastate transitions
# A_en_array[np.triu_indices(num_val_states,k=1)] = np.array(A_en)  #for calculations with all intrastate transitions

#For calculations considering only GS-->ES transitions
A_dip_array[0,np.arange(1,num_val_states,1),:]=(np.asmatrix(A_dip_AB)+np.asmatrix(A_dip_BA))/2   #for calculation with only GS->ES transitions
A_en_array[0,np.arange(1,num_val_states,1)] = np.array(A_en)   #for calculation with only GS->ES transitions

A_dip_array += np.transpose(A_dip_array,axes=[1,0,2])
A_en_array -= np.transpose(A_en_array)

#B BLOCK: shape=(#valence_excited,#core_excited)

B_dip_array = np.zeros((num_val_states,num_core_states,3))
B_en_array = np.zeros((num_val_states,num_core_states))

#For calculations considering all intrastate transitions
# B_dip_array = (np.array(B_dip_AB).reshape((num_val_states,num_core_states,3))+np.array(B_dip_BA).reshape((num_val_states,num_core_states,3)))/2
# B_en_array = np.array(B_en).reshape((num_val_states,num_core_states))

#For calculations considering only GS-->ES transitions
print(np.asmatrix(B_dip_AB).shape)
B_dip_array[0,np.arange(0,num_core_states,1),:] = (np.asmatrix(B_dip_AB)+np.asmatrix(B_dip_BA))/2   
B_en_array[0,np.arange(0,num_core_states,1)] = np.array(B_en)

#C BLOCK: shape=(#core_excited,#core_excited)

C_dip_array = np.zeros((num_core_states,num_core_states,3))
C_en_array = np.zeros((num_core_states,num_core_states))

#For calculations considering all intrastate transitions

C_dip_array[np.triu_indices(num_core_states,k=1)]=(np.asmatrix(C_dip_AB)+np.asmatrix(C_dip_BA))/2
C_dip_array += np.transpose(C_dip_array,axes=[1,0,2])

C_en_array[np.triu_indices(num_core_states,k=1)] = np.array(C_en)
C_en_array -= np.transpose(C_en_array)

#Assembling the energy array and the dipole arrays =>((A,B),(B.t,C))
#Energy array shape=(#valence_excited+#core_excited,#valence_excited+#core_excited)
#Dipole array shape=(#valence_excited+#core_excited,#valence_excited+#core_excited,3)

en_array = np.block([[A_en_array,B_en_array],[-np.transpose(B_en_array),C_en_array]])/energy_1auE_eV
dip_array = (np.concatenate((np.concatenate((A_dip_array,B_dip_array),axis=1),np.concatenate((np.transpose(B_dip_array,axes=[1,0,2]),C_dip_array),axis=1)),axis=0))

###########################################
#2-PHOTON - BUILDING REXS and RIXS TENSORS
###########################################

#Interpolation: the function takes the REXS/RIXS tensors from Qchem and interpolates them.
#The interpolated function is used to generate a tensor on a new grid defined by 'new_pump_array'.

def RIXS_interpolation(RIXS_list,pump_list,new_pump_array,process):

    global index_ab  #importing the index_ab

    #REXS process: the tensor has a shape (#omega_p,3,3). Interpolation along the axis=0.
    if process == 'REXS':
        real_part = sp.interpolate.CubicSpline(np.array(pump_list),np.array(RIXS_list,dtype='complex64').reshape((len(pump_list),3,3)).real,axis=0)
        imag_part = sp.interpolate.CubicSpline(np.array(pump_list),np.array(RIXS_list,dtype='complex64').reshape((len(pump_list),3,3)).imag,axis=0)
    
    #RIXS process: the tensor has a shape (#VE_states,#omega_p,3,3). Interpolation along the axis=1.
    elif process == 'RIXS':
        real_part = sp.interpolate.CubicSpline(np.array(pump_list),np.array(RIXS_list,dtype='complex64').reshape((len(index_ab)-1,len(pump_l),3,3)).real,axis=1)
        imag_part = sp.interpolate.CubicSpline(np.array(pump_list),np.array(RIXS_list,dtype='complex64').reshape((len(index_ab)-1,len(pump_l),3,3)).imag,axis=1)

    #Putting together the real and imaginary parts of the tensors.
    return(np.vectorize(complex)(real_part(new_pump_array),imag_part(new_pump_array)))

#The REXS arrays are broadcasted to => (#omega_p,#omega_p,3,3)
#The RIXS arrays are broadcasted to => (#AB_transitions,#omega_p,#omega_p,3,3)
#The pump_grid has shape => (#omega_p,#omega_p). The dump grid is pump_grid only transposed.

#Defintion of the pump array to be used for calculations (i.e. more points than the Qchem's output)
pump_array = np.linspace(min(pump_l),max(pump_l),dim)

#REXS tensor - output of the new REXS vector from the interpolation function and broadcast
AA_a = np.broadcast_to(RIXS_interpolation(AA,pump_l,pump_array,'REXS'),(dim,dim,3,3))

#RIXS tensor - output of the new RIXS tensor from the interpolation function, broadcast and averaging

#right and left calculation
AB_a = np.broadcast_to(RIXS_interpolation(AB,pump_l,pump_array,'RIXS')[:,np.newaxis,:,:,:],(len(index_ab)-1,dim,dim,3,3))
BA_a = np.broadcast_to(RIXS_interpolation(BA,pump_l,pump_array,'RIXS')[:,np.newaxis,:,:,:],(len(index_ab)-1,dim,dim,3,3))

#averaged calculation (averaging between right and left states)
AB_avg_a = (np.array(AB_a)+np.conjugate(np.array(BA_a)))/2

#Creation of the pump grid starting from the pump array
pump_grid = np.broadcast_to(pump_array,(dim,dim))
dump_grid = pump_grid.transpose()

#For 2D calculations - temporary
# AA_a = np.broadcast_to(np.array(AA).reshape((math.isqrt(len(pump_l)),3,3)),(math.isqrt(len(pump_l)),math.isqrt(len(pump_l)),3,3))
# AB_a = np.array(AB).reshape((len(index_ab)-1,math.isqrt(len(pump_l)),math.isqrt(len(pump_l)),3,3))
# BA_a = np.array(BA).reshape((len(index_ab)-1,math.isqrt(len(pump_l)),math.isqrt(len(pump_l)),3,3))
# pump_grid = np.array(pump_l).reshape((math.isqrt(len(pump_l)),math.isqrt(len(pump_l))))
# dump_grid = np.array(dump_l).reshape((math.isqrt(len(pump_l)),math.isqrt(len(pump_l))))

# RIXS_TM = np.zeros((num_val_states,num_val_states,math.isqrt(len(pump_l)),math.isqrt(len(pump_l)),3,3),dtype=complex) #2D calculation

#Building the RIXS tensor containing all the RIXS and REXS tensors
#RIXS_TM.shape = (state_A,state_B,#omega_p,#omega_d,3,3)

#Putting together the full RIXS tensor
#For 1D calculations

RIXS_TM = np.zeros((num_val_states,num_val_states,dim,dim,3,3),dtype='complex64')

#Right and left tensors
# for index,(row,col) in enumerate(index_ab):
    
#     if row == col:
#         RIXS_TM[row,col,...] = AA_a
#     else:
#         RIXS_TM[row,col,...] = AB_a[index-1,...]
        
#         RIXS_TM[col,row,...] = BA_a[index-1,...]

#Averaged tensors
#Only the RIXS excitations from the GS (index=0) to the VE (index>0) are considered.

for index,(row,col) in enumerate(index_ab):
    
    if row == col:
        RIXS_TM[row,col,...] = AA_a
    else:
        RIXS_TM[0,index,...] = AB_avg_a[index-1,...]
        
        RIXS_TM[index,0,...] = np.conjugate(AB_avg_a[index-1,...])

##################
#PULSE DEFINITION 
##################

print('RIXS read')

#Definition of Gaussian functions in the frequency domain

def gauss_freq_1D(omega,omega_carrier,time_shift,sigma,amplitude,pol_v):
    shift_factor = np.exp(complex(0,1)*(omega)*time_shift)
    envelope = np.exp(-(((omega-omega_carrier)**2)/(2*sigma**2)))
    return(np.einsum('x,f->xf',pol_v,amplitude*shift_factor*envelope))

def gauss_freq_2D(omega,omega_carrier,time_shift,sigma,amplitude,pol_v):
    shift_factor = np.exp(complex(0,1)*(omega)*time_shift)
    envelope = np.exp(-(((omega-omega_carrier)**2)/(2*sigma**2)))
    return(np.einsum('x,pd->xpd',pol_v,amplitude*shift_factor*envelope))


#The pulse is defined as E(\omega)=|E(\omega)|e^{i\phi(\omega)}.
#E(\omega) and E(\omega)* differ in the sign of the exponential. The phase \phi(\omega) is set to 0 in testing.

#DEFINITION OF |E_0| FROM IRRADIANCE
#The Irradiance definition is: I=1/2*c*epsilon_0*|E|
#Thus, |E_0|=sqrt(2I/(epsilon_0*c))
#1 a.u. of Irradiance is equal to 3.51e16 W/cm^2 (from https://onlinelibrary.wiley.com/doi/pdf/10.1002/3527605606.app9)
#Epsilon_0 in a.u. is obtained by dividing its value in F/m by the unit of a.u. of permittivity in F/m.
#The formula for the electric field is applied, obtaining the value of |E_0| in a.u.

#FOURIER TRANSFORM CONVENTION
#The convection used is the non-unitary one:
#F(\omega) = \int_{-\infty}^{\infty}dt f(t)e^{-i\omega t}
#f(t) = 1/(2\pi)\int_{-\infty}^{\infty}dt F(\omega)e^{i\omega t}

attenuation = 1e-2
irradiance_W_cm2_au = 3.51e16 #(W/cm^2)/a.u. 
irradiance_p = (1e18*attenuation)/irradiance_W_cm2_au
irradiance_d = (1e18*attenuation)/irradiance_W_cm2_au

E_0_p = math.sqrt((2*irradiance_p)/(epsilon_0_au*137))
E_0_d = math.sqrt((2*irradiance_d)/(epsilon_0_au*137))

#Definition of the pulse parameters
sigma_f = bandwidth/2.355

#Definition of the gaussian pulse-time domain
begin = 2000e-18/time_1aut_s #attoseconds
end = 4000e-18/time_1aut_s
step_time = 1e-18/time_1aut_s

#Calculation of the duration (attoseconds) - transform limited pulse
duration = (0.441*planck_eV)/(bandwidth*energy_1auE_eV)
sigma_t = (duration/2.355)/time_1aut_s
print('Pulse duration: %f as'%(duration/1e-18))

pump_time_shift = 0e-18/time_1aut_s
dump_time_shift = 0e-18/time_1aut_s

time_array = np.arange(-begin,end,step_time)
pump_time = np.einsum('x,t->xt',pol_p,E_0_p*(sigma_f*math.sqrt(2*math.pi))*np.exp(-0.5*(((time_array-pump_time_shift)**2)*(sigma_f**2)))*np.exp(-complex(0,1)*pump_carrier*time_array))
dump_time = np.einsum('x,t->xt',pol_d,E_0_d*(sigma_f*math.sqrt(2*math.pi))*np.exp(-0.5*(((time_array-dump_time_shift)**2)*(sigma_f**2)))*np.exp(-complex(0,1)*dump_carrier*time_array))
pulse_time = pump_time+dump_time

#Definition of the gaussian pulse-frequency domain

#1-photon frequency array
freq_array = pump_grid[0]
step_freq = pump_grid[1][1]-pump_grid[1][0]  #defining the grid step size

#Linearly polarized
pulse_1P = (gauss_freq_1D(freq_array,pump_carrier,pump_time_shift,sigma_f,E_0_p,pol_p)
            +gauss_freq_1D(freq_array,dump_carrier,dump_time_shift,sigma_f,E_0_d,pol_d)).astype('complex64')

#Circularly polarized - dump shifted by pi/2 with respect to pump
#pulse_1P = (gauss_freq_1D(freq_array,pump_carrier,pump_time_shift,sigma_f,E_0_p,pol_p)
#            +gauss_freq_1D(freq_array,dump_carrier+np.pi/(2*dump_carrier),dump_time_shift,sigma_f,E_0_d,pol_d)).astype('complex64')

#2-photon frequency grid

#Linearly polarized
pump_freq = gauss_freq_2D(pump_grid,pump_carrier,pump_time_shift,sigma_f,E_0_p,pol_p)+gauss_freq_2D(pump_grid,dump_carrier,dump_time_shift,sigma_f,E_0_d,pol_d)
dump_freq = gauss_freq_2D(dump_grid,pump_carrier,pump_time_shift,sigma_f,E_0_p,pol_p)+gauss_freq_2D(dump_grid,dump_carrier,dump_time_shift,sigma_f,E_0_d,pol_d)

#Circularly polarized - dump shifted by pi/2 with respect to pump
#pump_freq = gauss_freq_2D(pump_grid,pump_carrier,pump_time_shift,sigma_f,E_0_p,pol_p)+gauss_freq_2D(pump_grid,dump_carrier+np.pi/(2*dump_carrier),dump_time_shift,sigma_f,E_0_d,pol_d)
#dump_freq = gauss_freq_2D(dump_grid,pump_carrier,pump_time_shift,sigma_f,E_0_p,pol_p)+gauss_freq_2D(dump_grid,dump_carrier+np.pi/(2*dump_carrier),dump_time_shift,sigma_f,E_0_d,pol_d)

pulse_matrix = np.einsum('xpd,ypd->xypd',pump_freq,np.conjugate(dump_freq)).astype('complex64')

#fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,8))
#ax[0].plot(time_array*time_1aut_s/1e-15,pulse_time[0,:])
#ax[0].set_xlabel('Time (fs)')
#ax[0].minorticks_on()

# ax[1].plot(time_array*time_1aut_s/1e-15,pulse_time[1,:])

# ax[1].plot(freq_array,pulse_1P[2,:])

#spectrum = ax[1].imshow(pulse_matrix[0,0,...].real,origin='lower',extent=[np.min(pump_grid)*energy_1auE_eV,np.max(pump_grid)*energy_1auE_eV,np.min(dump_grid)*energy_1auE_eV,np.max(dump_grid)*energy_1auE_eV],aspect="auto")
#plt.colorbar(spectrum)
#plt.show()

#####################
#XAS WP COEFFICIENTS
#####################

#Definition of the integrand functions

if 'OK' in file:
	decay_rate = 0.005
elif 'SL1' in file:
	decay_rate = 0.0039
elif 'NK' in file:
	decay_rate = 0.0049

lifetime = ((1/decay_rate)*time_1aut_s)/1e-15
print('Decay rate: %f a.u.'%decay_rate)
print('Lifetime: %f fs'%lifetime)

def integrand(pulse,frequency,energy,dipole,gamma_m,time):
    
    numerator = np.exp(complex(0,1)*(energy-frequency-complex(0,1)*(gamma_m/2))*time)
    denominator = energy-frequency-(complex(0,1)*(gamma_m/2))

    #introduced the dot product between the vectorial pulse and the dipole moment
    return(np.einsum('xf,x,tf->tf',pulse,dipole,numerator/denominator,optimize='optimal'))

                       
def integrand_cc(pulse,frequency,energy,dipole,gamma_m,time_grid):
    
    numerator = np.exp(-complex(0,1)*(energy-frequency+complex(0,1)*(gamma_m/2))*time)
    denominator = energy-frequency+(complex(0,1)*(gamma_m/2))
    
    return(pulse*dipole*(numerator))#/denominator))

#Definition of the function for the transition from interaction to Schroedinger picture
#With 'sign' the transition factor for the bra or ket expansions can be controlled

def schroedinger_p(time,energy,gamma_m):
    return(np.exp(-complex(0,1)*(energy-complex(0,1)*(gamma_m/2))*time))
#     return(np.exp(-complex(0,1)*energy*time_array))


#Initialisation of the arrays of coefficients

c = np.zeros((num_val_states+num_core_states,len(time_array)),dtype=complex)
c_cc = np.zeros((num_val_states+num_core_states,len(time_array)),dtype=complex)

c[0,:]+=1


#At each time step, the pulse envelope is used in the formula.

freq_grid,time_grid = np.meshgrid(freq_array,time_array)

#The integrand is calculated on a grid with frequency on the x-axis and time on the y-axis.
#The time and frequency meshgrids are input to the 'integrand' function: the integrand is calculated on the grid.
#Integrating over the frequency axis, the coefficients as a function of time are obtained.
#The t-dependent coefficients for each state are stored in an array of shape (#states,length_time_array)

for state in range(num_val_states,num_val_states+num_core_states):
    c[state,:] += schroedinger_p(time_array,en_array[0,state],decay_rate)*((1/(math.sqrt(2*np.pi)))*sp.integrate.trapz(integrand(pulse_1P,freq_grid,en_array[0,state],dip_array[0,state],decay_rate,time_grid),dx=step_freq,axis=1))

#######################
#SRIXS WP COEFFICIENTS
#######################


#Definition of the pre-factor function
#The function uses the numexpr package for more efficient evaluation of the vectorial functions

def prefactor_red(energy_eq,time,pulse_mom,gamma_k):
    numerator = ne.evaluate('exp((complex(0,1)*energy_eq+(gamma_k/2))*time)')
    denominator = ne.evaluate('energy_eq-complex(0,1)*(gamma_k/2)')
    integrand = ne.evaluate('(numerator/denominator)*pulse_mom')
    return integrand

# def prefactor_red_cc(energy_eq,time,pulse_mom,gamma_k):
#     numerator = ne.evaluate('exp((-complex(0,1)*energy_eq+(gamma_k/2))*time)')
#     denominator = ne.evaluate('energy_eq+complex(0,1)*(gamma_k/2)')
#     return(ne.evaluate('(numerator/denominator)*pulse_mom'))


#Definition of the function to calculate the wp coefficients with the strip method

def wp_calc_opt(time_array, index, low, up, f, f_prime, resonance):
    
    time_begin = time_array[0]
    time_step = np.unique(np.diff(time_array))  #np.unique->finds unique elements of array; np.diff->calculates discrete difference between elements
    if time_step.size != 1:
    #    print('time_step=',time_step)
    #    print('PLEASE CHECK: timesteps are not equally spaced (for this code they must be) using', time_step[0])
        time_step = time_step[0]
    
    energy = energy_eq[0,index,...]
    pulse = pulse_mom[0,index,...]
    gamma_k = 0
    
    #COMPUTATION OF THE INTEGRAND
    #The integrand is split in two:
    #Initial term at t_0 containing the fraction and pulse_mom -> integrand_full_time_begin
    #Iterative term containing the exponential depending of delta_t -> integrand_timestep_factor
    #Big calculation outside the for, inside the for simple multiplication.

    # compute the integrand for time_begin
    numerator_m = ne.evaluate('exp((complex(0,1)*energy+(gamma_k/2))*time_begin)')
    denominator_m = ne.evaluate('energy-complex(0,1)*(gamma_k/2)')
    integrand_full_time_begin = ne.evaluate('(numerator_m/denominator_m)*pulse')
    integrand_timestep_factor = ne.evaluate('exp((complex(0,1)*energy+(gamma_k/2))*time_step)')

    result = np.zeros((time_array.size),dtype=np.complex128)
    for nr, time in enumerate(time_array):
        
        if nr == 0: # time begin computation:
            integrand_full = integrand_full_time_begin
        else: # for all other times
            integrand_full = np.multiply(integrand_full, integrand_timestep_factor)             
    
        ### integrate
        if index == 0: #REXS
            integrand_full[resonance[0],resonance[1]] = 0   #integrand=0 on the strip (row, col strip coords array)
        
        correction = (-np.sum(((sp.integrate.trapz(integrand_full[low[0],low[1]]*step_size/2, dx=step_size)*step_size)/2)
            +((sp.integrate.trapz(integrand_full[up[0],up[1]]*step_size/2, dx=step_size)*step_size)/2))
            +f_prime*(math.sin(delta*time)/(delta*time))+complex(0,1)*(2*sp.special.sici(delta*time)[0]+np.pi)*f)
    
        integral_0 = step_size*step_size*np.sum(integrand_full)   #0th order integration to save computational time
        
        result[nr] = (1/(2*np.pi))*(integral_0 + correction)
        
    return result
    
#def wp_calc_opt_gpu(time_array):
#    
#    time_begin = time_array[0]
#    time_step = np.unique(np.diff(time_array))
#    if time_step.size != 1:
#        print('time_step=',time_step)
#        print('error: timesteps are not equally spaced (for this code they must be)')
#        time_step = time_step[0]
#    
#    # variables copied from prefactor_red input
#    energy_eq_m = energy_eq[0,index,...]
#    pulse_mom_m = pulse_mom[0,index,...]
#    gamma_k = 0
#    
#    ## compute the integrand for time_begin
#    numerator_m = ne.evaluate('exp((complex(0,1)*energy_eq_m+(gamma_k/2))*time_begin)')
#    denominator_m = ne.evaluate('energy_eq_m-complex(0,1)*(gamma_k/2)')
#    
#    # make as gpu array
#    gpu_integrand_full_time_begin = cp.array(ne.evaluate('(numerator_m/denominator_m)*pulse_mom_m'))
#
#    # make as gpu array
#    gpu_integrand_timestep_factor = cp.array(ne.evaluate('exp((complex(0,1)*energy_eq_m+(gamma_k/2))*time_step)'))
#    
#    for nr, time in enumerate(time_array):
#        
#        if nr == 0: # time begin computation:
#            gpu_integrand_full = gpu_integrand_full_time_begin
#        else: # for all other times
#            gpu_integrand_full = cp.multiply(gpu_integrand_full, gpu_integrand_timestep_factor)             
#
#
#        ### integrate
#        if index == 0:
#            gpu_integrand_full[resonance_strip[0],resonance_strip[1]] = 0
#        
#        integral_0 = cp.asnumpy(step_size*step_size*cp.sum(gpu_integrand_full))
#        
#        integrand_full = cp.asnumpy(gpu_integrand_full)
#        
#        correction.append(-np.sum(((sp.integrate.trapz(integrand_full[low_nonull[0],low_nonull[1]]*step_size/2, dx=step_size)*step_size)/2)
#                                  +((sp.integrate.trapz(integrand_full[up_nonull[0],up_nonull[1]]*step_size/2, dx=step_size)*step_size)/2))
#                                  +f_prime_x0*(math.sin(delta*time)/(delta*time))+complex(0,1)*(2*sp.special.sici(delta*time)[0]+np.pi)*f_x0)       
#    
#        
#        wp_coeff_g_nostrip.append(integral_0)
#        
#    result = np.array(wp_coeff_g_nostrip)+np.array(correction)   
#    
#    return result
#    # return(np.array(wp_coeff_g_nostrip+correction),np.array(wp_coeff_g_nostrip_cc+correction_cc))
  

grid_dim = pump_grid.shape[0]

#Calculation of pulse_mom and energy_eq tensors 

print(en_array.shape)

pulse_mom = np.einsum('ijpdxy,xypd->ijpd',RIXS_TM,pulse_matrix,optimize='optimal')
energy_eq = np.broadcast_to(en_array[0:num_val_states,0:num_val_states,np.newaxis,np.newaxis],pulse_mom.shape)-np.broadcast_to(pump_grid[np.newaxis,np.newaxis,...],pulse_mom.shape)+np.broadcast_to(dump_grid[np.newaxis,np.newaxis,...],pulse_mom.shape)

delta = 2.25e-2  #doesn't work with 1e-2
# delta = 2e-2
step_size = pump_grid[1][1]-pump_grid[1][0]  #defining the grid step size




# uses global variable: grid_dim, delta, energy_eq, pulse_mom
def compute_strip_stats(index):
    #Resonance condition array. Points where the resonance condition is satisfied have value 0.
    resonance = energy_eq[0,index,...].round(3)
    #Coordinates of strip points within a +-delta from the center. shape = (coordinate,#points) => row=0, col=1
    resonance_strip = np.asarray(np.where((resonance<delta) & (resonance>-delta)))

    #Points defining the center of the resonance strip. shape = (coordinate,#points) => row=0, col=1
    resonance_center = np.asarray(np.where(resonance == 0))
    #DEFINITION OF THE PARAMETERS OF THE STRIP'S LIMITS. 
    #Strip array defined as the internal loop over x coordinates and the external one over the y coordinates 

    #The initial delta value is diminished by 1 since the loop overcounts
    for strip_point in range(0,len(resonance_strip[0])):
        #Determining the column of the initial point of the lower and upper strip's limits. 
        #This coordinate corresponds to the last column of the strip corresponding to row=0
        if resonance_strip[0][strip_point] == 0 and resonance_strip[0][strip_point+1] == 1:
            col_initial = resonance_strip[1][strip_point]
        #Determining the row of the final point (i.e. last column) of the upper strip's limit.
        #This coordinate corresponds to row of the last strip point of the grid on the last column
        if resonance_strip[1][strip_point] == grid_dim-1:
            row_final_up = resonance_strip[0][strip_point]    #update iteratively until the last point is reached
        if resonance_strip[0][strip_point] == grid_dim-1 and resonance_strip[0][strip_point-1] == grid_dim-2:
            col_final_up = resonance_strip[1][strip_point]
        
    #Determining the delta parameter (in #pixels). This corresponds to the row_max-row_min row=row_initial
    #The column coordinates of these points are stored in the list delta_l. The max of this list is taken (the min is 0).
    delta_l = []
    for strip_point in range(0,len(resonance_strip[0])):
        if resonance_strip[1][strip_point] == col_initial:
            delta_l.append(resonance_strip[0][strip_point])
                    
    delta_2 = max(delta_l)
    row_final_low = row_final_up-delta_2

    #DEFINITION OF THE STRIP'S LIMITS: both an 'internal' (i.e. considering the last value of the strip with value = 0) and an 'external' limit (i.e. first values outside the strip with values !=0)
    #4 (2 'lower' limit and 2 'upper' limits), 2D arrays will be created (x_coord,y_coord)
    #The limits of the strips are defined by the x_initial, y_final_up and delta_2 points. All the other points are deduced by construction.
    #for the non-zero strip, the upper and lower limit are mismatched since the grid curtails the non-zero lower limit.
    #To account for this mismatch between upper and lower limits, the upper limit is calculated by considering the the x+1,y+1 starting coordinate (instead of the x,y+1).


    if index == 0:
        #Lower limit
        low_null = np.vstack((np.arange(0,row_final_low+1,1),np.arange(col_initial,col_final_up+1,1)))
        low_nonull = np.vstack((np.arange(0,row_final_low,1),np.arange(col_initial+1,col_final_up+1,1)))
        #Upper limit (to consider the y+1 initial point, the delta+2 point is considered since the y reference is 0)
        up_null = np.vstack((np.arange(delta_2,row_final_up+1,1),np.arange(col_initial,col_final_up+1,1)))
        up_nonull = np.vstack((np.arange(delta_2+1,row_final_up+1,1),np.arange(col_initial+1,col_final_up+1,1)))
    else:
        #Lower limit
        low_null = np.vstack((np.arange(0,row_final_low+1,1),np.arange(col_initial,grid_dim,1)))
        low_nonull = np.vstack((np.arange(0,row_final_low,1),np.arange(col_initial+1,grid_dim,1)))
        #Upper limit (to consider the y+1 initial point, the delta+2 point is considered since the y reference is 0)
        up_null = np.vstack((np.arange(delta_2,row_final_up+1,1),np.arange(col_initial,grid_dim,1)))
        up_nonull = np.vstack((np.arange(delta_2+1,row_final_up+1,1),np.arange(col_initial+1,grid_dim,1)))
    
    #CALCULATION OF QUANTITIES RELATED TO THE PULSE-MOMENTUM PRODUCT. 
    #In the solution of the integral, it is necessary to integrate along the resonance line to find f(x0) and to consider the incremental ratio calculated based on the pulse-momentum product function.
    f_x0 = sp.integrate.trapz(pulse_mom[0,index,resonance_center[0],resonance_center[1]],dx=step_size)
    # f_x0_cc = sp.integrate.trapz(pulse_mom[index,0,resonance_center[0],resonance_center[1]],dx=step_size)
    f_prime_x0_integrand = pulse_mom[0,index,up_nonull[0],up_nonull[1]]-pulse_mom[0,index,low_nonull[0],low_nonull[1]]
    # f_prime_x0_integrand_cc = pulse_mom[index,0,up_nonull[0],up_nonull[1]]-pulse_mom[index,0,low_nonull[0],low_nonull[1]]
    f_prime_x0 = sp.integrate.trapz(f_prime_x0_integrand,dx=step_size)
    
    return low_null, up_null, f_x0, f_prime_x0, resonance_strip


num_processes = int(np.round(time_array.size/100))+1

####

#INTEGRAL COMPUTATION
#Strip data from compute_stri_stats for each state (corresponding to index)
#Strip data passed to wp_calc_opt and then parallelisation

def compute_integral(index):
    low, up, f, f_prime, resonan_strip = compute_strip_stats(index)
    pulse_mom[0,index,resonan_strip[0],resonan_strip[1]] = 0 #CHECK FOR REDUNDANCY
    result = np.zeros((time_array.size),dtype=np.complex64)
    time_grids = np.array_split(time_array,num_processes)
    with mp.Pool(num_processes) as p:
        result_ = p.starmap(wp_calc_opt, 
                               zip(
                                   time_grids, 
                                   itertools.repeat(index),
                                   itertools.repeat(low), 
                                   itertools.repeat(up),
                                   itertools.repeat(f), 
                                   itertools.repeat(f_prime),
                                   itertools.repeat(resonan_strip)
                                   ))
        begin_ = 0
        for res in result_:
            end_ = begin_+len(res)
            result[begin_:end_] = np.array(res)
            begin_ = end_
    return result


states = range(num_val_states) # 0, 1, 2, num_val_states-1
timer.tic()
for index in states:
    res = compute_integral(index)
    c[index,:] += np.asarray(res).reshape(len(time_array))*np.exp(-complex(0,1)*en_array[0,index]*time_array) 
timer.toc(msg='loop time')

#Calculating density matrix

DM = np.einsum('it,jt->ijt',c,np.conjugate(c))

#Saving data in external dictionary
WP_data = {'Density_Matrix': DM, 'time_array': time_array, 'pulse_time': pulse_time ,'#_val_states': num_val_states, '#_core_states': num_core_states}
np.save(outputfilename,WP_data)

print('finished')
