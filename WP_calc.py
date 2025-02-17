print('loading modules')

import multiprocessing as mp
import cmath,math,time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools
import numexpr as ne
import qchem_parse
import sys


## LOADING USEFUL COSTANTS ##

time_1aut_s = sp.constants.physical_constants['atomic unit of time'][0]
planck_eV_Hz = sp.constants.physical_constants['Planck constant in eV/Hz'][0]
epsilon_0_au = sp.constants.epsilon_0/sp.constants.physical_constants['atomic unit of permittivity'][0]
energy_1auE_eV = sp.constants.physical_constants['hartree-electron volt relationship'][0]

print('calling python script with')
print(f'file {sys.argv[1]}')
print(f'pulse option {sys.argv[2]}')
print(f'color #1 carrier {sys.argv[3]}')
print(f'color #2 carrier {sys.argv[4]}')
print(f'color #1 bandwidth {sys.argv[5]}')
print(f'color #2 bandwidth {sys.argv[6]}')
print(f'color #1 polarization {sys.argv[7]}')
print(f'color #2 polarization {sys.argv[8]}')
print(f'grid dimension (#points) {sys.argv[9]}')
print(f'storing result in filename: {sys.argv[10]}')

###############################
#DEFINITION OF INPUT VARIABLES
###############################

'''
Script is called with the following input:
1. file input
2. pulse polarisation option (see dictionary)
3. carrier frequency color 1
4. carrier frequency color 2 (optional)
5. bandwidth color 1
6. bandwidth color 2 (optional)
7. polarization color 1
8. polarization color 2
9. frequency grid dimension (#point)
10. Output filename
'''

polarization_dict = {'x' : np.array([1,0,0]),
                     'z' : np.array([0,0,1]),
                     'xz': (1/np.sqrt(2))*np.array([1,0,1]),
                     'xy': (1/np.sqrt(2))*np.array([1,1,0]),
                     'Rxy' : (1/np.sqrt(2))*np.array([1,complex(0,1),0]),
                     'Lxy' : (1/np.sqrt(2))*np.array([1,-complex(0,1),0])
                     }


file = sys.argv[1]
pulse_option = sys.argv[2]

if pulse_option == '1C':
    freq_carrier = float(sys.argv[3])/energy_1auE_eV
    bandwidth = float(sys.argv[5])/energy_1auE_eV
    pol = polarization_dict[sys.argv[7]]
    print(f'pol check {pol}')

elif pulse_option == '2C':
    carrier_C1 = float(sys.argv[3])/energy_1auE_eV
    carrier_C2 = float(sys.argv[4])/energy_1auE_eV
    bandwidth_C1 = float(sys.argv[5])/energy_1auE_eV
    bandwidth_C2 = float(sys.argv[6])/energy_1auE_eV
    pol_C1 = polarization_dict[sys.argv[7]]
    pol_C2 = polarization_dict[sys.argv[8]]

dim = int(sys.argv[9])
outputfilename = sys.argv[10]

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

qchem_out_data = qchem_parse.output_parse(file)

block_A_dim = qchem_out_data['state']['num_val_states']
block_C_dim = qchem_out_data['state']['num_core_states']

################
#SYMMETRIZE DMs
################

def dm_symm(dm_AB,dm_BA):
    '''
    Symmetrizes EOM-CC TDMs: dm = (sqrt(dm_AB*dm_BA)*sign(dm_AB))

    Arguments: real-valued dm_A->B, dm_B->A; shape = (#basis_el, #basis_el)
    Returns: symmetrized real-valued dm_AB; shape = (#basis_el, #basis_el)
    '''
    dm_phase = np.sign(dm_AB)
    dm_mod = np.sqrt(np.multiply(np.abs(dm_AB), np.abs(dm_BA).transpose(0,2,1))) #need abs for imprecise numerics from Qchem which lead to negative arguments for root
    return np.multiply(dm_phase,dm_mod)

#Initialize 1PDM tensor
opDM = np.zeros((block_A_dim+block_C_dim,block_A_dim+block_C_dim,qchem_out_data['calc_data']['mat_dim'],qchem_out_data['calc_data']['mat_dim']))

#State 1PDM on the diagonal
opDM[np.diag_indices(block_A_dim+block_C_dim,ndim=2)] = np.array(qchem_out_data['state']['state_dm'])

#Block A 1PDM: GS<->VE
opDM[0,1:block_A_dim] = dm_symm(qchem_out_data['transition']['block_A']['AB_tdm'],qchem_out_data['transition']['block_A']['BA_tdm'])
opDM[1:block_A_dim,0] = opDM[0,1:block_A_dim].transpose(0,2,1)

#Block B 1PDM: GS<->CE
opDM[0,block_A_dim:] = dm_symm(qchem_out_data['transition']['block_B']['AB_tdm'],qchem_out_data['transition']['block_B']['BA_tdm'])
opDM[block_A_dim:,0] = opDM[0,block_A_dim:].transpose(0,2,1)

#Block C 1PDM: CE<->CE
opDM[np.triu_indices(block_C_dim, k=1)[0]+block_A_dim,np.triu_indices(block_C_dim, k=1)[1]+block_A_dim] = dm_symm(qchem_out_data['transition']['block_C']['AB_tdm'],qchem_out_data['transition']['block_C']['BA_tdm'])
opDM[np.triu_indices(block_C_dim, k=1)[1]+block_A_dim,np.triu_indices(block_C_dim, k=1)[0]+block_A_dim] = opDM[np.triu_indices(block_C_dim, k=1)[0]+block_A_dim,np.triu_indices(block_C_dim, k=1)[1]+block_A_dim].transpose(0,2,1)

###########################################################
#1-PHOTON - ASSEMBLING TRANSITION ENERGY AND DIPOLE ARRAYS
###########################################################

def dip_mom_sym(dip_AB,dip_BA):
    '''
    Symmetrizes EOM-CC dipole moments: dip = (sqrt(dip_AB*dip_BA))*sign(dip_AB)

    Arguments: arrays with dipole A->B and dipole_B->A; shape=(#A<->B transitions,3)
    Returns: symmetrized dipole array; shape=(#A<->B transitions,3)
    '''
    dip_phase = np.sign(dip_AB)
    dip_mod = np.sqrt(np.multiply(np.abs(dip_AB),np.abs(dip_BA))) #abs shouldn't be needed because dipoles should have same sign. Qchem output sometimes with different sign (probably due to too small basis).
    return np.multiply(dip_mod,dip_phase)

#BLOCK A - energy array definition: shape=(GS+#valence_excited,GS+#valence_excited)
A_en_array = np.zeros((block_A_dim,block_A_dim))
A_en_array[0,np.arange(1,block_A_dim,1)] = np.array(qchem_out_data['transition']['block_A']['tr_energies'])   #for calculation with only GS->ES transitions
A_en_array -= np.transpose(A_en_array)

#BLOCK A - dipole array definition: shape=(GS+#valence_excited,GS+#valence_excited,3)
A_dip_array = np.zeros((block_A_dim,block_A_dim,3))
A_dip_array[0,np.arange(1,block_A_dim,1),:] = dip_mom_sym(qchem_out_data['transition']['block_A']['AB_dipole'],qchem_out_data['transition']['block_A']['BA_dipole'])
A_dip_array += np.transpose(A_dip_array,axes=[1,0,2])

#BLOCK B - energy array definition: shape=(GS+#valence_excited,#core_excited)
B_en_array = np.zeros((block_A_dim,block_C_dim))
B_en_array[0,np.arange(0,block_C_dim,1)] = np.array(qchem_out_data['transition']['block_B']['tr_energies'])

#BLOCK B - dipole array definition: shape=(GS+#valence_excited,#core_excited,3)
B_dip_array = np.zeros((block_A_dim,block_C_dim,3))
B_dip_array[0,np.arange(0,block_C_dim,1),:] = dip_mom_sym(qchem_out_data['transition']['block_B']['AB_dipole'],qchem_out_data['transition']['block_B']['BA_dipole'])
print(B_dip_array[0,np.arange(0,block_C_dim,1),:])

#BLOCK C: energy array definition: shape=(#core_excited,#core_excited)
C_en_array = np.zeros((block_C_dim,block_C_dim))
C_en_array[np.triu_indices(block_C_dim,k=1)] = np.array(qchem_out_data['transition']['block_C']['tr_energies'])
C_en_array -= np.transpose(C_en_array)

#BLOCK C: dipole array definition: shape=(#core_excited,#core_excited,3)
C_dip_array = np.zeros((block_C_dim,block_C_dim,3))
C_dip_array[np.triu_indices(block_C_dim,k=1)] = dip_mom_sym(qchem_out_data['transition']['block_C']['AB_dipole'],qchem_out_data['transition']['block_C']['BA_dipole'])
C_dip_array += np.transpose(C_dip_array,axes=[1,0,2])

#Assembling the energy array and the dipole arrays =>((A,B),(B.t,C))
#Energy array shape=(#valence_excited+#core_excited,#valence_excited+#core_excited)
#Dipole array shape=(#valence_excited+#core_excited,#valence_excited+#core_excited,3)

en_array = np.block([[A_en_array,B_en_array],[-np.transpose(B_en_array),C_en_array]])/energy_1auE_eV
dip_array = (np.concatenate((np.concatenate((A_dip_array,B_dip_array),axis=1),np.concatenate((np.transpose(B_dip_array,axes=[1,0,2]),C_dip_array),axis=1)),axis=0))

###########################################
#2-PHOTON - BUILDING REXS and RIXS TENSORS
###########################################

def RIXS_interpolation(TM_array,pump_list,new_p_array,process):
    '''
    Interpolates the REXS/RIXS transition moment arrays printed out by Qchem. Allows flexibility in the choice of frequency grid for integration.

    Arguments: TM_array = REXS TM array of shape (#omega_p,3,3) or RIXS TM array of shape (#num_val_states,#omega_p,3,3); 
               pump_list = frequency array used by Qchem; 
               new_p_array = new frequency array to express TMs;
               process = RIXS or REXS.
    Returns: REXS TM of shape (len(omega_p_array),3,3); RIXS TM of shape (#num_val_states,len(omega_p_array),3,3).
    '''

    #REXS process: the tensor has a shape (#omega_p,3,3). Interpolation along the axis=0.
    if process == 'REXS':
        real_part = sp.interpolate.CubicSpline(np.array(pump_list),TM_array.real,axis=0)
        imag_part = sp.interpolate.CubicSpline(np.array(pump_list),TM_array.imag,axis=0)
    
    #RIXS process: the tensor has a shape (#VE_states,#omega_p,3,3). Interpolation along the axis=1.
    elif process == 'RIXS':
        real_part = sp.interpolate.CubicSpline(np.array(pump_list),TM_array.real,axis=1)
        imag_part = sp.interpolate.CubicSpline(np.array(pump_list),TM_array.imag,axis=1)

    #Putting together the real and imaginary parts of the tensors.
    return np.vectorize(complex)(real_part(new_p_array),imag_part(new_p_array))

def RIXS_mom_sym(TM_AB,TM_BA):
    '''
    Symmetrizes RIXS TMs: TM_AB = (sqrt(TM_A->B*TM_B->A))*(TM_A->B/|TM_A->B|)

    Arguments: TM_A->B, TM_B->A
    Returns: symmetrized RIXS TM_AB tensor of shape (#val_states,omega_p,omega_d,3,3)
    '''
    TM_phase = np.divide(TM_AB,np.abs(TM_AB), where=TM_AB != 0)
    TM_mod = np.sqrt(np.multiply(TM_AB,TM_BA))
    return np.multiply(TM_mod,TM_phase)

#Definition of the pump array to be used for calculations (i.e. more points than the Qchem's output)
pump_array = np.linspace(min(qchem_out_data['transition']['block_A']['RIXS_grid_p']),max(qchem_out_data['transition']['block_A']['RIXS_grid_p']),dim)

#REXS tensor - output of the new REXS vector from the interpolation function and broadcast. Broadcast shape = (#omega_d,#omega_p,3,3)
AA_a = np.broadcast_to(RIXS_interpolation(qchem_out_data['transition']['block_A']['REXS_tm'],qchem_out_data['transition']['block_A']['RIXS_grid_p'],pump_array,'REXS'),(dim,dim,3,3))

#RIXS tensor - output of the new RIXS tensor from the interpolation function, broadcast and averaging. Broadcast shape = (#val_states,#omega_d,#omega_p,3,3)
AB_a = np.broadcast_to(RIXS_interpolation(qchem_out_data['transition']['block_A']['RIXS_AB_tm'],qchem_out_data['transition']['block_A']['RIXS_grid_p'],pump_array,'RIXS')[:,np.newaxis,:,:,:],(block_A_dim-1,dim,dim,3,3))
BA_a = np.broadcast_to(RIXS_interpolation(qchem_out_data['transition']['block_A']['RIXS_BA_tm'],qchem_out_data['transition']['block_A']['RIXS_grid_p'],pump_array,'RIXS')[:,np.newaxis,:,:,:],(block_A_dim-1,dim,dim,3,3))

#Calculation of symmetrized RIXS_TM
RIXS_AB_a = RIXS_mom_sym(AB_a,BA_a)

#Creation of the pump grid starting from the pump array
pump_grid = np.broadcast_to(pump_array,(dim,dim))
dump_grid = pump_grid.transpose()


#Full RIXS tensor of shape (#val_states+GS,#val_states+GS,omega_p,omega_d,3,3)

RIXS_TM = np.zeros((block_A_dim,block_A_dim,dim,dim,3,3),dtype='complex64')

for index in range(0,block_A_dim,1):

    if index == 0:
        RIXS_TM[0,0] = AA_a  #REXS GS->GS
    else:
        RIXS_TM[0,index,...] = RIXS_AB_a[index-1,...]
        RIXS_TM[0,index,...] = np.conjugate(RIXS_AB_a[index-1,...])

#Right and left tensors
# for index,(row,col) in enumerate(index_ab):
    
#     if row == col:
#         RIXS_TM[row,col,...] = AA_a
#     else:
#         RIXS_TM[row,col,...] = AB_a[index-1,...]
        
#         RIXS_TM[col,row,...] = BA_a[index-1,...]

#Averaged tensors
#Only the RIXS excitations from the GS (index=0) to the VE (index>0) are considered.

print('RIXS read')

##################
#PULSE DEFINITION 
##################

def gauss_freq_1D(omega,omega_carrier,time_shift,alpha,amplitude,pol_v):
    '''
    Defines 1D Gaussian pulse envelope in frequency domain.

    Arguments: omega = frequency array
               omega_carrier = carrier frequency
               time_shift = time shift (a.u.t.) with respect to time zero.
               alpha = ((bandwidth*pi)**2)/(2*log(2))
               amplitude = E_0
               pol_v = polarization vector
    Returns: pulse array of shaep (pol_v,omega)
    '''

    shift_factor = np.exp(complex(0,1)*(omega)*time_shift)
    envelope = np.exp(-(((omega-omega_carrier)**2)/(4*alpha)))
    return np.einsum('x,f->xf',pol_v,amplitude*shift_factor*envelope)

def gauss_freq_2D(omega,omega_carrier,time_shift,alpha,amplitude,pol_v):
    '''
    Defines 2D Gaussian pulse envelope in frequency domain.

    Arguments: omega = frequency grid
               omega_carrier = carrier frequency
               time_shift = time shift (a.u.t.) with respect to time zero.
               alpha = ((bandwidth*pi)**2)/(2*log(2))
               amplitude = E_0
               pol_v = polarization vector
    Returns: pulse array of shaep (pol_v,omega)
    '''

    shift_factor = np.exp(complex(0,1)*(omega)*time_shift)
    envelope = np.exp(-(((omega-omega_carrier)**2)/(4*alpha)))
    return np.einsum('x,pd->xpd',pol_v,amplitude*shift_factor*envelope)

#The pulse is defined as E(\omega)=|E(\omega)|e^{i\phi(\omega)}.
#E(\omega) and E(\omega)* differ in the sign of the exponential. The phase \phi(\omega) is set to 0 in testing.

#DEFINITION OF |E_0| FROM IRRADIANCE
#The Irradiance definition is: I=1/2*c*epsilon_0*|E|
#Thus, |E_0|=sqrt(2I/(epsilon_0*c))
#1 a.u. of Irradiance is equal to 3.51e16 W/cm^2 (from https://onlinelibrary.wiley.com/doi/pdf/10.1002/3527605606.app9)
#Epsilon_0 in a.u. is obtained by dividing its value in F/m by the unit of a.u. of permittivity in F/m.
#The formula for the electric field is applied, obtaining the value of |E_0| in a.u.

#FOURIER TRANSFORM CONVENTION
#The convection used is the unitary one:
#F(\omega) = 1/(\sqrt(2\pi))\int_{-\infty}^{\infty}dt f(t)e^{-i\omega t}
#f(t) = 1/(\sqrt(2\pi))\int_{-\infty}^{\infty}dt F(\omega)e^{i\omega t}

irradiance_W_cm2_au = 3.51e16 #(W/cm^2)/a.u. 

#########################
##### 1 COLOR PULSE #####
#########################

### SETTINGS ###

irradiance = 1e16/irradiance_W_cm2_au
E_0 = math.sqrt((2*irradiance)/(epsilon_0_au*137))   #pulse's height at peak
bw_Hz_au = ((bandwidth*energy_1auE_eV)/planck_eV_Hz)*time_1aut_s
alpha = ((bw_Hz_au*np.pi)**2)/(2*math.log(2))

# sigma_f = bandwidth/2.355  #pulse's bandwidth

### TIME DOMAIN ###

#Definition of the gaussian pulse-time domain
begin = 2000e-18/time_1aut_s #attoseconds
end = 4000e-18/time_1aut_s
step_time = 1e-18/time_1aut_s

#Calculation of the duration (attoseconds) - transform limited pulse
duration = np.sqrt((2*math.log(2))/alpha)
print('Pulse duration: %f as'%((duration*time_1aut_s)/1e-18))

time_shift = 0e-18/time_1aut_s

time_array = np.arange(-begin,end,step_time)
pulse_time = np.einsum('x,t->xt',pol,E_0*np.sqrt(alpha/np.pi)*np.exp(-alpha*(time_array-time_shift)**2)*np.exp(-complex(0,1)*freq_carrier*time_array))

### FREQUENCY DOMAIN ###

### first order ###
freq_array = pump_grid[0]
step_freq = pump_grid[1][1]-pump_grid[1][0]  #defining the grid step size

pulse_1P = gauss_freq_1D(freq_array,freq_carrier,time_shift,alpha,E_0,pol)

### second order ###
pump_freq = gauss_freq_2D(pump_grid,freq_carrier,time_shift,alpha,E_0,pol)
dump_freq = gauss_freq_2D(dump_grid,freq_carrier,time_shift,alpha,E_0,pol)

pulse_matrix = np.einsum('xpd,ypd->xypd',pump_freq,np.conjugate(dump_freq)).astype('complex64')


########################
#### 2 COLORS PULSE ####
########################

### SETTINGS ###

# irradiance_C1 = 1e18/irradiance_W_cm2_au
# irradiance_C2 = 1e18/irradiance_W_cm2_au

# E_0_C1 = math.sqrt((2*irradiance_C1)/(epsilon_0_au*137))
# E_0_C2 = math.sqrt((2*irradiance_C2)/(epsilon_0_au*137))

# #Definition of the gaussian pulse parameters for color 1 and color 2
# bw_Hz_au_C1 = ((bandwidth_C1*energy_1auE_eV)/planck_eV_Hz)*time_1aut_s
# alpha_C1 = ((bw_Hz_au_C1*np.pi)**2)/(2*math.log(2))

# bw_Hz_au_C2 = ((bandwidth_C2*energy_1auE_eV)/planck_eV_Hz)*time_1aut_s
# alpha_C2 = ((bw_Hz_au_C2*np.pi)**2)/(2*math.log(2))

### TIME DOMAIN ###

#Definition of the gaussian pulse-time domain
# begin = 2000e-18/time_1aut_s #attoseconds
# end = 4000e-18/time_1aut_s
# step_time = 1e-18/time_1aut_s

#Calculation of the duration (attoseconds) - transform limited pulse
# duration_C1 = np.sqrt((2*math.log(2))/alpha_C1)
# duration_C2 = np.sqrt((2*math.log(2))/alpha_C2)

# print('Pulse duration color 1: %f as'%((duration_C1*time_1aut_s)/1e-18))
# print('Pulse duration color 2: %f as'%((duration_C2*time_1aut_s)/1e-18))

# time_shift_C1 = 0e-18/time_1aut_s
# time_shift_C2 = 0e-18/time_1aut_s

# time_array = np.arange(-begin,end,step_time)
# pulse_time_C1 = np.einsum('x,t->xt',pol_C1,E_0_C1*np.sqrt(alpha_C1/np.pi)*np.exp(-alpha_C1*(time_array-time_shift_C1)**2)*np.exp(-complex(0,1)*carrier_C1*time_array))
# pulse_time_C2 = np.einsum('x,t->xt',pol_C2,E_0_C2*np.sqrt(alpha_C2/np.pi)*np.exp(-alpha_C2*(time_array-time_shift_C2)**2)*np.exp(-complex(0,1)*carrier_C2*time_array))
# pulse_time = pulse_time_C1+pulse_time_C2

### FREQUENCY DOMAIN ###

### first order ###
# freq_array = pump_grid[0]
# step_freq = pump_grid[1][1]-pump_grid[1][0]  #defining the grid step size

# pulse_1P = (gauss_freq_1D(freq_array,carrier_C1,time_shift_C1,alpha_C1,E_0_C1,pol_C1)
#             +gauss_freq_1D(freq_array,carrier_C2,time_shift_C2,alpha_C2,E_0_C2,pol_C2)).astype('complex64')

### second order ###

# pump_freq = gauss_freq_2D(pump_grid,carrier_C1,time_shift_C1,alpha_C1,E_0_C1,pol_C1)+gauss_freq_2D(pump_grid,carrier_C2,time_shift_C2,alpha_C2,E_0_C2,pol_C2)
# dump_freq = gauss_freq_2D(dump_grid,carrier_C1,time_shift_C1,alpha_C1,E_0_C1,pol_C1)+gauss_freq_2D(dump_grid,carrier_C2,time_shift_C2,alpha_C2,E_0_C2,pol_C2)

# pulse_matrix = np.einsum('xpd,ypd->xypd',pump_freq,np.conjugate(dump_freq)).astype('complex64')

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
    '''
    Defines the integrand function of first order coefficients integrand

    Arguments: pulse = pulse array definition on frequency domain of shape (3,\omega)
               dipole = transition dipole moment array of shape (3)
               time = time array representing the time span to calculate integral.
               energy = transition energy (with respect to GS) of excited state.
               frequency = frequency domain of the integral.
    Returns: integrand of shape (time, frequency)
    '''

    numerator = np.exp(complex(0,1)*(energy-frequency-complex(0,1)*(gamma_m/2))*time)
    denominator = energy-frequency-(complex(0,1)*(gamma_m/2))

    #introduced the dot product between the vectorial pulse and the dipole moment
    return np.einsum('xf,x,tf->tf',pulse,dipole,numerator/denominator,optimize='optimal')

                       
def integrand_cc(pulse,frequency,energy,dipole,gamma_m,time_grid):
    
    numerator = np.exp(-complex(0,1)*(energy-frequency+complex(0,1)*(gamma_m/2))*time)
    denominator = energy-frequency+(complex(0,1)*(gamma_m/2))
    
    return pulse*dipole*(numerator)#/denominator)

def schroedinger_p(time,energy,gamma_m):
    '''
    Defines array for conversion from interaction to Schroedinger picture.

    Arguments: time = time array over which the wp coefficients are calculated.
               energy = transition energy of each state (with respect to GS)
               gamma_m = decay rate of the state.
    Returns: conversion factors array (num_states, time)
    '''

    return np.exp(-complex(0,1)*(energy-complex(0,1)*(gamma_m/2))*time)
#     return(np.exp(-complex(0,1)*energy*time_array))


#Initialisation of the arrays of coefficients

c = np.zeros((block_A_dim+block_C_dim,len(time_array)),dtype=complex)
c_cc = np.zeros((block_A_dim+block_C_dim,len(time_array)),dtype=complex)

c[0,:]+=1


#At each time step, the pulse envelope is used in the formula.

freq_grid,time_grid = np.meshgrid(freq_array,time_array)

#The integrand is calculated on a grid with frequency on the x-axis and time on the y-axis.
#The time and frequency meshgrids are input to the 'integrand' function: the integrand is calculated on the grid.
#Integrating over the frequency axis, the coefficients as a function of time are obtained.
#The t-dependent coefficients for each state are stored in an array of shape (#states,length_time_array)

for state in range(block_A_dim,block_A_dim+block_C_dim):
    c[state,:] += schroedinger_p(time_array,en_array[0,state],decay_rate)*((1/(math.sqrt(2*np.pi)))*sp.integrate.trapezoid(integrand(pulse_1P,freq_grid,en_array[0,state],dip_array[0,state],decay_rate,time_grid),dx=step_freq,axis=1))

#######################
#SRIXS WP COEFFICIENTS
#######################

def prefactor_red(energy_eq,time,pulse_mom,gamma_k):
    '''
    Calculates the exponential pre-factor (coming before pulse and RIXS TM).
    Utilizes the numexpr package for more efficient evaluation of vectorial functions.

    Arguments: energy_eq = \omega_kg - \omega_p + \omega_d
               pulse_mom = pulse_matrix * RIXS_TM
               gamma_k = decay rate valence-excited states (set 0 because of analytical approach)
               time = time in a.u. at which prefactor is evaluated.
    Returns: 2D frequency grid over which prefactor function is evaluated.
    '''

    numerator = ne.evaluate('exp((complex(0,1)*energy_eq+(gamma_k/2))*time)')
    denominator = ne.evaluate('energy_eq-complex(0,1)*(gamma_k/2)')
    integrand = ne.evaluate('(numerator/denominator)*pulse_mom')
    return integrand

# def prefactor_red_cc(energy_eq,time,pulse_mom,gamma_k):
#     numerator = ne.evaluate('exp((-complex(0,1)*energy_eq+(gamma_k/2))*time)')
#     denominator = ne.evaluate('energy_eq+complex(0,1)*(gamma_k/2)')
#     return(ne.evaluate('(numerator/denominator)*pulse_mom'))

def wp_calc_opt(time_array, index, low, up, f, f_prime, resonance):
    '''
    Calculates the wp coefficients relative to GS and valence-excited states.
    Utilizes a numerical technique outsied of the strip and an analytical approach within the strip.

    Arguments: time_array = array of time over which to calculate the integral
               index = GS=0; VE>0
               low, up, f, f_prime, resonance = information on the strip from compute_strip_stats
    Returns: WP coefficient as a function of time.
    '''
    
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
        
        correction = (-np.sum(((sp.integrate.trapezoid(integrand_full[low[0],low[1]]*step_size/2, dx=step_size)*step_size)/2)
            +((sp.integrate.trapezoid(integrand_full[up[0],up[1]]*step_size/2, dx=step_size)*step_size)/2))
            +f_prime*2*(math.sin(delta*time)/time)+complex(0,1)*(2*sp.special.sici(delta*time)[0]+np.pi)*f)
    
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
#        correction.append(-np.sum(((sp.integrate.trapezoid(integrand_full[low_nonull[0],low_nonull[1]]*step_size/2, dx=step_size)*step_size)/2)
#                                  +((sp.integrate.trapezoid(integrand_full[up_nonull[0],up_nonull[1]]*step_size/2, dx=step_size)*step_size)/2))
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
energy_eq = np.broadcast_to(en_array[0:block_A_dim,0:block_A_dim,np.newaxis,np.newaxis],pulse_mom.shape)-np.broadcast_to(pump_grid[np.newaxis,np.newaxis,...],pulse_mom.shape)+np.broadcast_to(dump_grid[np.newaxis,np.newaxis,...],pulse_mom.shape)

delta = 0.725e-2 #OCS
#delta = 0.75e-2 #OXAZOLE

step_size = pump_grid[1][1]-pump_grid[1][0]  #defining the grid step size


# uses global variable: grid_dim, delta, energy_eq, pulse_mom
def compute_strip_stats(index):
    '''
    Calculates quantities needed to the analytical form of the strip integral for GS and each valence-excited state.
    The strip is defined according to its center and its width (i.e. delta parameter.)

    Arguments: state index (=0: GS, >0: valence-excited)
    Returns: resonance_strip = grid points corresponding to strip
             low_null, up_null = grid points corresponding to strip's lower and upper point.
             f_x0, f_prime_x0 = evaluation of pulse_mom and its first derivative on the strip.
    '''

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
    f_x0 = sp.integrate.trapezoid(pulse_mom[0,index,resonance_center[0],resonance_center[1]],dx=step_size)
    # f_x0_cc = sp.integrate.trapezoid(pulse_mom[index,0,resonance_center[0],resonance_center[1]],dx=step_size)
    f_prime_x0_integrand = pulse_mom[0,index,up_nonull[0],up_nonull[1]]-pulse_mom[0,index,low_nonull[0],low_nonull[1]]
    # f_prime_x0_integrand_cc = pulse_mom[index,0,up_nonull[0],up_nonull[1]]-pulse_mom[index,0,low_nonull[0],low_nonull[1]]
    f_prime_x0 = sp.integrate.trapezoid(f_prime_x0_integrand,dx=step_size)
    
    return low_null, up_null, f_x0, f_prime_x0, resonance_strip


num_processes = int(np.round(time_array.size/100))+1

####

def compute_integral(index):
    '''
    Computes the double integral. Strip data from compute_stri_stats for each valence-excited state.
    Strip data passed to wp_calc_opt, calculate the intregral and then parallelisation.

    Arguments: state index (=0: GS, >0: valence-excited)
    Returns: wp coefficient as a function of time relative to state index.
    '''

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


states = range(block_A_dim) # 0, 1, 2, num_val_states-1
timer.tic()
for index in states:
    res = compute_integral(index)
    c[index,:] += np.asarray(res).reshape(len(time_array))*np.exp(-complex(0,1)*en_array[0,index]*time_array) 
timer.toc(msg='loop time')

#Calculating density matrix

DM = np.einsum('it,jt->ijt',c,np.conjugate(c))

#Saving data in external dictionary
WP_data = {'Density_Matrix': DM, '1PDM': opDM, 'time_array': time_array, 'pulse_time': pulse_time , '#_val_states': block_A_dim, '#_core_states': block_C_dim}
np.save(outputfilename,WP_data)

print('finished')