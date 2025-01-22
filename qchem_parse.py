import numpy as np
import matplotlib.pyplot as plt

#######################
#READ-OUT QCHEM OUTPUT
#######################

#The code in this section reads the output from Qchem. The following variables are read:
#1-photon properties: GS->VE (A), GS->CE (B) and CE->CE (C), transition energies and transition dipole moments.
#2-photon properties: GS->GS (REXS) and GS->VE (RIXS) transition moments and relative frequency grids.

def ao_to_mo(C,DM_ao):
    '''
    Converts the DM from ao basis to mo basis.

    Arguments: MO coefficients matrix C (ao,mo), density matrix AO basis (DM_ao)
    Performs: C^T*DM_ao*C = DM_mo -> (mo,ao)(ao,ao)(ao,mo) = (mo,mo)
    Returns: DM_mo (mo,mo)
    '''
    return np.einsum('jp,jk,kq->pq',C,DM_ao,C)

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


path_2P = '../dm_printout/calc/'
file_2P = 'OCS-OK-XAS-SRIXS.out'


#Importing all the data from Qchem's output files
num_val_states = 0    #valence-excited states considered in the transition
num_core_states = 0

#State density matrix variables
DM_out = False
DM_state = []

#MO matrix variables
MO_mat_out = False
MO_list_sub = []
MO_list = []

#1-photon: checkpoint variables
block_A = False
block_B = False
block_C = False
block_AB = False

AB_mat_out = False
BA_mat_out = False

A_TDM_AB = []
A_TDM_BA = []

B_TDM_AB = []
B_TDM_BA = []

C_TDM_AB = []
C_TDM_BA = []


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

        if 'basis functions' in line:

            #Getting basis set dimension -> matrices dimension
            mat_dim = int(line.split(' ')[6].strip())

        '''
        MO coefficients matrix
        Matrix printed in blocks of shape (mat_dim,a), with a<=6.
        Printout rows: line_count = 1 and line_count = mat_dim; cols: 1-a<=6
        Elements of each block appended to MO_list_sub. Turned to numpy array and appended to MO_list.
        Blocks in MO_list concatenated to give C matrix.
        '''

        if 'Final Alpha MO Coefficients' in line:
            MO_mat_out = True
            line_count = -1

        if MO_mat_out == True:

            if line_count > 0 and line_count <= mat_dim:
                list_el = list(map(lambda y: float(y),(map(lambda x: x.strip(),list(filter(None,line.split(' ')))[1:]))))
                MO_list_sub+=list_el

            if line_count == mat_dim:
                MO_list.append(np.array(MO_list_sub).reshape((mat_dim,len(MO_list_sub)//mat_dim)))
                MO_list_sub = []
                line_count = -1
            
            line_count += 1

            if 'Final Alpha Density Matrix' in line:
                C = np.linalg.inv(np.concatenate(MO_list,axis=1).transpose())
                MO_mat_out = False
        
        '''
        Saving number of states and labels
        label_list order: GS, VE, CE
        '''


        if 'Reference state properties' in line:
            label_list = ['GS']

        elif 'Excited state properties' in line:

            if 'CVS' in line:
                label_list.append('CE-'+line.split('  ')[1].split(' ')[2].strip())
                num_core_states += 1
            else:
                label_list.append('VE-'+line.split('  ')[1].split(' ')[2].strip())
                num_val_states += 1

        '''
        Saving state density matrices
        Matrix printed out between 'DM-S' and 'DM-E' strings.
        Matrix lines saved temporarily in DM_state_list, then appended to DM_state at end reading.
        DM_state list follows same ordering of label_list
        '''

        if 'DM-S' in line and 'T' not in line:
            DM_line_count = -1
            DM_state_list = []
            DM_out = True

        elif 'DM-E' in line and 'T' not in line:
            DM_out = False
            DM_state.append(ao_to_mo(C,np.array(np.array(DM_state_list).reshape((mat_dim,mat_dim)))))

        if DM_out == True:

            if DM_line_count >0:
                DM_state_list+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

            DM_line_count += 1


        #####################
        #1-photon properties#
        #####################

        '''
        Output of blocks A, B and C
        Transition energies, transition density matrices and dipole moments(A->B, B->A)
        Block A: GS<->VE transitions 
        Block B: GS<->CE transitions
        Block C: CE<->CE transitions
        '''
        
        #BLOCKS A+B: Finding out where the blocks of the matrix output are printed out

        if 'State A: ccsd:' in line:
            block_AB = True

        if block_AB == True and 'State B: eomee_ccsd/rhfref/singlets:' in line:
            block_A = True
            block_B = False
        
        if block_AB == True and 'State B: cvs_eomee_ccsd/rhfref/singlets:' in line:
            block_B = True
            block_A = False

        if 'State A: cvs_eomee_ccsd/rhfref/singlets:' in line:
            block_AB = False
            block_B = False
            block_C = True

        #BLOCK A: data acquisition - transition dipoles + energies
        
        if block_A == True and 'Energy GAP' in line:
            A_en.append(float(line.split('=')[2].strip().replace('eV','').strip()))
            A_dp_line = count + 2

        if block_A == True and 'A->B:' in line and count == A_dp_line:
            A_dip_AB.append(dipole_moment_processing(line))

        elif block_A == True and 'B->A:' in line and count == A_dp_line+1:
            A_dip_BA.append(dipole_moment_processing(line))

        #BLOCK B: data acquisition - transition density matrices

        if block_A == True and 'A->B TDM-S' in line:
            TDM_line_count = -1
            TDM_temp = []
            AB_mat_out = True
                
        elif block_A == True and 'A->B TDM-E' in line:
            AB_mat_out = False
            A_TDM_AB.append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

        if block_A == True and 'B->A TDM-S' in line:
            TDM_line_count = -1
            TDM_temp = []
            BA_mat_out = True
                
        elif block_A == True and 'B->A TDM-E' in line:
            BA_mat_out = False
            A_TDM_BA.append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

        if block_A == True and (AB_mat_out == True or BA_mat_out == True):

            if TDM_line_count > 0:
                TDM_temp+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

            TDM_line_count += 1

        #BLOCK B: data acquisition - transition dipoles + energies

        if block_B == True and 'Energy GAP' in line:
            B_en.append(float(line.split('=')[2].strip().replace('eV','').strip()))
            B_dp_line = count + 2

        if block_B == True and 'A->B:' in line and count == B_dp_line:
            B_dip_AB.append(dipole_moment_processing(line))

        elif block_B == True and 'B->A:' in line and count == B_dp_line+1:
            B_dip_BA.append(dipole_moment_processing(line))

        #BLOCK B: data acquisition - transition density matrices

        if block_B == True and 'A->B TDM-S' in line:
            TDM_line_count = -1
            TDM_temp = []
            AB_mat_out = True
                
        elif block_B == True and 'A->B TDM-E' in line:
            AB_mat_out = False
            B_TDM_AB.append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

        if block_B == True and 'B->A TDM-S' in line:
            TDM_line_count = -1
            TDM_temp = []
            BA_mat_out = True
                
        elif block_B == True and 'B->A TDM-E' in line:
            BA_mat_out = False
            B_TDM_BA.append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

        if block_B == True and (AB_mat_out == True or BA_mat_out == True):

            if TDM_line_count >0:
                TDM_temp+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

            TDM_line_count += 1

        #BLOCK C: data acquisition - transition dipoles + energies

        if block_C == True and 'Energy GAP' in line:
            C_en.append(float(line.split('=')[2].strip().replace('eV','').strip()))
            C_dp_line = count + 2

        if block_C == True and 'A->B:' in line and count == C_dp_line:
            C_dip_AB.append(dipole_moment_processing(line))

        elif block_C == True and 'B->A:' in line and count == C_dp_line+1:
            C_dip_BA.append(dipole_moment_processing(line))

        #BLOCK C: data acquisition - transition density matrices

        if block_C == True and 'A->B TDM-S' in line:
            TDM_line_count = -1
            TDM_temp = []
            AB_mat_out = True
                
        elif block_C == True and 'A->B TDM-E' in line:
            AB_mat_out = False
            C_TDM_AB.append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

        if block_C == True and 'B->A TDM-S' in line:
            TDM_line_count = -1
            TDM_temp = []
            BA_mat_out = True
                
        elif block_C == True and 'B->A TDM-E' in line:
            BA_mat_out = False
            C_TDM_BA.append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

        if block_C == True and (AB_mat_out == True or BA_mat_out == True):

            if TDM_line_count >0:
                TDM_temp+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

            TDM_line_count += 1
        
#         #####################
#         #2-photon properties#
#         #####################
        
#         #REXS OUTPUT
        
#         if 'CCSD REXS Moments M_IJ, I,J=X,Y,Z (a.u.):' in line:
#             REXS_out = True
#             index_REXS = count
            
#         if 'REXS Scattering Strength Tensor S (a.u.):' in line:
#             REXS_out = False
        
#         if REXS_out == True and count in range(index_REXS+1,index_REXS+4):
#             AA.append(REXS_TM_mod(line))
        
# #         #VALENCE STATE TRANSITION PROPERTIES
# #         #Storing the couple of A,B indeces 
        
# #         if 'State A: eomee_ccsd/rhfref/singlets:' in line or 'State A: ccsd:' in line:
# #             index_a_temp = int(line.split(':')[2].split('/')[0])
        
# #         if 'State B: eomee_ccsd/rhfref/singlets:' in line:
# #             index_ab.append((index_a_temp,int(line.split(':')[2].split('/')[0])))
# #             val_out = True
# #             index_state = count
            
#         #RIXS OUTPUT
            
#         if 'RIXS Moments M_IJ (A-->B), I,J=X,Y,Z (a.u.):' in line:
#             RIXS_out = True
#             index_RIXS = count
            
#         if 'RIXS Scattering Strength Tensor S (a.u.):' in line:
#             RIXS_out = False
            
#         if RIXS_out == True and count in range(index_RIXS+1,index_RIXS+4):
#             AB.append(RIXS_TM_mod(line))
            
#         if RIXS_out == True and count in range(index_RIXS+5,index_RIXS+8):
#             BA.append(RIXS_TM_mod(line))
            
#         #GRID OUTPUT
            
#         if 'Absorbed photon' in line and len(index_ab) == 2:
#             pump_l.append(float(line.split('=')[1].replace('a.u.','').strip()))
            
#         if 'Emitted photon' in line and len(index_ab) == 2:
#             dump_l.append(float(line.split('=')[1].replace('a.u.','').strip()))