import numpy as np

'''
QCHEM OUTPUT FILE PARSER
This script contains functions to parse the output file produced by a EOM-CCSD calculation, considering core-excited (XAS) and valence-excited (RIXS) states.
The output file is produced by a developer version of Qchem 6.1, modified to print out state and A->B/B->A transition density matrices.
The script aims at reading the calculation data and export it as a dictionary.
'''

__author__ = 'Emanuele Rossi'
__version__ = '2.1'
__version_date__ = 'February 2025'

def ao_to_mo(DM_ao,C):
    '''
    Converts the DM from ao basis to mo basis.

    Arguments: MO coefficients matrix C (ao,mo), density matrix AO basis (DM_ao)
    Performs: C^T*DM_ao*C = DM_mo -> (mo,ao)(ao,ao)(ao,mo) = (mo,mo)
    Returns: DM_mo (mo,mo)
    '''
    return np.einsum('pj,ijk,qk->ipq',np.linalg.inv(C), np.array(DM_ao, dtype=float), np.linalg.inv(C))

def triang_to_full(tri_mat,mat_dim):
    '''
    Converts (upper) triangular state DM printout into full matrix
    
    Arguments: triangular matrix elements (tri_mat), matrix dimension (mat_dim)
    Returns: square state DM matrix
    '''
    tri = np.zeros((mat_dim, mat_dim))
    tri[np.tril_indices(mat_dim, 0)] = tri_mat
    return tri+np.triu(np.transpose(tri),1)


def string_to_complex(string):
    '''
    Converts the RIXS tensor output from string to complex number

    Arguments: string containing complex RIXS tensor components (Qchem format)
    Returns: RIXS tensor component as a complex number 
    '''
    string_format = string.split('+ i')
    complex_num = complex(float(string_format[0].strip()),float(string_format[1].replace('(','').replace(')','').strip()))
    return(complex_num)


def RIXS_TM_mod(row,first_row = False):
    '''
    Converts the RIXS tensor (3x3) into list of complex numbers

    Arguments: row of the RIXS tensor (first row contains also 'A->B' or 'B->A')
    Returns: row of the RIXS tensor as list of complex
    '''

    if first_row == True:
        return list(map(lambda x: string_to_complex(x),list(filter(None,row.split('   ')))[1:-1]))
    
    elif first_row == False:
        return list(map(lambda x: string_to_complex(x),list(filter(None,row.split('   ')))[0:-1]))
    

def dipole_moment_processing(vector_string):
    '''
    Reads x, y, z components of string dipole moment and returns them as a list of floats

    Arguments: string containing x, y, z components of dipole moment (Qchem format)
    Returns: [x,y,z] list of floats corresponding to dipole moment.
    '''
    x = float(vector_string.split('(')[1].split(',')[0].replace('X','').strip())
    y = float(vector_string.split('(')[1].split(',')[1].replace('Y','').strip())
    z = float(vector_string.split('(')[1].split(',')[2].replace('Z','').replace(')','').strip())

    return(list([x,y,z]))


def output_parse(file):
    '''
    Parses the Qchem 6.1 output file, modified to printout the state and transition (A->B, B->A) density matrices.
    Transition properties divided in blocks:
    Block A: GS<->VE transitions 
    Block B: GS<->CE transitions
    Block C: CE<->CE transitions

    Arguments: Qchem output file path, Qchem output file name
    Returns: dictionary 'file_content' with the state and transition properties
    '''

    file_content = {'calc_data' :{'mat_dim': 0},
                    'state': {'state_labels' : [], 
                              'state_dm' : [],
                              'num_val_states': 0,
                              'num_core_states': 0},
                    'transition': {'block_A': {'state_labels': [],
                                               'tr_energies': [],
                                               'AB_dipole': [],
                                               'BA_dipole': [],
                                               'AB_tdm': [],
                                               'BA_tdm': [],
                                               'REXS_tm': [],
                                               'RIXS_AB_tm': [],
                                               'RIXS_BA_tm': [],
                                               'RIXS_grid_p': [],
                                               'RIXS_grid_d': []},

                                    'block_B': {'state_labels': [],
                                               'tr_energies': [],
                                               'AB_dipole': [],
                                               'BA_dipole': [],
                                               'AB_tdm': [],
                                               'BA_tdm': []},

                                    'block_C': {'state_labels': [],
                                               'tr_energies': [],
                                               'AB_dipole': [],
                                               'BA_dipole': [],
                                               'AB_tdm': [],
                                               'BA_tdm': []}}}

    #Output file - 1-photon: block checkpoint variables
    state_out = False
    block_B = False
    block_C = False

    #Output file - 2-photon: RIXS and REXS output checkpoint variables
    RIXS_out = False   #Output of RIXS tensor (relative to a couple of frequency points)
    REXS_out = False   #Output of REXS tensor (relative to an omega_p point)

    #output file - 2-photon: RIXS TM A->B, B->A lists initialisation
    RIXS_TM_AB = []
    RIXS_TM_BA = []
    
    #fchk file - AO<->MO transformation matrix, checkpoint and storing variables
    MO_C_out = False
    MO_num_out = 0
    MO_list = []

    #fchk file - Initialize B, C blocks + A->B, B->A outputs
    AB_out = False
    BA_out = False
    B_out = False
    C_out = False

    #fchk file - Initialize CE and VE state lists
    CE_state_list = []
    VE_state_list = []

    #fchk file - Initialize A->B, B->A output lists for each block
    A_AB_list = []
    A_BA_list = []
    B_AB_list = []
    B_BA_list = []
    C_AB_list = []
    C_BA_list = []

    with open(file+'-SRIXS.in.fchk') as f:
        '''
        MO coefficient, state and transition DMs from checkpoint file - block A
        Each line is appended to MO_list. MO_list is reshaped into array when mat_dim is available.
        '''
        for count,line in enumerate(f):

            if 'Number of basis functions' in line:
                num_MO = int(line.split('I')[1].strip())
                file_content['calc_data']['mat_dim'] = num_MO

            if 'Alpha MO coefficients' in line:
                MO_C_out = True
                MO_num_out += 1
                header_line_MO_C = count

                if int(line.split('=')[1].strip()) % 5 > 0:
                    num_lines = int(line.split('=')[1].strip()) // 5 + 1
                else:
                    num_lines = int(line.split('=')[1].strip()) // 5

            if MO_C_out == True and MO_num_out == 1:
                
                if count > header_line_MO_C and count < (header_line_MO_C+num_lines):
                    MO_list += list(i.strip() for i in filter(None,line.split(' ')))
                    
                elif count == (header_line_MO_C+num_lines):
                    MO_list += list(i.strip() for i in filter(None,line.split(' ')))
                    MO_C = np.array(MO_list, dtype=float).reshape((num_MO,num_MO), order='F')
                    MO_C_out = False

            if 'CC State Density' in line or 'State Density' in line:
                state_out = True
                header_line = count
                state_list = []
            
                #Getting the number of lines of DM's printout's number of lines - based on number of elements and elements per line data
                if int(line.split('=')[1].strip()) % 5 > 0:
                    num_lines = int(line.split('=')[1].strip()) // 5 + 1
                else:
                    num_lines = int(line.split('=')[1].strip()) // 5

            if state_out == True: 
                
                if count > header_line and count < (header_line+num_lines):
                    state_list += list(i.strip() for i in filter(None,line.split(' ')))
                    
                elif count == (header_line+num_lines):
                    state_list += list(i.strip() for i in filter(None,line.split(' ')))
                    VE_state_list.append(triang_to_full(np.array(state_list,dtype=float),num_MO))
                    state_out = False

            if 'Transition DM' in line:
                header_line_A = count
                tdm_list = []
                
                if int(line.split('=')[1].strip()) % 5 > 0:
                    num_lines = int(line.split('N=')[1].strip()) // 5 + 1
                else:
                    num_lines = int(line.split('N=')[1].strip()) // 5
                
                if 'A->B:' in line:
                    AB_out = True
                    BA_out = False

                if 'B->A:' in line:
                    BA_out = True
                    AB_out = False

            if AB_out == True or BA_out == True:

                if count > header_line_A and count < (header_line_A+num_lines):
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    
                elif count == (header_line_A+num_lines) and AB_out == True:
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    A_AB_list.append(np.array(tdm_list, dtype=float).reshape((num_MO,num_MO), order='F'))
                    AB_out = False
                
                elif count == (header_line_A+num_lines) and BA_out == True:
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    A_BA_list.append(np.array(tdm_list, dtype=float).reshape((num_MO,num_MO), order='F'))
                    BA_out = False

    file_content['state']['state_dm'] = ao_to_mo(VE_state_list,MO_C)
    file_content['transition']['block_A']['AB_tdm'] = ao_to_mo(A_AB_list,MO_C)
    file_content['transition']['block_A']['BA_tdm'] = ao_to_mo(A_BA_list,MO_C)

    with open(file+'-XAS.in.fchk') as f:
        '''
        MO coefficient, state and transition DMs from checkpoint file - blocks B, C
        Each line is appended to MO_list. MO_list is reshaped into array when mat_dim is available.
        '''
        for count,line in enumerate(f):

            if 'Number of basis functions' in line:
                num_MO = int(line.split('I')[1].strip())

            if 'Alpha MO coefficients' in line:
                MO_C_out = True
                MO_num_out += 1
                header_line_MO_C = count

                if int(line.split('=')[1].strip()) % 5 > 0:
                    num_lines = int(line.split('=')[1].strip()) // 5 + 1
                else:
                    num_lines = int(line.split('=')[1].strip()) // 5

            if MO_C_out == True and MO_num_out == 1:
                
                if count > header_line_MO_C and count < (header_line_MO_C+num_lines):
                    MO_list += list(i.strip() for i in filter(None,line.split(' ')))
                    
                elif count == (header_line_MO_C+num_lines):
                    MO_list += list(i.strip() for i in filter(None,line.split(' ')))
                    MO_C = np.array(MO_list, dtype=float).reshape((num_MO,num_MO), order='F')
                    MO_C_out = False

            if 'State Density' in line:
                state_out = True
                header_line = count
                state_list = []
            
                if int(line.split('=')[1].strip()) % 5 > 0:
                    num_lines = int(line.split('=')[1].strip()) // 5 + 1
                else:
                    num_lines = int(line.split('=')[1].strip()) // 5

            if state_out == True: 
                
                if count > header_line and count < (header_line+num_lines):
                    state_list += list(i.strip() for i in filter(None,line.split(' ')))
                    
                elif count == (header_line+num_lines):
                    state_list += list(i.strip() for i in filter(None,line.split(' ')))
                    CE_state_list.append(triang_to_full(np.array(state_list,dtype=float),num_MO))
                    state_out = False

            if 'Transition DM' in line:
                
                if int(line.split('=')[1].strip()) % 5 > 0:
                    num_lines = int(line.split('N=')[1].strip()) // 5 + 1
                else:
                    num_lines = int(line.split('N=')[1].strip()) // 5
                
                if 'cvs' in line.split('<-->')[0]:
                    C_out = True
                    header_line_C = count
                    tdm_list = []
                    
                elif ':ccsd' in line.split('<-->')[0] and 'cvs' in line.split('<-->')[1]:
                    B_out = True
                    header_line_B = count
                    tdm_list = []

                if 'A->B:' in line:
                    AB_out = True
                    BA_out = False

                if 'B->A:' in line:
                    BA_out = True
                    AB_out = False
            
            if B_out == True:
            
                if count > header_line_B and count < (header_line_B+num_lines):
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))

                if count == (header_line_B+num_lines) and AB_out == True:
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    B_AB_list.append(np.array(tdm_list, dtype=float).reshape((num_MO,num_MO), order='F'))
                    B_out = False
                    AB_out = False

                if count == (header_line_B+num_lines) and BA_out == True:
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    B_BA_list.append(np.array(tdm_list, dtype=float).reshape((num_MO,num_MO), order='F'))
                    B_out = False
                    BA_out = False
        
            if C_out == True:
            
                if count > header_line_C and count < (header_line_C+num_lines):
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    
                elif count == (header_line_C+num_lines) and AB_out == True:
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    C_AB_list.append(np.array(tdm_list, dtype=float).reshape((num_MO,num_MO), order='F'))
                    C_out = False
                    AB_out = False
                
                elif count == (header_line_C+num_lines) and BA_out == True:
                    tdm_list += list(i.strip() for i in filter(None,line.split(' ')))
                    C_BA_list.append(np.array(tdm_list, dtype=float).reshape((num_MO,num_MO), order='F'))
                    C_out = False
                    BA_out = False

    file_content['state']['state_dm'] = np.concatenate((file_content['state']['state_dm'],ao_to_mo(CE_state_list,MO_C)))
    file_content['transition']['block_B']['AB_tdm'] = ao_to_mo(B_AB_list,MO_C)
    file_content['transition']['block_B']['BA_tdm'] = ao_to_mo(B_BA_list,MO_C)
    file_content['transition']['block_C']['AB_tdm'] = ao_to_mo(C_AB_list,MO_C)
    file_content['transition']['block_C']['BA_tdm'] = ao_to_mo(C_BA_list,MO_C)

    with open(file+'-SRIXS.out','r') as f: 
        '''
        Saving the 1-photon output of block A (GS<->VE transitions )
        Transition energies and transition dipole moments (A->B, B->A)
        '''
        for count, line in enumerate(f):
            '''
            Saving number of valence-excited states and labels
            label_list order: GS, VE,
            '''
            if 'Reference state properties' in line:
                file_content['state']['state_labels'] = ['GS']

            elif 'Excited state properties' in line:
                file_content['state']['state_labels'].append('VE-'+line.split('  ')[1].split(' ')[2].strip())
                file_content['state']['num_val_states'] += 1

            if 'State B: eomee_ccsd/rhfref/singlets:' in line:
                file_content['transition']['block_A']['state_labels'].append('GS<->' + line.split(':')[2].strip())

            if 'Energy GAP' in line:
                file_content['transition']['block_A']['tr_energies'].append(float(line.split('=')[2].strip().replace('eV','').strip()))
                A_dp_line = count + 2

            if 'A->B:' in line and count == A_dp_line:
                file_content['transition']['block_A']['AB_dipole'].append(dipole_moment_processing(line))

            elif 'B->A:' in line and count == A_dp_line+1:
                file_content['transition']['block_A']['BA_dipole'].append(dipole_moment_processing(line))

            #2-photon properties - REXS + RIXS output, frequency grid

            #GRID OUTPUT
                
            if 'Absorbed photon' in line and len(file_content['transition']['block_A']['state_labels']) == 2:
                file_content['transition']['block_A']['RIXS_grid_p'].append(float(line.split('=')[1].replace('a.u.','').strip()))
                
            if 'Emitted photon' in line and len(file_content['transition']['block_A']['state_labels']) == 2:
                file_content['transition']['block_A']['RIXS_grid_d'].append(float(line.split('=')[1].replace('a.u.','').strip()))

            #REXS OUTPUT: REXS_TM shape:(omega_p,3,3)
            
            if 'CCSD REXS Moments M_IJ, I,J=X,Y,Z (a.u.):' in line:
                REXS_out = True
                tensor_temp = []
                index_REXS = -1
                
            if 'REXS Scattering Strength Tensor S (a.u.):' in line:
                REXS_out = False
                file_content['transition']['block_A']['REXS_tm'].append(np.array(tensor_temp).reshape(3,3))
            
            if REXS_out == True:
                tensor_temp += RIXS_TM_mod(line)
                index_REXS+=1
                        
            #RIXS OUTPUT: RIXS_TM shape:(num_val_states*omega_p,3,3)
                
            if 'RIXS Moments M_IJ (A-->B), I,J=X,Y,Z (a.u.):' in line:
                RIXS_out = True
                tensor_temp = []
                index_RIXS = count

            if 'RIXS Moments M_IJ (B-->A), I,J=X,Y,Z (a.u.):' in line:
                RIXS_TM_AB.append(np.array(tensor_temp).reshape(3,3))
                tensor_temp = []
                
            if 'RIXS Scattering Strength Tensor S (a.u.):' in line:
                RIXS_out = False
                RIXS_TM_BA.append(np.array(tensor_temp).reshape(3,3))
                
            if RIXS_out == True and count in range(index_RIXS+1,index_RIXS+4):

                if count == index_RIXS+1:
                    tensor_temp += RIXS_TM_mod(line,True)
                else:
                    tensor_temp += RIXS_TM_mod(line)

            if RIXS_out == True and count in range(index_RIXS+5,index_RIXS+8):

                if count == index_RIXS+5:
                    tensor_temp += RIXS_TM_mod(line,True)
                else:
                    tensor_temp += RIXS_TM_mod(line)


    #Reshaping RIXS TM's to (num_val_states, omega_p, 3, 3) and REXS TMs to (omega_p, 3, 3)
    file_content['transition']['block_A']['REXS_tm'] = np.array(file_content['transition']['block_A']['REXS_tm'],dtype='complex64').reshape(len(file_content['transition']['block_A']['RIXS_grid_p']),3,3)
    file_content['transition']['block_A']['RIXS_AB_tm'] = np.array(RIXS_TM_AB,dtype='complex64').reshape(file_content['state']['num_val_states'],len(file_content['transition']['block_A']['RIXS_grid_p']),3,3)
    file_content['transition']['block_A']['RIXS_BA_tm'] = np.array(RIXS_TM_BA,dtype='complex64').reshape(file_content['state']['num_val_states'],len(file_content['transition']['block_A']['RIXS_grid_p']),3,3)
    
    
    with open(file+'-XAS.out','r') as f:
        '''
        1-photon output of block B (GS<->CE transitions) and C (CE<->CE transitions)
        Transition energies and transition dipole moments (A->B, B->A)
        ''' 
        for count, line in enumerate(f):            
            '''
            Saving number of states and labels
            label_list order: Irreducible representation, state number 
            '''
            if 'Excited state properties' in line:
                file_content['state']['state_labels'].append('CE-'+line.split('  ')[1].split(' ')[2].strip())
                file_content['state']['num_core_states'] += 1

            #BLOCKS B vs C: Finding out where the blocks of the matrix output are printed out

            if 'State A: ccsd:' in line:
                block_B = True

            if block_B == True and 'State B:' in line:
                file_content['transition']['block_B']['state_labels'].append('GS<->' + line.split(':')[2].strip())

            elif 'State A: cvs_eomee_ccsd/rhfref/singlets:' in line:
                block_C = True
                block_B = False
                block_C_stateA = line.split(':')[2].strip()

            if block_C == True and 'State B: cvs_eomee_ccsd/rhfref/singlets:' in line:
                file_content['transition']['block_C']['state_labels'].append(block_C_stateA + '<->' + line.split(':')[2].strip())

            #BLOCK B: data acquisition - transition dipoles + energies

            if block_B == True and 'Energy GAP' in line:
                file_content['transition']['block_B']['tr_energies'].append(float(line.split('=')[2].strip().replace('eV','').strip()))
                B_dp_line = count + 2

            if block_B == True and 'A->B:' in line and count == B_dp_line:
                file_content['transition']['block_B']['AB_dipole'].append(dipole_moment_processing(line))

            elif block_B == True and 'B->A:' in line and count == B_dp_line+1:
                file_content['transition']['block_B']['BA_dipole'].append(dipole_moment_processing(line))

            #BLOCK C: data acquisition - transition dipoles + energies

            if block_C == True and 'Energy GAP' in line:
                file_content['transition']['block_C']['tr_energies'].append(float(line.split('=')[2].strip().replace('eV','').strip()))
                C_dp_line = count + 2

            if block_C == True and 'A->B:' in line and count == C_dp_line:
                file_content['transition']['block_C']['AB_dipole'].append(dipole_moment_processing(line))

            elif block_C == True and 'B->A:' in line and count == C_dp_line+1:
                file_content['transition']['block_C']['BA_dipole'].append(dipole_moment_processing(line))

    return file_content