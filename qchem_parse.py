import numpy as np

'''
QCHEM OUTPUT FILE PARSER
This script contains functions to parse the output file produced by a EOM-CCSD calculation, considering core-excited (XAS) and valence-excited (RIXS) states.
The output file is produced by a developer version of Qchem 6.1, modified to print out state and A->B/B->A transition density matrices.
The script aims at reading the calculation data and export it as a dictionary.
'''

__author__ = 'Emanuele Rossi'
__version__ = '2.0'
__version_date__ = 'January 2025'


def ao_to_mo(C,DM_ao):
    '''
    Converts the DM from ao basis to mo basis.

    Arguments: MO coefficients matrix C (ao,mo), density matrix AO basis (DM_ao)
    Performs: C^T*DM_ao*C = DM_mo -> (mo,ao)(ao,ao)(ao,mo) = (mo,mo)
    Returns: DM_mo (mo,mo)
    '''
    return np.einsum('jp,jk,kq->pq',C,DM_ao,C)


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

    file_content = {'state': {'state_labels' : [], 
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


    #State density matrices output checkpoints
    DM_out = False

    #AO<->MO transformation matrix - checkpoint and storing variables
    MO_mat_out = False
    MO_list_sub = []
    MO_list = []

    #1-photon: block checkpoint variables
    block_A = False
    block_B = False
    block_C = False
    block_AB = False

    #Transition density matrices output checkpoint
    AB_mat_out = False
    BA_mat_out = False

    #2-photon: RIXS and REXS output checkpoint variables
    RIXS_out = False   #Output of RIXS tensor (relative to a couple of frequency points)
    REXS_out = False   #Output of REXS tensor (relative to an omega_p point)

    #2-photon: RIXS TM A->B, B->A lists initialisation
    RIXS_TM_AB = []
    RIXS_TM_BA = []

    with open(file,'r') as f: 

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
                file_content['state']['state_labels'] = ['GS']
                label_list = ['GS']

            elif 'Excited state properties' in line:

                if 'CVS' in line:
                    label_list.append('CE-'+line.split('  ')[1].split(' ')[2].strip())
                    file_content['state']['state_labels'].append('CE-'+line.split('  ')[1].split(' ')[2].strip())
                    file_content['state']['num_core_states'] += 1
                else:
                    label_list.append('VE-'+line.split('  ')[1].split(' ')[2].strip())
                    file_content['state']['state_labels'].append('VE-'+line.split('  ')[1].split(' ')[2].strip())
                    file_content['state']['num_val_states'] += 1

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
                file_content['state']['state_dm'].append(ao_to_mo(C,np.array(np.array(DM_state_list).reshape((mat_dim,mat_dim)))))

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
                file_content['transition']['block_A']['state_labels'].append('GS<->' + line.split(':')[2].strip())
            
            if block_AB == True and 'State B: cvs_eomee_ccsd/rhfref/singlets:' in line:
                block_B = True
                block_A = False
                file_content['transition']['block_B']['state_labels'].append('GS<->' + line.split(':')[2].strip())

            if 'State A: cvs_eomee_ccsd/rhfref/singlets:' in line:
                block_AB = False
                block_B = False
                block_C = True
                block_C_stateA = line.split(':')[2].strip()

            if block_C == True and 'State B: cvs_eomee_ccsd/rhfref/singlets:' in line:
                file_content['transition']['block_C']['state_labels'].append(block_C_stateA + '<->' + line.split(':')[2].strip())


            #BLOCK A: data acquisition - transition dipoles + energies
            
            if block_A == True and 'Energy GAP' in line:
                file_content['transition']['block_A']['tr_energies'].append(float(line.split('=')[2].strip().replace('eV','').strip()))
                A_dp_line = count + 2

            if block_A == True and 'A->B:' in line and count == A_dp_line:
                file_content['transition']['block_A']['AB_dipole'].append(dipole_moment_processing(line))

            elif block_A == True and 'B->A:' in line and count == A_dp_line+1:
                file_content['transition']['block_A']['BA_dipole'].append(dipole_moment_processing(line))

            #BLOCK A: data acquisition - transition density matrices

            if block_A == True and 'A->B TDM-S' in line:
                TDM_line_count = -1  #skipping matrix header
                TDM_temp = []
                AB_mat_out = True
                    
            elif block_A == True and 'A->B TDM-E' in line:
                AB_mat_out = False
                file_content['transition']['block_A']['AB_tdm'].append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

            if block_A == True and 'B->A TDM-S' in line:
                TDM_line_count = -1
                TDM_temp = []
                BA_mat_out = True
                    
            elif block_A == True and 'B->A TDM-E' in line:
                BA_mat_out = False
                file_content['transition']['block_A']['BA_tdm'].append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

            if block_A == True and (AB_mat_out == True or BA_mat_out == True):

                if TDM_line_count > 0:
                    TDM_temp+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

                TDM_line_count += 1

            #BLOCK B: data acquisition - transition dipoles + energies

            if block_B == True and 'Energy GAP' in line:
                file_content['transition']['block_B']['tr_energies'].append(float(line.split('=')[2].strip().replace('eV','').strip()))
                B_dp_line = count + 2

            if block_B == True and 'A->B:' in line and count == B_dp_line:
                file_content['transition']['block_B']['AB_dipole'].append(dipole_moment_processing(line))

            elif block_B == True and 'B->A:' in line and count == B_dp_line+1:
                file_content['transition']['block_B']['BA_dipole'].append(dipole_moment_processing(line))

            #BLOCK B: data acquisition - transition density matrices

            if block_B == True and 'A->B TDM-S' in line:
                TDM_line_count = -1
                TDM_temp = []
                AB_mat_out = True
                    
            elif block_B == True and 'A->B TDM-E' in line:
                AB_mat_out = False
                file_content['transition']['block_B']['AB_tdm'].append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

            if block_B == True and 'B->A TDM-S' in line:
                TDM_line_count = -1
                TDM_temp = []
                BA_mat_out = True
                    
            elif block_B == True and 'B->A TDM-E' in line:
                BA_mat_out = False
                file_content['transition']['block_B']['BA_tdm'].append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

            if block_B == True and (AB_mat_out == True or BA_mat_out == True):

                if TDM_line_count >0:
                    TDM_temp+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

                TDM_line_count += 1

            #BLOCK C: data acquisition - transition dipoles + energies

            if block_C == True and 'Energy GAP' in line:
                file_content['transition']['block_C']['tr_energies'].append(float(line.split('=')[2].strip().replace('eV','').strip()))
                C_dp_line = count + 2

            if block_C == True and 'A->B:' in line and count == C_dp_line:
                file_content['transition']['block_C']['AB_dipole'].append(dipole_moment_processing(line))

            elif block_C == True and 'B->A:' in line and count == C_dp_line+1:
                file_content['transition']['block_C']['BA_dipole'].append(dipole_moment_processing(line))

            #BLOCK C: data acquisition - transition density matrices

            if block_C == True and 'A->B TDM-S' in line:
                TDM_line_count = -1
                TDM_temp = []
                AB_mat_out = True
                    
            elif block_C == True and 'A->B TDM-E' in line:
                AB_mat_out = False
                file_content['transition']['block_C']['AB_tdm'].append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

            if block_C == True and 'B->A TDM-S' in line:
                TDM_line_count = -1
                TDM_temp = []
                BA_mat_out = True
                    
            elif block_C == True and 'B->A TDM-E' in line:
                BA_mat_out = False
                file_content['transition']['block_C']['BA_tdm'].append(ao_to_mo(C,np.array(np.array(TDM_temp).reshape((mat_dim,mat_dim)))))

            if block_C == True and (AB_mat_out == True or BA_mat_out == True):

                if TDM_line_count >0:
                    TDM_temp+=list(map(lambda y: float(y), filter(None,map(lambda x: x.strip(),line.split(' ')))))

                TDM_line_count += 1
            
            #####################
            #2-photon properties#
            #####################

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

    #Reshaping RIXS TM's to (num_val_states,omega_p,3,3) and REXS TMs to (omega_p,3,3)

    file_content['transition']['block_A']['REXS_tm'] = np.array(file_content['transition']['block_A']['REXS_tm'],dtype='complex64').reshape(len(file_content['transition']['block_A']['RIXS_grid_p']),3,3)
    file_content['transition']['block_A']['RIXS_AB_tm'] = np.array(RIXS_TM_AB,dtype='complex64').reshape(file_content['state']['num_val_states'],len(file_content['transition']['block_A']['RIXS_grid_p']),3,3)
    file_content['transition']['block_A']['RIXS_BA_tm'] = np.array(RIXS_TM_BA,dtype='complex64').reshape(file_content['state']['num_val_states'],len(file_content['transition']['block_A']['RIXS_grid_p']),3,3)

    return file_content