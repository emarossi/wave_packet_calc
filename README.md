# wave_packet_calc
This folder contains the code necessary to calculate the density matrix of a wave-packet excited in a molecule by an attosecond X-Ray pulse.
The code calculates the coherences and populations of the density matrix expanded within time-dependent perturbation theory (TDPT) up to second order.
The dynamical state created by the interaction with the X-Ray pulse is expanded in terms of the ground-, core-excited and valence-excited stationary states of the molecule.
The molecular properties (i.e. transition energies, transition dipole moments and polarizabilities) are obtained from Qchem 6.1.
The code takes as input Qchem's output file, using the molecular properties to calculate the transition amplitudes relative to the core-excited (i.e. first order TDPT), the ground and valence-excited states (i.e. second order TDPT). The transition amplitudes are then used to build the time-dependent density matrix representing the dynamical state.
The attosecond X-ray pulse is implemented as a gaussian pulse, with the following parameters being fully regulable: pulse irradiance, central frequency, bandwdith/duration, polarization, time-delay. Both one-color and two-color pulse schemes are available.
