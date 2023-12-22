# wave_packet_calc
This folder contains the code necessary to calculate the density matrix of a wave-packet excited in a molecule by an attosecond XFEL pulse.
The code calculates the coherences and populations of the density matrix expanded within time-dependent perturbation theory (TDPT) up to second order.
The dynamical state created by the interaction with the light-pulse is expanded in terms of the ground-state, the core-excited and the valence-excited states of the molecule.
The molecular properties (i.e. transition energies, transition dipole moments and polarizabilities), are obtained from Qchem 6.1.
The code takes as input Qchem's output file, using the molecular properties to calculate the transition amplitudes relative to the core-excited (i.e. first order TDPT), the ground and valence-excited states (i.e. second order TDPT). The transition amplitudes are then used to build the time-dependent density matrix representing the dynamical state.
The code a two-color pulse formed by two gaussian pulses with a wide range of freely and independently regulable setting for each color. These include: pulse irradiance, central frequency, bandwdith/duration, polarization, time-delay.

PULL trial
