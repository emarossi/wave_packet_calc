# AXWP: Attosecond X-Ray Wave Packet - v1.0
This code simulates the electronic dynamics in a neutral molecule triggered by an attosecond X-Ray pulse. The code wraps around the Qchem quantum chemistry package, which is used to calculate the molecular electronic structure and its properties.

## Theoretical model

The dynamic molecular state $\ket{\Psi(t)}$ is expanded as
<br><br>
$$\ket{\Psi(t)} = a_g(t)\ket{\psi_g} + \sum_c a_c(t)e^{-i(\omega_c-i\frac{\Gamma_v}{2})t}\ket{\psi_c} + \sum_v a_v(t)e^{-i(\omega_v-i\frac{\Gamma_v}{2})t}\ket{\psi_v}$$.
<br><br>
Here, $\ket{\psi_g}$, $\ket{\psi_c}$ and $\ket{\psi_v}$ represent the ground, core-excited and valence-excited states, respectively. The $\Gamma$ terms in the exponents correspond to the states' decay rates, which are inversely proportional to their lifetime. The expressions for the 'a(t)' coefficients (i.e. the probability amplitudes) are derived within time-dependent perturbation theory. The core transition amplitudes are approximated at the first perturbative order, while the valence transition amplitudes at the second perturbative order. The wave packet expansion is used to obtain the density matrix of the system. The density matrix is the final output of the code.

## Numerical implementation 

At each time step, the code calculates a single (i.e. for $a_c(t)$) and a double (i.e. for $a_v(t)$) integral on a frequency grid. The integrands consider molecular properties (i.e. transition energies, transition dipole moments and polarizabilities) obtained at the EOM-CC level of theory from Qchem 6.1. The polarizabilities (i.e. the RIXS transition moments) are defined on a 2D frequency grid, which poses challenges in the numerical integration. The frequency axes of the grid are called $\omega_p$ and $\omega_d$. The Qchem calculation considers only the $\omega_p$ dependence of the RIXS transition moment. Its dependence on $\omega_d$ is linear and ininfluent on the result of the calculation. The code interpolates through $\omega_p$-dependent data output by Qchem. The interpolated function is then used to generate the 2D RIXS transition moment on a custom frequency grid. The main advantage of this solution is that it avoids lengthy quantum chemical calculations to obtain dense frequency grids, necessary for accurate calculations. The integrands are poroportional to a factor
<br><br>
$$\frac{1}{\omega-\omega_p-i\frac{\Gamma}{2}}$$.
<br><br>
The numerical integration requires that the grid step size is comparable to/smaller than $\frac{\Gamma}{2}$. While the integration is affordable for $\Gamma\equiv\Gamma_c$, this is not the case for $\Gamma\equiv\Gamma_v$. This code tackles this problem by splitting the integration in two parts, depending on the 'frequency region' on the grid. In regions where $\omega-\omega_p\ll 0$ or $\gg 0$ the integration is performed numerically on computationally affordable grid. In regions where $\omega-\omega_p\approx 0$, the integral is approximated analytically.

## Pulse parameters

The code allows free choice of the following pulse parameters:
<ol>
  1. bandwidth<br>
  2. central frequency<br>
  3. polarisation (linear and circular)
</ol>
Changing between different edges of a same molecule requires obtaining the relative molecular properties from Qchem.

## Dependencies

The following packages are required: 
<ol>
  - multiprocessing<br>
  - cmath, math, time<br>
  - numpy, scipy, matplotlib<br>
  - itertools<br>
  - numexpr<br><br>
</ol>

## Work in progress

Version 2.0: object-oriented reformulation, integration of electronic density observable, separate pulse object.
