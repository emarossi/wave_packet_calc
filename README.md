<h1>AXWP: Attosecond X-Ray Wave Packet - v1.0</h1> 
This code simulates the electronic dynamics in a neutral molecule triggered by an attosecond X-Ray pulse. The code wraps around the Qchem quantum chemistry package, which is used to calculate the molecular electronic structure and its properties.

<h2>Theoretical model</h2>

The dynamic molecular state $\ket{\Psi(t)}$ is expanded as
<br><br>
$$\ket{\Psi(t)} = (1+a_g^{(2)}(t))\ket{\psi_g} + \sum_c a_c^{(1)}(t)e^{-i(\omega_c-i\frac{\Gamma_v}{2})t}\ket{\psi_c} + \sum_v a_v^{(2)}(t)e^{-i(\omega_v-i\frac{\Gamma_v}{2})t}\ket{\psi_v}$$.
<br><br>
Here, $\ket{\psi_g}$, $\ket{\psi_c}$ and $\ket{\psi_v}$ represent the ground, core-excited and valence-excited states, respectively. The $\Gamma$ terms in the exponents correspond to the states' decay rates, which are inversely proportional to their lifetime. The expressions for the 'a(t)' coefficients (i.e. the probability amplitudes) are derived within time-dependent perturbation theory. The core transition amplitudes are approximated at the first perturbative order, while the valence transition amplitudes at the second perturbative order. The wave packet expansion is used to obtain the density matrix of the system. The density matrix is the final output of the code.

<h2>Numerical implementation</h2>
<p>
  The probability amplitudes $a_c^{(1)}(t)$ are calculated by computing the expression
</p>
<figure>
  <img src="https://github.com/user-attachments/assets/f9f30d01-8a21-484b-93ab-618d34e2f43e"><br>
  <em>Eq. 1: core-excited state probability amplitudes</em>
</figure>
<p>
<br>
The excitation energy, $\omega_{cg}$, and the transition dipole moment, $\mu_{cg}$, are calculated by the quantum chemistry package. The integration is performed numerically on a frequency array, choosing $\Gamma_c$ according to the selected excitation edge. The probability amplitudes $a_k^{(2)}(t)$, with $k\in [g,v]$, are calculated according to the expression  
</p>
<figure>
  <img src="https://github.com/user-attachments/assets/d331be41-895c-4a58-a1fe-58946b860d89"><br>
  <em>Eq. 2: valence-/ground-state probability amplitudes</em>
</figure>
<p>
<br>
Here, the term $M_{kg}(\omega_p,\omega_d)$ represents the RIXS transition moment, obtained at the EOM-CC level of theory with Qchem 6.1. Qchem considers only the $\omega_p$ dependence $M_{kg}(\omega_p,\omega_d)$, as justified by its linear dependence on $\omega_d$, which doesn't influence the calculation result. The calculation evaluates $M_{kg}(\omega_p,\omega_d)$ over an interval of $\omega_p$, fixing $\omega_d$ to the first value of the $\omega_p$ vector. The code interpolates the $\omega_p$-dependent data from Qchem; the interpolating function is used to generate $M_{kg}(\omega_p,\omega_d)$ over a square frequency grid of choice, with the $\omega_p$-dependent array being 'repeated' for each value of $\omega_d$.
</p>
<figure class="inline end" markdown>
  <img src="https://github.com/user-attachments/assets/0a115e10-c067-4af1-80fd-72a825236ff4" width="300" align='right'>
</figure>
<p>
Performing the double integral numerically is very difficult, either because, when k=v, and $\Gamma_v$ is small and requires a large grid, or because, when k=g, $\Gamma_g=0$ and the integral diverges. The code tackles this problem by integrating numerically only in the areas of the frequency grid (in grey in the picture) where $\omega-\omega_p\ll 0$ or $\gg 0$, which allow using a computationally affordable grid. In regions where $\omega-\omega_p\approx 0$ (i.e. the blue 'strip' in the picture), the integral is solved analitically on each "strip element" (i.e. the green segments in the picture) according to
</p>
<figure>
  <img src="https://github.com/user-attachments/assets/38c02274-b8fc-48af-a1f1-51c216cb1e51")<br><br>
  <em>Eq. 3: valence-/ground-state probability amplitude approximation on strip element</em>
</figure>
<p>
<br>
Here, the function $f(\omega _0)$ corresponds to the terms highlighted in red in equation 2, which is Taylor-expanded up to first order. $\omega_0$ corresponds to the centre of the strip element (i.e. intersection of green and red lines in the figure), while $\delta$ to the strip's half width. The code calculates via eq. 3 the contributions from all the discrete strip elements, summing them together to obtain an approximate of the integral in equation 2 on the "blue strip". The contributions from the strip are then summed to those obtained from the numerical integration over the "grey parts" of the grid, leading to the approximate value of $a_k^{(2)}(t)$.  
</p> 

<h2> Fine tuning the "strip" approximation </h2>

<h2>Pulse parameters</h2>

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
