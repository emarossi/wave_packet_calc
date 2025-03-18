<h1>AXWP: Attosecond X-Ray Wave Packet - v1.0</h1> 
This code simulates the electronic dynamics in a neutral molecule triggered by an attosecond X-Ray pulse. The code wraps around the Qchem quantum chemistry package, which is used to calculate the molecular electronic structure and its properties.

<h2>Theoretical model</h2>

The molecular state $\ket{\Psi(t)}$ is expanded as
<br><br>
$$\ket{\Psi(t)} = (1+a_g^{(2)}(t))\ket{\psi_g} + \sum_c a_c^{(1)}(t)e^{-i(\omega_c-i\frac{\Gamma_v}{2})t}\ket{\psi_c} + \sum_v a_v^{(2)}(t)e^{-i(\omega_v-i\frac{\Gamma_v}{2})t}\ket{\psi_v}$$.
<br><br>
Here, $\ket{\psi_g}$, $\ket{\psi_c}$ and $\ket{\psi_v}$ represent the ground, core-excited and valence-excited states, respectively. The $\Gamma$ terms in the exponents correspond to the states' decay rates, which are inversely proportional to their lifetime. The expressions for the probability amplitudes 'a(t)' are derived within time-dependent perturbation theory. As indicated by the superscripts, the core amplitudes are approximated at the first perturbative order, while the valence amplitudes at the second perturbative order. The probability amplitudes are combined to obtained the density matrix, which is the final output of the code. 

<h2>Numerical implementation</h2>
<p>
  The probability amplitudes $a_c^{(1)}(t)$ are calculated by computing the expression
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/c5a6a925-1167-4c56-8ccd-04f81b444129"><br>
  <em>Equation 1: core-excited state probability amplitudes.</em>
</p>
<p>
The excitation energy, $\omega_{cg}$, and the transition dipole moment, $\mu_{cg}$, are calculated by the quantum chemistry package. The integration is performed numerically on a frequency array, choosing $\Gamma_c$ according to the selected excitation edge. The probability amplitudes $a_k^{(2)}(t)$, with $k\in [g,v]$, are calculated according to the expression  
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/cbd34a45-588e-4a4d-a196-2ce3a5d18d7d"><br>
  <em>Equation 2: valence-/ground-state probability amplitudes.</em>
</p>
<p>
Here, the term $M_{kg}(\omega_p,\omega_d)$ represents the RIXS transition moment, obtained at the EOM-CC level of theory. Qchem considers only the $\omega_p$ dependence $M_{kg}(\omega_p,\omega_d)$, neglecting its linear dependence on $\omega_d$. Qchem calculates $M_{kg}(\omega_p,\omega_d)$ over an interval of $\omega_p$, fixing $\omega_d$ to the first value of the $\omega_p$ vector. The code interpolates the $\omega_p$-dependent data from Qchem; the interpolating function is used to generate $M_{kg}(\omega_p,\omega_d)$ over a square frequency grid of choice, with the $\omega_p$-dependent array being 'repeated' for each value of $\omega_d$.
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/0a115e10-c067-4af1-80fd-72a825236ff4" width="300"><br>
  <em>Figure 1: frequency grid division in terms of integration style: grey = numerical; blue = analytical. The red line indicates the strip's center, where the resonance condition $\omega_{kg}-\omega_p+\omega_d = 0$ is satisfied. The green segment indicates the discrete "strip element" the strip is divided into.</em>
</p>
<p>
Performing the double integral in eq.2 numerically is very difficult, either because, when k=v, and $\Gamma_v$ is small and requires a large grid, or because, when k=g, $\Gamma_g=0$ and the integral diverges. The code tackles this problem by integrating numerically only in the 'grey' areas of the frequency grid in fig.1 - where $\omega-\omega_p\ll 0$ or $\gg 0$ - where the properties of the integrand allow to use a computationally affordable grid. The regions where $\omega-\omega_p\approx 0$ (the blue 'strip' in fig.1), are divided into discrete "strip elements" (the green segments in fig. 1), where the integral in eq.2 is approximated in the limit $\Gamma_k\to 0$ as
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/38c02274-b8fc-48af-a1f1-51c216cb1e51")<br><br>
  <em>Eq. 3: valence-/ground-state probability amplitude approximation on strip element.</em>
</p>
<p>
Here, the function $f(\omega _0)$ corresponds to the terms highlighted in red in equation 2, which is Taylor-expanded up to first order. $\omega_0$ corresponds to the centre of the strip element (i.e. intersection of green and red lines in fig. 1), while $\delta$ to the strip's half width. The code calculates the contributions from all the discrete strip elements, summing them together to obtain an approximation of the integral in equation 2 on the "blue strip". The contributions from the strip are then summed to those obtained from the numerical integration over the "grey parts" of the grid, leading to the approximate value of $a_k^{(2)}(t)$.  
</p>

<h2> Implementation of the "strip" approximation </h2>

The solution in eq.3 is implemented on each strip element, whose number depends on the frequency grid chosen by the user. The derivative at $\omega_0$ is estimated numerically as the difference quotient involving the upper and lower limits of the strip. To converge to the correct qualitative behaviour of $a_k^{(2)}(t)$, the parameter $\delta$, which controls the width of "strip", and the finesse of the frequency grid need to be optimised. The correct behaviour of $a_k^{(2)}(t)$ can be estimated by monitoring the corresponding populations, $|a_k^{(2)}(t)|^2$. An example of "non-converged" behaviour is shown in figure 2, where in some cases $|a_k^{(2)}(t)|^2$ diverges to infinity is are non-zero before the pulse.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d09d1d5b-092e-4a40-a6e0-aebf394630dc")
  <em>Figure 2: $|a_k^{(2)}(t)|^2$ in a "non-converged" calculation for a pulse centred around 0 as. Here, the chosen $\delta$ and frequency grid finesse are not ideal. </em>
</p>
<p>
  A "converged" behaviour can be obtained by fine-tuning the $\delta$ and frequency grid size until a result similar to fig. 3 is obtained. Here, $|a_k^{(2)}(t)|^2$ shows the qualitatively correct behaviour, not diverging to infinity and being zero before the pulse arrives at 0 as.
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/cb904fdc-c1c3-409c-919f-04bf8f08d0f9")
  <em>Figure 2: $|a_k^{(2)}(t)|^2$ in a "converged" calculation for a pulse centred around 0 as. Here, the chosen $\delta$ and frequency grid finesse are the ideal ones. </em>
</p>
<p>
The current implementation imposes the same $\delta$ (set inside the code) and frequency grid size (input to the script) to approximate eq. 2 for all the considered ground/valence-excited states. In testing, this led in some case to a slight overestimation/underestimation of $|a_k^{(2)}(t)|^2$. However, the correct qualitative behaviour is achieved across all the considered molecules, excitation edges and pulse parameters examined. 
</p>
<h3> Improvements - in progress </h3>
<ol>
  <li> Implement selection of $\delta$ and grid step size for groung and each valence-excited state. </li>
  <li> Add higher orders to Taylor expansion in eq. 3. </li>
  <li> Implement option for analytical derivative (for Gaussian pulses). </li>
</ol>

<h2> Dependencies </h2>

The following packages are required: 
<ol>
  - multiprocessing<br>
  - cmath, math, time<br>
  - numpy, scipy, matplotlib<br>
  - itertools<br>
  - numexpr<br><br>
</ol>

<h2> Calculation setup </h2>
<p>
A Qchem calculation with the sample SRIXS and XAS input files must be performed. The output files must end with '-XAS.out' and '-SRIXS.out', respectively. The wave packet calculation can be started by performing
</p>
<p>
<code> python3 wp_calc.py 'path_Qchem_out' '#colors' C1_en C2_en C1_bw C2_bw 'C1_pol' 'C2_pol' #freq_grid 'output_array' </code>  
</p>
<p>
The keys have the corresponding meaning
<ol>
  <li>path_Qchem_out: name of output files before ending suffix</li>
  <li>#colors: '1C'= 1 color Gaussian, '2C' = 2 color Gaussian </li>
  <li> C1_en/C2_en: color1/color2 central energy (eV)</li>
  <li> C1_bw/C2_bw: color1/color2 bandwidth (eV)</li>
  <li> C1_pol/C2_pol: color1/color2 polarisation. Options are:
    <ul>
      <li>linear polarised: 'x', 'z', 'xz'</li>
      <li>circularly polarised: 'Rxy', 'Lxy'</li>
    </ul>
  </li>
  <li> #freq_grid: number of points frequecy grid with shape (#freq_grid,#freq_grid)</li>
  <li> output_array: path+name output</li>
</ol>
</p>
<p>
Output contains python dictionary with entries:
  <ul>
    <li>'Density Matrix': density matrix of shape=(#states,#states), where #states is ordered as gs, valence-excited, core-excited.</li>
    <li>'1PDM': 1PDMs array of shape=(#states,#states,#MO,#MO); state 1PDM diagonal axes 0,1; transition 1PDM off-diagonal axes 0,1.</li>
    <li>'time_array': array of considered time points.</li>
    <li>'pulse_time': pulse in the time domain.</li>
    <li>'#_val_states': number valence-excited states + 1 (gs).</li>
    <li>'#_core_states': number of core-excited states.</li>
  </ul>
</p>
<h3>Improvements - in progress</h3>
<ul>
  <li>Parameters input via input file:
  <ul>
    <li>Section for pulse parameters</li>
    <li>Section for integration and strip parameters</li>
  </ul>
  </li>
  <li>Separate module for pulses</li>
</ul>
