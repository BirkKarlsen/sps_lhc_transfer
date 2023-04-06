

import numpy as np
import matplotlib.pyplot as plt
import os

from simulation_functions.machine_beam_processes import fetch_momentum_program, plot_momentum_program
from blond.beam.beam import Proton
from analytical_functions.longitudinal_beam_dynamics import rf_bucket_height
from blond.llrf.signal_processing import smooth_step


mom_dir = f'../momentum_programs/LHC_momentum_programme_6.8TeV.csv'
mom_cut = 451e9
p_flatbottom = 450e9
V_cap = 7e6

E_sep = rf_bucket_height(V=V_cap)
mom_targ = p_flatbottom + 2 * E_sep
print(f'Bucket height at LHC injection with RF voltage of {V_cap * 1e-6:.2f} MV is {E_sep / 1e6:.1f} MeV')
print(f'Need to accelerate to at least {mom_targ/1e9:.3f} GeV')

momentum = fetch_momentum_program(fname=mom_dir, C=26658.883,
                                  particle_mass=Proton().mass, target_momentum=mom_cut)

print(f'Number of turns needed for ramp to {mom_cut/1e9:.3f} GeV is {len(momentum)}')

plot_momentum_program(momentum, mom_targ)

plt.show()


