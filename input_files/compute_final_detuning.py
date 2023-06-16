'''
File to simulate the detuning algorithm for a given number of turns with a given beam.

Author: Birk Emil Karlsen-Baeck
'''
# Parse Arguments -----------------------------------------------------------------------------------------------------
import argparse
from lxplus_setup.parsers import lhc_llrf_argument_parser

parser = argparse.ArgumentParser(parents=[lhc_llrf_argument_parser()],
                                 description='Script to simulate LHC injection.', add_help=True,
                                 prefix_chars='~')

parser.add_argument('~~beam_name', '~bn', type=str,
                    help='Input beam for this calculation')
parser.add_argument('~~profile_length', '~pl', type=int, default=500,
                    help='Length of profile object')
parser.add_argument('~~number_of_turns', '~nt', type=int, default=500,
                    help='Number of turns to track the detuning')

args = parser.parse_args()

beam_ID = args.beam_name

# Imports -------------------------------------------------------------------------------------------------------------
print('\nImporting...')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c
import yaml

import beam_dynamics_tools.analytical_functions.longitudinal_beam_dynamics as lbd

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop

# Options -------------------------------------------------------------------------------------------------------------
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
LXPLUS = True
if 'birkkarlsen-baeck' in os.getcwd():
    lxdir = '../'
    LXPLUS = False
    print('\nRunning locally...')
else:
    print('\nRunning in lxplus...')

# Import Settings from Generation -------------------------------------------------------------------------------------
with open(f'{lxdir}generated_beams/{beam_ID}/generation_settings.yaml') as file:
    gen_dict = yaml.full_load(file)

# Parameters ----------------------------------------------------------------------------------------------------------
# Accelerator parameters
C = 26658.883                                   # Machine circumference [m]
p_s = 450e9                                     # Synchronous momentum [eV/c]
h = 35640                                       # Harmonic number [-]
gamma_t = args.gamma_t                          # Transition gamma [-]
alpha = 1./gamma_t/gamma_t                      # First order mom. comp. factor [-]
V = args.voltage * 1e6                          # RF voltage [V]
dphi = 0                                        # Phase modulation/offset [rad]

# RFFB parameters
G_a = args.analog_gain                          # Analog FB gain [A/V]
G_d = args.digital_gain                         # Digital FB gain [-]
tau_loop = args.loop_delay                      # Overall loop delay [s]
tau_a = args.analog_delay                       # Analog FB delay [s]
tau_d = args.digital_delay                      # Digital FB delay [s]
a_comb = args.comb_alpha                        # Comb filter alpha [-]
tau_otfb = args.otfb_delay                      # LHC OTFB delay [s]
G_o = args.otfb_gain                            # LHC OTFB gain [-]
Q_L = args.loaded_q                             # Loaded Quality factor [-]
mu = float(args.detuning_mu)                    # Tuning algorithm coefficient [-]
df = args.delta_frequency                       # Initial detuning frequnecy [Hz]

# Beam parameters
N_p = gen_dict['bunch intensity']               # Average Bunch intensity [p/b]
N_buckets = args.profile_length                 # Length of profile [RF buckets]
N_bunches = gen_dict['number of bunches']       # Total number of bunches
N_mpb = gen_dict['macroparticles per bunch']    # Number of macroparticles
N_m = N_mpb * N_bunches
N_p *= N_bunches

# Objects for simulation ----------------------------------------------------------------------------------------------
print('\nInitializing Objects...')
# LHC ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=args.number_of_turns)

# 400MHz RF station
rfstation = RFStation(ring, [h], [V], [dphi])

# Beam
if bool(args.simulated_beam):
    imported_beam = np.load(lxdir + f'generated_beams/{beam_ID}/simulated_beam.npy')
else:
    imported_beam = np.load(lxdir + f'generated_beams/{beam_ID}/generated_beam.npy')

beam = Beam(ring, len(imported_beam[1, :]), N_p)

ddt = 1000 * rfstation.t_rf[0, 0]
Dt = (((2 * np.pi * lbd.R_SPS)/(lbd.h_SPS * c * lbd.beta)) - rfstation.t_rf[0, 0])/2
beam.dE = imported_beam[1, :] + args.energy_error * 1e6
beam.dt = imported_beam[0, :] - Dt + ddt + args.phase_error / 360 * rfstation.t_rf[0, 0]

# Beam Profile
profile = Profile(beam, CutOptions(cut_left=-1.5 * rfstation.t_rf[0, 0] + ddt,
                                   cut_right=(N_buckets + 1.5) * rfstation.t_rf[0, 0] + ddt,
                                   n_slices=(N_buckets + 3) * 2**7))
# Modify cuts of the Beam Profile
beam.statistics()
profile.cut_options.track_cuts(beam)
profile.set_slices_parameters()
profile.track()

# LHC Cavity Controller
print(mu, type(mu))
RFFB = LHCRFFeedback(G_a=G_a, G_d=G_d, tau_d=tau_d, tau_a=tau_a, alpha=a_comb, mu=mu, G_o=G_o,
                     clamping=False)

CL = LHCCavityLoop(rfstation, profile, RFFB=RFFB,
                   f_c=rfstation.omega_rf[0, 0]/(2 * np.pi) + df,
                   Q_L=Q_L, tau_loop=tau_loop, n_pretrack=100, tau_otfb=tau_otfb)

# RF tracker object
rftracker = RingAndRFTracker(rfstation, beam, Profile=profile, interpolation=True,
                             CavityFeedback=CL)

LHC_tracker = FullRingAndRF([rftracker])

# Simulating ----------------------------------------------------------------------------------------------------------
print('\nSimulating...')

detunings = np.zeros(args.number_of_turns + 1)
detunings[0] = CL.d_omega / (2 * np.pi)

for i in range(args.number_of_turns):
    CL.track()
    detunings[i + 1] = CL.d_omega / (2 * np.pi)
    if i == 0:
        print('Beam Injected! - For-loop successfully entered')

    if i % 50 == 0:
        print(f'Turn {i}')

print(f'Total beam intensity is {beam.intensity / 1e11} * 1e11 protons')
print(f'so there are {beam.intensity / N_bunches / 1e11} * 1e11 p/b ')
print(f'Final detuning was {CL.d_omega / (2 * np.pi)} Hz')

plt.figure()
plt.title('Profile')
plt.plot(profile.bin_centers, profile.n_macroparticles * beam.ratio)

plt.figure()
plt.title('Detuning')
plt.plot(detunings)

plt.show()
