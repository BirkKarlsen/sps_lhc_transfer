'''
Simulation of LHC flattop with a beam generated in the SPS.

Author: Birk Emil Karlsen-Bæck
'''
# Parse Arguments -----------------------------------------------------------------------------------------------------
import argparse
from lxplus_setup.parsers import simulation_argument_parser, lhc_llrf_argument_parser

parser = argparse.ArgumentParser(parents=[simulation_argument_parser(), lhc_llrf_argument_parser()],
                                 description='Script to simulate LHC injection.', add_help=True)

args = parser.parse_args()

beam_ID = args.beam_name

# Imports -------------------------------------------------------------------------------------------------------------
print('\nImporting...')
import numpy as np
import os
from scipy.constants import c
import yaml

import analytical_functions.longitudinal_beam_dynamics as lbd
from simulation_functions.diagnostics_functions import LHCDiagnostics

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
from blond.impedances.impedance_sources import InputTable
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage

# Options -------------------------------------------------------------------------------------------------------------
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
LXPLUS = True
if not lxdir in os.getcwd():
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
Q_L = args.loaded_q                             # Loaded Quality factor [-]
mu = args.detuning_mu                           # Tuning algorithm coefficient [-]
df = args.delta_frequency                       # Initial detuning frequnecy [Hz]

# Beam parameters
N_p = gen_dict['bunch intensity'] * 1e11        # Average Bunch intensity [p/b]
N_buckets = args.profile_length                 # Length of profile [RF buckets]
N_bunches = gen_dict['number of bunches']       # Total number of bunches
N_mpb = gen_dict['macroparticles per bunch']    # Number of macroparticles
N_m = N_mpb * N_bunches
N_p *= N_bunches

# Simulation parameters
N_t = args.number_of_turns                      # Number of turns

# Objects for simulation ----------------------------------------------------------------------------------------------
print('\nInitializing Objects...')
# LHC ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)

# 400MHz RF station
rfstation = RFStation(ring, [h], [V], [dphi])

# Beam
beam = Beam(ring, N_m, N_p)
if bool(args.simulated_beam):
    imported_beam = np.load(lxdir + f'generated_beams/{beam_ID}/simulated_beam.npy')
else:
    imported_beam = np.load(lxdir + f'generated_beams/{beam_ID}/generated_beam.npy')

Dt = (((2 * np.pi * lbd.R_SPS)/(lbd.h_SPS * c * lbd.beta)) - rfstation.t_rf[0, 0])/2
beam.dE = imported_beam[0, :]
beam.dt = imported_beam[1, :] + Dt

# Beam Profile
profile = Profile(beam, CutOptions(cut_left=-0.5 * rfstation.t_rf[0, 0],
                                   cut_right=(N_buckets + 0.5) * rfstation.t_rf[0, 0],
                                   n_slices=(N_buckets + 1) * 2**7))
profile.track()

# Impedance model
if args.include_impedance:
    n_necessary = 57418             # Necessary indices to keep when we want to resolve up to 50 GHz
    imp_data = np.loadtxt(lxdir + 'impedance/Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat', skiprows=1)
    imp_table = InputTable(imp_data[:n_necessary, 0], imp_data[:n_necessary, 1], imp_data[:n_necessary, 2])

    ind_volt_freq = InducedVoltageFreq(beam, profile, [imp_table])
    total_Vind = TotalInducedVoltage(beam, profile, [ind_volt_freq])
else:
    total_Vind = None

# LHC Cavity Controller
RFFB = LHCRFFeedback(G_a=G_a, G_d=G_d, tau_d=tau_d, tau_a=tau_a, alpha=a_comb, mu=mu)

CL = LHCCavityLoop(rfstation, profile, RFFB=RFFB,
                   f_c=rfstation.omega_rf[0, 0]/(2 * np.pi) + df,
                   Q_L=Q_L, tau_loop=tau_loop, n_pretrack=100, tau_otfb=tau_otfb)

# RF tracker object
rftracker = RingAndRFTracker(rfstation, beam, Profile=profile, interpolation=True,
                             CavityFeedback=CL, TotalInducedVoltage=total_Vind)

LHC_tracker = FullRingAndRF([rftracker])

# Simulating ----------------------------------------------------------------------------------------------------------
print('\nSimulating...')

# Setting diagnostics function
diagnostics = LHCDiagnostics(rftracker, profile, total_Vind, CL, args.save_to, N_bunches,
                             setting=args.diag_setting, dt_cont=args.dt_cont,
                             dt_beam=args.dt_beam, dt_cl=args.dt_cl)

# Main for loop
for i in range(N_t):
    LHC_tracker.track()
    profile.track()
    total_Vind.induced_voltage_sum()
    CL.track()

    diagnostics.track()

    if i == 0:
        print('Beam Injected! - For-loop successfully entered')

