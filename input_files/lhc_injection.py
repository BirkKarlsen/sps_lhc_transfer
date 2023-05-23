'''
Simulation of LHC flattop with a beam generated in the SPS.

Author: Birk Emil Karlsen-Bæck
'''
# Parse Arguments -----------------------------------------------------------------------------------------------------
import argparse
from lxplus_setup.parsers import simulation_argument_parser, lhc_llrf_argument_parser

parser = argparse.ArgumentParser(parents=[simulation_argument_parser(), lhc_llrf_argument_parser()],
                                 description='Script to simulate LHC injection.', add_help=True,
                                 prefix_chars='~')

parser.add_argument('~~date', '~dte', type=str,
                    help='Input date of the simulation; if none is parsed then todays date will be taken')

args = parser.parse_args()

beam_ID = args.beam_name

# Imports -------------------------------------------------------------------------------------------------------------
print('\nImporting...')
import numpy as np
import os
from scipy.constants import c
import yaml
from datetime import date

import beam_dynamics_tools.analytical_functions.longitudinal_beam_dynamics as lbd
from beam_dynamics_tools.simulation_functions.diagnostics_functions import LHCDiagnostics
from beam_dynamics_tools.simulation_functions.machine_beam_processes import fetch_momentum_program
from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
from blond.llrf.beam_feedback import BeamFeedback
from blond.impedances.impedance_sources import InputTable
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage

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
mu = args.detuning_mu                           # Tuning algorithm coefficient [-]
df = args.delta_frequency                       # Initial detuning frequnecy [Hz]

# Beam parameters
N_p = gen_dict['bunch intensity']               # Average Bunch intensity [p/b]
N_buckets = args.profile_length                 # Length of profile [RF buckets]
N_bunches = gen_dict['number of bunches']       # Total number of bunches
N_mpb = gen_dict['macroparticles per bunch']    # Number of macroparticles
N_m = N_mpb * N_bunches
N_p *= N_bunches

# Simulation parameters
N_t = args.number_of_turns                      # Number of turns

# LHC ramp
if args.ramp is not None:
    ramp = fetch_momentum_program(lxdir + 'momentum_programs/' + args.momentum_program,
                                  C=C, particle_mass=Proton().mass, target_momentum=args.ramp * 1e9)
    cycle = np.concatenate((np.linspace(p_s, p_s, N_t), ramp))
    N_t = len(cycle) - 1
    print(f'Ramp length is {len(ramp)} turns')
else:
    cycle = np.linspace(p_s, p_s, N_t + 1)

# Objects for simulation ----------------------------------------------------------------------------------------------
print('\nInitializing Objects...')
# LHC ring
ring = Ring(C, alpha, cycle, Proton(), n_turns=N_t)

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

# Impedance model
if bool(args.include_impedance):
    n_necessary = 57418             # Necessary indices to keep when we want to resolve up to 50 GHz
    imp_data = np.loadtxt(lxdir + 'impedance/' + args.impedance_model, skiprows=1)
    imp_table = InputTable(imp_data[:n_necessary, 0], imp_data[:n_necessary, 1], imp_data[:n_necessary, 2])

    ind_volt_freq = InducedVoltageFreq(beam, profile, [imp_table])
    total_Vind = TotalInducedVoltage(beam, profile, [ind_volt_freq])
else:
    total_Vind = None

# LHC Cavity Controller
RFFB = LHCRFFeedback(G_a=G_a, G_d=G_d, tau_d=tau_d, tau_a=tau_a, alpha=a_comb, mu=mu, G_o=G_o,
                     clamping=False)

CL = LHCCavityLoop(rfstation, profile, RFFB=RFFB,
                   f_c=rfstation.omega_rf[0, 0]/(2 * np.pi) + df,
                   Q_L=Q_L, tau_loop=tau_loop, n_pretrack=100, tau_otfb=tau_otfb)

# LHC beam-phase loop and synchronization loop
if args.pl_gain is None:
    PL_gain = 1 / (5 * ring.t_rev[0])
else:
    PL_gain = args.pl_gain

if args.sl_gain is None:
    SL_gain = PL_gain / 10
else:
    SL_gain = args.sl_gain

if bool(args.include_global):
    bl_config = {'machine': 'LHC',
                 'PL_gain': PL_gain,
                 'SL_gain': SL_gain}
    BL = BeamFeedback(ring, rfstation, profile, bl_config)
else:
    BL = None

# RF tracker object
rftracker = RingAndRFTracker(rfstation, beam, Profile=profile, interpolation=True,
                             CavityFeedback=CL, BeamFeedback=BL,
                             TotalInducedVoltage=total_Vind)

LHC_tracker = FullRingAndRF([rftracker])

# Simulating ----------------------------------------------------------------------------------------------------------
print('\nSimulating...')

# Make simulation output folder
if args.date is None:
    today = date.today()
    save_to = lxdir + f'simulation_results/{today.strftime("%b-%d-%Y")}/{args.simulation_name}/'
    if not os.path.isdir(save_to):
        os.makedirs(save_to)
else:
    save_to = lxdir + f'simulation_results/{args.date}/{args.simulation_name}/'
    if not os.path.isdir(save_to):
        os.makedirs(save_to)

# Fetch injection scheme
injection_scheme = fetch_from_yaml(args.scheme, lxdir + 'injection_schemes/')

# Setting diagnostics function
diagnostics = LHCDiagnostics(rftracker, profile, total_Vind, CL, ring, save_to, lxdir, N_bunches,
                             injection_scheme=injection_scheme, setting=args.diag_setting, dt_cont=args.dt_cont,
                             dt_beam=args.dt_beam, dt_cl=args.dt_cl, dt_prfl=args.dt_prfl, dt_ld=args.dt_ld)

# Main for loop
for i in range(N_t):
    LHC_tracker.track()
    profile.track()
    if args.include_impedance:
        total_Vind.induced_voltage_sum()

    diagnostics.track()

    if i == 0:
        print('Beam Injected! - For-loop successfully entered')

