'''
Generation of SPS beams which either go to a SPS flattop simulation or directly to the LHC.

Author: Birk Emil Karlsen-Baeck
'''

# Parse Arguments -----------------------------------------------------------------------------------------------------
from lxplus_setup.parsers import generation_argument_parser

parser = generation_argument_parser(add_help=True)
args = parser.parse_args()

# Imports -------------------------------------------------------------------------------------------------------------
print('\nImporting...')
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from beam_dynamics_tools.simulation_functions.beam_generation_functions import generate_bunch_spacing, generate_beam_ID

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageFreq
from blond.impedances.impedance_sources import InputTable

from SPS.impedance_scenario import scenario, impedance2blond

# Parameters ----------------------------------------------------------------------------------------------------------
# Accelerator parameters
C = 2 * np.pi * 1100.009            # Machine circumference [m]
p_s = 450e9                         # Synchronous momentum [eV/c]
h = 4620                            # Harmonic number [-]
gamma_t = 18.0                      # Transition gamma [-]
alpha = 1./gamma_t/gamma_t          # First order mom. comp. factor [-]
V = args.voltage_200 * 1e6          # 200 MHz RF voltage [V]
V_800 = args.voltage_800 * V        # 800 MHz RF voltage [V]
dphi = 0                            # 200 MHz Phase modulation/offset [rad]
dphi_800 = np.pi                    # 800 MHz Phase modulation/offset [rad]


# Beam parameters
N_p = args.intensity * 1e11         # Bunch intensity [p/b]
exponent = args.exponent
bl = args.bunchlength * 1e-9
N_bunches = args.number_bunches
ps_batch_length = args.ps_batch_length
ps_batch_spacing = args.ps_batch_spacing

# Initialize the bunch
if not bool(args.custom_beam):
    bunch_lengths = bl * np.ones(N_bunches)
    exponents = exponent * np.ones(N_bunches)
    bunch_intensities = N_p / N_bunches * np.ones(N_bunches)
    bunch_positions = generate_bunch_spacing(N_bunches, 5, ps_batch_length, ps_batch_spacing, args.beam_type)

else:
    try:
        bunch_lengths = np.load(args.custom_beam_dir + 'bunch_lengths.npy')
    except:
        bunch_lengths = bl * np.ones(N_bunches)

    try:
        exponents = np.load(args.custom_beam_dir + 'exponents.npy')
    except:
        exponents = exponent * np.ones(N_bunches)

    try:
        bunch_intensities = np.load(args.custom_beam_dir + 'bunch_intensities.npy')
    except:
        bunch_intensities = N_p / N_bunches * np.ones(N_bunches)

    try:
        bunch_positions = np.load(args.custom_beam_dir + 'bunch_positions.npy')
    except:
        bunch_positions = generate_bunch_spacing(N_bunches, 5, ps_batch_length, ps_batch_spacing, args.beam_type)


# Simulation parameters
N_t = 1                             # Number of turns
N_m = args.n_macroparticles         # Number of macroparticles
N_m *= N_bunches
N_p = np.sum(bunch_intensities)
N_buckets = args.profile_length

beam_ID = generate_beam_ID(beam_type=args.beam_type, number_bunches=args.number_bunches,
                           ps_batch_length=args.ps_batch_length, intensity=args.intensity,
                           bunchlength=args.bunchlength, voltage_200=args.voltage_200,
                           voltage_800=args.voltage_800)

# Parameters for the SPS Impedance Model
freqRes = 43.3e3                                # Frequency resolution [Hz]
modelStr = "futurePostLS2_SPS_f1.txt"           # Name of Impedance Model

# Options -------------------------------------------------------------------------------------------------------------
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
LXPLUS = True
if 'birkkarlsen-baeck' in os.getcwd():
    lxdir = '../'
    LXPLUS = False
    print('\nRunning locally...')
else:
    print('\nRunning in lxplus...')

# Objects -------------------------------------------------------------------------------------------------------------
print('\nInitializing Objects...')

# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=1)

# RF Station
rfstation = RFStation(ring, [h, 4 * h], [V, V_800], [dphi, dphi_800], n_rf=2)

# Beam
beam = Beam(ring, N_m, N_p)

# Profile
profile = Profile(beam, CutOptions=CutOptions(cut_left=rfstation.t_rf[0, 0] * (-0.5),
            cut_right=rfstation.t_rf[0, 0] * (N_buckets + 0.5),
            n_slices=int(round(2 ** 7 * (1 + N_buckets)))))

# SPS Impedance Model
impScenario = scenario(modelStr)
impModel = impedance2blond(impScenario.table_impedance)

impFreq = InducedVoltageFreq(beam, profile, impModel.impedanceList, freqRes)
SPSimpedance_table = InputTable(impFreq.freq,impFreq.total_impedance.real*profile.bin_size,
                                    impFreq.total_impedance.imag*profile.bin_size)
impedance_freq = InducedVoltageFreq(beam, profile, [SPSimpedance_table],
                                       frequency_resolution=freqRes)
total_imp = TotalInducedVoltage(beam, profile, [impedance_freq])

# Tracker Object without SPS OTFB
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=None, Profile=profile, interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])


distribution_options_list = {'bunch_length': bunch_lengths,
                             'type': 'binomial',
                             'density_variable': 'Hamiltonian',
                             'bunch_length_fit': 'fwhm',
                             'exponent': exponents,
                             }

if args.beam_name is not None:
    beam_ID = args.beam_name

if not os.path.isdir(f'{lxdir}generated_beams/{beam_ID}/'):
    os.system(f'mkdir {lxdir}generated_beams/{beam_ID}/')

if not os.path.isfile(f'{lxdir}generated_beams/{beam_ID}/generation_settings.yaml'):
    os.system(f'touch {lxdir}generated_beams/{beam_ID}/generation_settings.yaml')

with open(f'{lxdir}generated_beams/{beam_ID}/generation_settings.yaml', 'w') as file:
    avg_int = np.mean(bunch_intensities)
    avg_bl = np.mean(bunch_lengths)
    dict_settings = {'voltage 200 MHz': args.voltage_200, 'voltage 800 MHz (fraction)': args.voltage_800,
                     'bunch intensity': float(avg_int),
                     'macroparticles per bunch': args.n_macroparticles,
                     'exponent': args.exponent, 'bunch length': float(avg_bl),
                     'beam type': args.beam_type, 'number of bunches': args.number_bunches,
                     'PS batch length': args.ps_batch_length, 'PS batch spacing': args.ps_batch_spacing,
                     'Beam ratio': beam.ratio}
    document = yaml.dump(dict_settings, file)

# If this fails, then generate without OTFB in the tracker and redefine the tracker after with OTFB.
matched_from_distribution_density_multibunch(beam, ring, SPS_tracker, distribution_options_list, N_bunches,
                                             bunch_positions, intensity_list=bunch_intensities, n_iterations_input=5,
                                             TotalInducedVoltage=total_imp)

profile.track()
if not LXPLUS:
    plt.figure()
    plt.plot(profile.bin_centers, profile.n_macroparticles)
    plt.show()


np.save(lxdir + f'generated_beams/{beam_ID}/generated_beam.npy', np.array([beam.dt, beam.dE]))
np.save(lxdir + f'generated_beams/{beam_ID}/generated_profile.npy',
        np.array([profile.bin_centers, profile.n_macroparticles * beam.ratio]))
