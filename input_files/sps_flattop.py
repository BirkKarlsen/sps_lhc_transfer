'''
Simulation of SPS flattop with beams generated in the generate_sps_beams.py script.

Author: Birk Emil Karlsen-Baeck
'''

# Imports -------------------------------------------------------------------------------------------------------------
print('\nImporting...')
import numpy as np
import yaml
import os
from datetime import date

from beam_dynamics_tools.simulation_functions.diagnostics.sps_diagnostics import SPSDiagnostics

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.cavity_feedback import SPSCavityLoopCommissioning, SPSCavityFeedback
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageFreq
from blond.impedances.impedance_sources import InputTable

from SPS.impedance_scenario import scenario, impedance2blond


def sps_simulation(args, LXPLUS, lxdir, pre_beam=None, generation_dict=None):

    beam_ID = args.beam_name

    # Import Settings from Generation ----------------------------------------------------------------------------------
    if generation_dict is None:
        with open(f'{lxdir}generated_beams/{beam_ID}/generation_settings.yaml') as file:
            gen_dict = yaml.full_load(file)
    else:
        gen_dict = generation_dict

    # Parameters -------------------------------------------------------------------------------------------------------
    # Accelerator parameters
    C = 2 * np.pi * 1100.009                                # Machine circumference [m]
    p_s = 450e9                                             # Synchronous momentum [eV/c]
    h = 4620                                                # Harmonic number [-]
    gamma_t = 18.0                                          # Transition gamma [-]
    alpha = 1./gamma_t/gamma_t                              # First order mom. comp. factor [-]
    V = gen_dict['voltage 200 MHz'] * 1e6                   # 200 MHz RF voltage [V]
    V_800 = gen_dict['voltage 800 MHz (fraction)'] * V      # 800 MHz RF voltage [V]
    dphi = 0                                                # 200 MHz Phase modulation/offset [rad]
    dphi_800 = np.pi                                        # 800 MHz Phase modulation/offset [rad]

    # Cavity Controller parameters
    if args.g_ff_2 is None:
        G_ff = [args.g_ff_1, args.g_ff_1]
    else:
        G_ff = [args.g_ff_1, args.g_ff_2]

    if args.g_llrf_2 is None:
        G_llrf = [args.g_llrf_1, args.g_llrf_1]
    else:
        G_llrf = [args.g_llrf_1, args.g_llrf_2]

    if args.g_tx_2 is None:
        G_tx = [args.g_tx_1, args.g_tx_1]
    else:
        G_tx = [args.g_tx_1, args.g_tx_2]

    # Beam parameters
    N_p = gen_dict['bunch intensity']                       # Average bunch intensity [p/b]
    N_bunches = gen_dict['number of bunches']               # Total number of bunches

    # Simulation parameters
    N_t = args.number_of_turns                              # Number of turns
    N_m = gen_dict['macroparticles per bunch']              # Number of macroparticles
    N_m *= N_bunches
    N_p *= N_bunches
    N_buckets = args.profile_length

    # Parameters for the SPS Impedance Model
    freqRes = 43.3e3                                        # Frequency resolution [Hz]
    modelStr = "futurePostLS2_SPS_noMain200TWC.txt"         # Name of Impedance Model

    # Objects ----------------------------------------------------------------------------------------------------------
    print('\nInitializing Objects...')

    # SPS Ring
    ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)

    # RF Station
    rfstation = RFStation(ring, [h, 4 * h], [V, V_800], [dphi, dphi_800], n_rf=2)

    # Beam
    ddt = 0 * rfstation.t_rf[0, 0]
    beam = Beam(ring, N_m, N_p)
    if pre_beam is None:
        gen_beam = np.load(f'{lxdir}generated_beams/{beam_ID}/generated_beam.npy')
    else:
        gen_beam = pre_beam

    beam.dE = gen_beam[1, :]
    beam.dt = gen_beam[0, :] + ddt

    # Profile
    profile = Profile(beam, CutOptions=CutOptions(cut_left=rfstation.t_rf[0, 0] * (-0.5) + ddt,
                      cut_right=rfstation.t_rf[0, 0] * (N_buckets + 0.5) + ddt,
                      n_slices=int(round(2 ** 7 * (1 + N_buckets)))))

    # Modify cuts of the Beam Profile
    beam.statistics()
    profile.cut_options.track_cuts(beam)
    profile.set_slices_parameters()
    profile.track()

    # SPS Impedance Model
    impScenario = scenario(modelStr)
    impModel = impedance2blond(impScenario.table_impedance)

    impFreq = InducedVoltageFreq(beam, profile, impModel.impedanceList, freqRes)
    SPSimpedance_table = InputTable(impFreq.freq,
                                    impFreq.total_impedance.real*profile.bin_size,
                                    impFreq.total_impedance.imag*profile.bin_size)
    impedance_freq = InducedVoltageFreq(beam, profile, [SPSimpedance_table], frequency_resolution=freqRes)
    total_imp = TotalInducedVoltage(beam, profile, [impedance_freq])

    # SPS Cavity Controller
    Commissioning = SPSCavityLoopCommissioning(debug=False, open_loop=False, open_FB=False, open_drive=False,
                                               open_FF=bool(args.open_ff), cpp_conv=False, pwr_clamp=False)
    CF = SPSCavityFeedback(rfstation, profile, Commissioning=Commissioning, post_LS2=True,
                           G_ff=G_ff, G_llrf=G_llrf, G_tx=G_tx, a_comb=args.a_comb,
                           V_part=args.v_part, turns=1000, df=0)

    # Tracker Object without SPS OTFB
    SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                      CavityFeedback=CF, Profile=profile, interpolation=True)
    SPS_tracker = FullRingAndRF([SPS_rf_tracker])

    # Simulating -------------------------------------------------------------------------------------------------------
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

    # Setting diagnostics function
    diagnostics = SPSDiagnostics(SPS_rf_tracker, profile, total_imp, CF, ring, save_to, lxdir, N_bunches,
                                 setting=args.diag_setting, dt_cont=args.dt_cont,
                                 dt_beam=args.dt_beam, dt_cl=args.dt_cl, dt_prfl=args.dt_prfl, dt_ld=args.dt_ld)

    # Main for loop
    for i in range(N_t):
        SPS_tracker.track()
        profile.track()
        total_imp.induced_voltage_sum()

        diagnostics.track()

        if i == 0:
            print('\nFor-loop successfully entered')

    with open(f'{lxdir}generated_beams/{beam_ID}/generation_settings.yaml', 'w') as file:
        gen_dict['Turns simulated'] = N_t
        document = yaml.dump(gen_dict, file)

    return beam, profile, gen_dict


def main():
    # Parse Arguments --------------------------------------------------------------------------------------------------
    import argparse
    from lxplus_setup.parsers import simulation_argument_parser, sps_llrf_argument_parser

    parser = argparse.ArgumentParser(parents=[simulation_argument_parser(), sps_llrf_argument_parser()],
                                     description='Script to simulate beams in the '
                                                 'SPS with intensity effects at flattop.',
                                     add_help=True, prefix_chars='~')

    parser.add_argument('~~date', '~dte', type=str,
                        help='Input date of the simulation; if none is parsed then todays date will be taken')

    args = parser.parse_args()

    # Options ----------------------------------------------------------------------------------------------------------
    lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
    LXPLUS = True
    if 'birkkarlsen-baeck' in os.getcwd():
        lxdir = '../'
        LXPLUS = False
        print('\nRunning locally...')
    else:
        print('\nRunning in lxplus...')

    beam, profile, gen_dict = sps_simulation(args, LXPLUS, lxdir)

    beam_ID = args.beam_name

    np.save(lxdir + f'generated_beams/{beam_ID}/simulated_beam.npy', np.array([beam.dt, beam.dE]))
    np.save(lxdir + f'generated_beams/{beam_ID}/simulated_profile.npy',
            np.array([profile.bin_centers, profile.n_macroparticles * beam.ratio]))
    print('\nBeam Extracted!')


if __name__ == "__main__":
    main()
