'''
Argument parser for simulation scripts and simulation setups.

Author: Birk Emil Karlsen-Baeck
'''

import argparse


def generation_argument_parser(add_help=False):
    r'''
    Parser for generation of beams in the SPS.

    :return: parser
    '''

    parser = argparse.ArgumentParser(description='Script to generated beams in the SPS with intensity effects.',
                                     add_help=add_help, prefix_chars='~')

    parser.add_argument('~~voltage_200', '~v1', type=float, default=7.5,
                        help='Voltage of the 200 MHz RF system [MV]; default is 7.5')
    parser.add_argument('~~voltage_800', '~v2', type=float, default=0.15,
                        help='Voltage of the 800 MHz RF system in fract of 200 MHz voltage; default is 0.15')
    parser.add_argument('~~intensity', '~in', type=float, default=1.4,
                        help='Average intensity per bunch in units of 1e11; default is 1.4')
    parser.add_argument('~~n_macroparticles', '~nm', type=float, default=1000000,
                        help='Number of macroparticles per bunch; default is 1 million.')
    parser.add_argument('~~exponent', '~ex', type=float, default=1.5,
                        help='Binomial exponent for bunches; if passed all bunches have the same exponent; default is 1.5')
    parser.add_argument('~~bunchlength', '~bl', type=float, default=1.6,
                        help='Bunch length FWHM for the bunches; if passed all bunches have the same bunch length; default'
                             'is 1.6 ns')
    parser.add_argument('~~beam_type', '~bt', type=str, default='BCMS',
                        help='Beam type to be generated; default is BCMS.')
    parser.add_argument('~~number_bunches', '~nb', type=int, default=72,
                        help='Number of bunches in the beam; default is 36. If the beam type is 8b4e then it has to be a '
                             'multiple of 8')
    parser.add_argument('~~profile_length', '~pl', type=int, default=800,
                        help='Length of profile object in units of RF buckets; default is 800')
    parser.add_argument('~~ps_batch_length', '~psbl', type=int, default=36,
                        help='The length of the batches delivered from the PS to the SPS; default is 36 bunches')
    parser.add_argument('~~ps_batch_spacing', '~psbs', type=int, default=45,
                        help='The spacing between PS batches in units of RF buckets; default is 45 buckets')

    parser.add_argument('~~beam_name', '~bn', type=str,
                        help='Option to give custom name to the beam; default is a name specified by the bunch parameters.')
    parser.add_argument('~~custom_beam', '~cb', type=int, default=0,
                        help='Option to have custom distribution of bunch parameters from measurements.')
    parser.add_argument('~~custom_beam_dir', '~cbd', type=str,
                        help='Directory of the custom beam parameters.')

    return parser


def simulation_argument_parser(add_help=False):
    r'''
    Arguments for simulations.

    :param add_help:
    :return:
    '''
    parser = argparse.ArgumentParser(description='Arguments for simulations.', add_help=add_help,
                                     prefix_chars='~')

    # General inputs
    parser.add_argument('~~simulation_name', '~sm', type=str, default='test',
                        help='Name of the simulation to be launched.')

    # Parsers for beam
    parser.add_argument('~~beam_name', '~bn', type=str, default='LHC_25ns_1.6e11_mu1.5',
                        help='Option to give custom name to the beam; default is a name specified by the bunch '
                             'parameters.')
    parser.add_argument('~~profile_length', '~pl', type=int, default=1000,
                        help='Length of profile object in units of RF buckets; default is 800')

    # Parsers for simulation
    parser.add_argument('~~number_of_turns', '~nt', type=int, default=2000,
                        help='Number of turns to track; default is 2000 turns')
    parser.add_argument('~~diag_setting', '~ds', type=int, default=0, choices=[0, 1, 2],
                        help='Different simulation diagnostics settings; default is 0')
    parser.add_argument('~~dt_cont', '~dct', type=int, default=1,
                        help='The turns between the continuous signals are sampled; default is every turn')
    parser.add_argument('~~dt_beam', '~dbm', type=int, default=1000,
                        help='The turns between beam parameters are measured; default is every 1000 turns')
    parser.add_argument('~~dt_cl', '~dcl', type=int, default=1000,
                        help='The turns between cavity controller signals are measured; default is every 1000 turns')
    parser.add_argument('~~dt_prfl', '~dpr', type=int, default=500,
                        help='The turns between repositioning the profile cuts; default is every 500 turns')
    parser.add_argument('~~dt_ld', '~dld', type=int, default=100,
                        help='The turns between measuring the beam line density; default is every 100 turns')

    return parser


def sps_llrf_argument_parser(add_help=False):
    r'''
    Argument parser for the SPS cavity controller model.

    :param add_help: Option to pass the help to parent parser.
    :return: parser
    '''

    parser = argparse.ArgumentParser(description='Arguments for the SPS cavity controller.', add_help=add_help,
                                     prefix_chars='~')

    # Parsers for SPS cavity controller
    parser.add_argument('~~g_ff_1', '~gf1', type=float, default=1,
                        help='FF gain for 3-section cavities; default is 1')
    parser.add_argument('~~g_llrf_1', '~gl1', type=float, default=20,
                        help='LLRF gain for 3-section cavities; default is 20')
    parser.add_argument('~~g_tx_1', '~gt1', type=float, default=1,
                        help='Transmitter gain for 3-section cavities; default is 1')
    parser.add_argument('~~a_comb', '~ac', type=float, default=63 / 64,
                        help='Comb filter coefficient; default is 63/64')
    parser.add_argument('~~v_part', '~vp', type=float, default=0.6,
                        help='Voltage partitioning between 3- and 4-section cavities; default is 0.6')
    parser.add_argument('~~g_ff_2', '~gf2', type=float,
                        help='FF gain for 4-section cavities; default is same as 3-section')
    parser.add_argument('~~g_llrf_2', '~gl2', type=float,
                        help='LLRF gain for 4-section cavities; default is same as 3-section')
    parser.add_argument('~~g_tx_2', '~gt2', type=float,
                        help='Transmitter gain for 4-section cavities; default is same as 3-section')
    parser.add_argument('~~open_ff', '~ff', type=int, default=1,
                        help='Open the SPS FF; default is 1 (True)')

    return parser


def lhc_llrf_argument_parser(add_help=False):
    r'''
    Argument parser for the LHC cavity controller.

    :param add_help: Option to pass on help
    :return: parser
    '''

    parser = argparse.ArgumentParser(description='Arguments for the LHC cavity controller.', add_help=add_help,
                                     prefix_chars='~')

    # Parsers for the LHC cavity loop
    parser.add_argument('~~analog_gain', '~ga', type=float, default=6.79e-6,
                        help='Analog gain in the LHC RFFB; default is 6.79e-6 A/V')
    parser.add_argument('~~analog_delay', '~ta', type=float, default=170e-6,
                        help='Analog feedback delay in the LHC RFFB; default is 170e-6 s')
    parser.add_argument('~~digital_gain', '~gd', type=float, default=10,
                        help='Digital gain in the LHC RFFB; default is 10')
    parser.add_argument('~~digital_delay', '~td', type=float, default=400e-6,
                        help='Digital feedback delay in the LHC RFFB; default is 400e-6 s')
    parser.add_argument('~~loop_delay', '~tl', type=float, default=650e-9,
                        help='Total loop delay in the LHC RFFB; default is 650e-9 s')
    parser.add_argument('~~loaded_q', '~ql', type=float, default=20000,
                        help='Loaded quality in the LHC cavity; default is 20000')
    parser.add_argument('~~comb_alpha', '~ca', type=float, default=15 / 16,
                        help='Comb filter coefficient for the LHC OTFB; default is 15/16')
    parser.add_argument('~~otfb_delay', '~to', type=float, default=1.2e-6,
                        help='Complementary delay in the LHC OTFB; default is  1.2e-6 s')
    parser.add_argument('~~otfb_gain', '~go', type=float, default=10,
                        help='The OTFB gain; default is 10')
    parser.add_argument('~~detuning_mu', '~dl', type=float, default=0,
                        help='The tuning parameter which determines the number of turns it takes to detune the cavity; '
                             'default is 0.')
    parser.add_argument('~~delta_frequency', '~df', type=float, default=0,
                        help='The detuning at the start of the simulation; default is 0')
    parser.add_argument('~~pre_detune', '~pd', type=int, default=0,
                        help='Option to enable pre-detuning of the RF cavities; default is False (0)')
    parser.add_argument('~~clamping_thres', '~ct', type=float, default=300e3,
                        help='The available power in klystron [W]; default is 300 kW')
    parser.add_argument('~~clamp', '~cp', type=int, default=0,
                        help='Option to enable power limitation; default is disabled (0)')

    # Parsers for the global feedback
    parser.add_argument('~~include_global', '~igl', type=int, default=0,
                        help='Option to include phase and synchro loop in the simulations.')
    parser.add_argument('~~pl_gain', '~plg', type=float,
                        help='The beam-phase loop gain; default is 1/(5 T_rev).')
    parser.add_argument('~~sl_gain', '~slg', type=float,
                        help='The synchro loop gain; default is PL_gain/10.')

    # Parsers for the LHC globally
    parser.add_argument('~~voltage', '~vo', type=float, default=4,
                        help='Voltage of the 400 MHz RF system [MV]; default is 4')
    parser.add_argument('~~gamma_t', '~gt', type=float, default=53.606713,
                        help='Transition gamma of the LHC; default is 53.606713')
    parser.add_argument('~~include_impedance', '~imp', type=int, default=1,
                        help='Option to include LHC impedance model; default is to include it (1)')
    parser.add_argument('~~simulated_beam', '~sb', type=int, default=0,
                        help='Input a beam simulated at SPS flattop or a beam directly from generation; '
                             'default is from generation')
    parser.add_argument('~~energy_error', '~eer', type=float, default=0,
                        help='Option to add energy offset [MeV] to initial batch; default is 0 MeV')
    parser.add_argument('~~phase_error', '~per', type=float, default=0,
                        help='Option to add phase offset [degrees] to initial batch; default is 0')
    parser.add_argument('~~impedance_model', '~im', type=str,
                        default='LHC_450GeV.dat',
                        help='Option to choose which impedance model to use; default is standard LHC injection.')
    parser.add_argument('~~ramp', '~r', type=float,
                        help='Option to include a different final energy [GeV] and this a ramp in simulation; '
                             'if nothing is passed then there will be no ramp.')
    parser.add_argument('~~momentum_program', '~mp', type=str, default='LHC_momentum_programme_6.8TeV.csv',
                        help='Option to choose the momentum program for the LHC default is the 6.8 TeV ramp')
    parser.add_argument('~~scheme', '~shm', type=str, default='single_injection.yaml',
                        help='Option to input injection scheme into the simulation, default is no injections.')

    return parser


def generate_parsed_string(args, sim=False, machine='sps', n_master_parsers=2):
    arg_dict = vars(args)
    arg_str = ''
    arguments = list(arg_dict.keys())
    n_args = len(arguments)

    # Number of SPS arguments
    sps_llrf = sps_llrf_argument_parser()
    sps_args = vars(sps_llrf.parse_known_args()[0])
    n_sps = len(list(sps_args.keys()))

    # Number of LHC arguments
    lhc_llrf = lhc_llrf_argument_parser()
    lhc_args = vars(lhc_llrf.parse_known_args()[0])
    n_lhc = len(list(lhc_args.keys()))

    for i in range(n_args):
        if not arg_dict[arguments[i]] is None:
            if not sim:
                arg_str += f'~~{arguments[i]} {arg_dict[arguments[i]]} '
            else:
                if machine == 'sps' and i < n_args - n_lhc - n_master_parsers:
                    arg_str += f'~~{arguments[i]} {arg_dict[arguments[i]]} '

                if machine == 'lhc' and (i < n_args - n_lhc - n_sps - n_master_parsers or
                                         i >= n_args - n_lhc - n_master_parsers and i < n_args - n_master_parsers):
                    arg_str += f'~~{arguments[i]} {arg_dict[arguments[i]]} '

                if machine == 'both' and i < n_args - n_master_parsers:
                    arg_str += f'~~{arguments[i]} {arg_dict[arguments[i]]} '

    return arg_str


def parse_arguments_from_dictonary(input_dict: dict):
    parse_string = ''

    for i, (argmnt, arg_val) in enumerate(input_dict.items()):
        parse_string += f'~~{argmnt} {arg_val} '

    return parse_string