'''
File to generate, simulate and injection beam from the SPS to the LHC all in one simulation.

Author: Birk Emil Karlsen-Bæck
'''

import argparse
import os

from input_files.generate_sps_beams import sps_generation
from input_files.sps_flattop import sps_simulation
from input_files.lhc_injection import lhc_injection
from lxplus_setup.parsers import (generation_argument_parser,
                                  simulation_argument_parser,
                                  lhc_llrf_argument_parser,
                                  sps_llrf_argument_parser)


def main():
    # Options ----------------------------------------------------------------------------------------------------------
    lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
    LXPLUS = True
    if 'birkkarlsen-baeck' in os.getcwd():
        lxdir = '../'
        LXPLUS = False
        print('\nRunning locally...')
    else:
        print('\nRunning in lxplus...')

    parser = argparse.ArgumentParser(parents=[generation_argument_parser(), simulation_argument_parser(),
                                              sps_llrf_argument_parser(), lhc_llrf_argument_parser()],
                                     description='Script to simulate LHC injection.', add_help=True)

    parser.add_argument('--date', '-dte', type=str,
                        help='Input date of the simulation; if none is parsed then todays date will be taken')

    args = parser.parse_args()

    beam, profile, gen_dict = sps_generation(args, LXPLUS, lxdir)
    beam, profile, gen_dict = sps_simulation(args, LXPLUS, lxdir, pre_beam=beam, generation_dict=gen_dict)
    lhc_injection(args, LXPLUS, lxdir, pre_beam=beam, generation_dict=gen_dict)


if __name__ == "__main__":
    main()
