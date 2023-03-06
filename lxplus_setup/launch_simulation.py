'''
Launching simulations either SPS flattop or LHC injection.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import argparse
import os
from lxplus_setup.parsers import simulation_argument_parser, sps_llrf_argument_parser, lhc_llrf_argument_parser

# Arguments -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(parents=[simulation_argument_parser(),
                                          sps_llrf_argument_parser(),
                                          lhc_llrf_argument_parser()],
                                 description="This file launches simulations in lxplus.",
                                 add_help=True)

args = parser.parse_args()