'''
Launching generation of different SPS beams in lxplus.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import argparse
import os
from lxplus_setup.parsers import generation_argument_parser

# Arguments -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(parents=[generation_argument_parser()],
                                 description="This file launches generation of beams in lxplus.",
                                 add_help=True)

args = parser.parse_args()