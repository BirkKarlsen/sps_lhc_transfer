'''
Launching simulations either SPS flattop or LHC injection.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import argparse
import os
from lxplus_setup.parsers import simulation_argument_parser, sps_llrf_argument_parser, lhc_llrf_argument_parser, \
    generate_parsed_string

# Arguments -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(parents=[simulation_argument_parser(),
                                          sps_llrf_argument_parser(),
                                          lhc_llrf_argument_parser()],
                                 description="This file launches simulations in lxplus.",
                                 add_help=True)

parser.add_argument('--machine', '-ma', type=str, choices=['sps', 'lhc'], default='sps',
                    help='Choose accelerator to simulate in; default is the SPS.')
parser.add_argument('--simulation_name', '-sm', type=str, default='test',
                    help='Name of the simulation to be launched.')

args = parser.parse_args()

disable = False

# Parameters ----------------------------------------------------------------------------------------------------------
bash_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/bash_files/'
sub_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/submission_files/'

beam_ID = args.beam_name

# Setting up files to run beam generation -----------------------------------------------------------------------------
if args.machine == 'sps':
    bash_file_name = 'sim_' + beam_ID + '.sh'
    sub_file_name = 'sim_' + beam_ID + '.sub'
    file_name = 'sim_' + beam_ID
else:
    bash_file_name = args.simulation_name + '.sh'
    sub_file_name = args.simulation_name + '.sub'
    file_name = args.simulation_name

print(f'\nGenerating bash and submission file for {file_name}...')

# Bash file
if not disable:
    os.system(f'touch {bash_dir}{bash_file_name}')

inputs = generate_parsed_string(args, sim=True, machine=args.machine)

if args.machine == 'sps':
    script_name = 'sps_flattop'
else:
    script_name = 'lhc_injection'


bash_content = f'#!/bin/bash\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'python /afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/input_files/{script_name}.py ' \
               f'{inputs}'


if not disable:
    os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_name}')
    os.system(f'chmod a+x {bash_dir}{bash_file_name}')

# Submission file
if not disable:
    os.system(f'touch {sub_dir}{sub_file_name}')

sub_content = f'executable = {bash_dir}{bash_file_name}\n' \
              f'arguments = \$(ClusterId)\$(ProcId)\n' \
              f'output = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).out\n' \
              f'error = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).err\n' \
              f'log = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).log\n' \
              f'+JobFlavour = \\"testmatch\\"\n' \
              f'queue'

if not disable:
    os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_name}')
    os.system(f'chmod a+x {sub_dir}{sub_file_name}')

    os.system(f'condor_submit {sub_dir}{sub_file_name}')