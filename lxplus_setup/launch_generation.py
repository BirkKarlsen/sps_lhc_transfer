'''
Launching generation of different SPS beams in lxplus.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import argparse
import os
from lxplus_setup.parsers import generation_argument_parser, generate_parsed_string
from simulation_functions.beam_generation_functions import generate_beam_ID

# Arguments -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(parents=[generation_argument_parser()],
                                 description="This file launches generation of beams in lxplus.",
                                 add_help=True)

args = parser.parse_args()

# Parameters ----------------------------------------------------------------------------------------------------------
bash_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/bash_files/'
sub_dir =  f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/submittion_files/'

if args.beam_name is None:
    beam_ID = generate_beam_ID(beam_type=args.beam_type, number_bunches=args.number_bunches,
                               ps_batch_length=args.ps_batch_length, intensity=args.intensity,
                               bunchlength=args.bunchlength, voltage_200=args.voltage_200,
                               voltage_800=args.voltage_800)
else:
    beam_ID = args.beam_name

# Setting up files to run beam generation -----------------------------------------------------------------------------
bash_file_name = 'gen_' + beam_ID + '.sh'
sub_file_name = 'gen_' + beam_ID + '.sub'
file_name = 'gen_' + beam_ID

print(f'\nGenerating bash and submission file for generation of {beam_ID}...')

# Bash file
os.system(f'touch {bash_dir}{bash_file_name}')

inputs = generate_parsed_string(args)
bash_content = f'#!/bin/bash\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'python /afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/input_files/generate_sps_beams.py ' \
               f'{inputs}'

os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_name}')
os.system(f'chmod a+x {bash_dir}{bash_file_name}')

# Submission file
os.system(f'touch {sub_dir}{sub_file_name}')

sub_content = f'executable = {bash_dir}{bash_file_name}\n' \
              f'arguments = \$(ClusterId)\$(ProcId)\n' \
              f'output = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).out\n' \
              f'error = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).err\n' \
              f'log = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).log\n' \
              f'+JobFlavour = \\"testmatch\\"\n' \
              f'queue'

os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_name}')
os.system(f'chmod a+x {sub_dir}{sub_file_name}')

os.system(f'condor_submit {sub_dir}{sub_file_name}')