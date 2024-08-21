'''
Launching simulations for both generation, SPS flattop and LHC injection.

Author: Birk Emil Karlsen-Baeck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import argparse
import os
from lxplus_setup.parsers import (generation_argument_parser, simulation_argument_parser,
                                  sps_llrf_argument_parser, lhc_llrf_argument_parser,
                                  generate_parsed_string)
from lxplus_setup.staging_simulations import stage_out_simulation_results
from datetime import date

# Arguments -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(parents=[generation_argument_parser(),
                                          simulation_argument_parser(),
                                          sps_llrf_argument_parser(),
                                          lhc_llrf_argument_parser()],
                                 description="This file launches simulations in lxplus.",
                                 add_help=True)

job_flavours = ['espresso',         # 20 minutes
                'microcentury',     # 1 hour
                'longlunch',        # 2 hours
                'workday',          # 8 hours
                'tomorrow',         # 1 day
                'testmatch',        # 3 days
                'nextweek']         # 1 week
parser.add_argument('--flavour', '-f', type=str, choices=job_flavours, default='testmatch',
                    help='Length of allocated for the simulaton; default is testmatch (3 days)')

args = parser.parse_args()

disable = False

# Parameters ----------------------------------------------------------------------------------------------------------
bash_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/bash_files/'
sub_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/submission_files/'
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
today = date.today()
save_to = lxdir + f'simulation_results/{today.strftime("%b-%d-%Y")}/{args.simulation_name}/'

beam_ID = args.beam_name

# Setting up files to run beam generation -----------------------------------------------------------------------------
bash_file_name = args.simulation_name + '.sh'
sub_file_name = args.simulation_name + '.sub'
file_name = args.simulation_name

print(f'\nGenerating bash and submission file for {file_name}...')


# Bash file
if not disable:
    os.system(f'touch {bash_dir}{bash_file_name}')

inputs = generate_parsed_string(args, sim=True, machine='both', n_master_parsers=1)

script_name = 'transfer_complete'
stage_data_out = stage_out_simulation_results(save_to, 'bkarlsen', args.simulation_name)

bash_content = f'#!/bin/bash\n' \
               f'export EOS_MGM_URL=root://eosuser.cern.ch\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'which /afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 --version\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 ' \
               f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/input_files/{script_name}.py ' \
               f'{inputs} -dte {today.strftime("%b-%d-%Y")} \n\n' \
               f'{stage_data_out}'

if not disable:
    os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_name}')
    os.system(f'chmod a+x {bash_dir}{bash_file_name}')
else:
    print(bash_content)

# Submission file
if not disable:
    os.system(f'touch {sub_dir}{sub_file_name}')

sub_content = f'executable = {bash_dir}{bash_file_name}\n' \
              f'arguments = \$(ClusterId)\$(ProcId)\n' \
              f'output = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).out\n' \
              f'error = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).err\n' \
              f'log = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).log\n' \
              f'+JobFlavour = \\"{args.flavour}\\"\n' \
              f'queue'

if not disable:
    os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_name}')
    os.system(f'chmod a+x {sub_dir}{sub_file_name}')

    os.system(f'condor_submit {sub_dir}{sub_file_name}')