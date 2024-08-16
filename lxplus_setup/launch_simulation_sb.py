'''
Launching simulations either SPS flattop or LHC injection.

Author: Birk Emil Karlsen-Baeck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import argparse
import os
from lxplus_setup.parsers import single_bunch_simulation_parser, generate_parsed_string
from lxplus_setup.staging_simulations import stage_data_for_simulation, stage_out_simulation_results
from datetime import date

# Arguments -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(parents=[single_bunch_simulation_parser()],
                                 description="This file launches simulations in lxplus.",
                                 add_help=True, prefix_chars='~')

job_flavours = ['espresso',         # 20 minutes
                'microcentury',     # 1 hour
                'longlunch',        # 2 hours
                'workday',          # 8 hours
                'tomorrow',         # 1 day
                'testmatch',        # 3 days
                'nextweek']         # 1 week
parser.add_argument('~~flavour', '~f', type=str, choices=job_flavours, default='testmatch',
                    help='Length of allocated for the simulaton; default is testmatch (3 days)')
parser.add_argument('~~as_gpu', '~agpu', type=int, default=0,
                    help='Option to run the simulation on a GPU; default is False (0)')


args = parser.parse_args()

disable = False

# Parameters ----------------------------------------------------------------------------------------------------------
bash_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/bash_files/'
sub_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/submission_files/'
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
today = date.today()
save_to = lxdir + f'simulation_results/{today.strftime("%b-%d-%Y")}/{args.simulation_name}/'

# Setting up files to run beam generation -----------------------------------------------------------------------------
bash_file_name = args.simulation_name + '.sh'
sub_file_name = args.simulation_name + '.sub'
file_name = args.simulation_name

print(f'\nGenerating bash and submission file for {file_name}...')

# Bash file
if not disable:
    os.system(f'touch {bash_dir}{bash_file_name}')

inputs = generate_parsed_string(args, sim=True, machine='both', n_master_parsers=2)

if disable:
    print(inputs)

script_name = 'single_bunch_injection'
stage_data_out = stage_out_simulation_results(save_to, 'bkarlsen', args.simulation_name)

# f'export EOS_MGM_URL=root://eosuser.cern.ch\n' \ at second line
# f'{stage_data}\n' \ after source .bashrc

bash_content = f'#!/bin/bash\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'which /afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 --version\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 ' \
               f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/input_files/{script_name}.py ' \
               f'{inputs} ~dte {today.strftime("%b-%d-%Y")} \n\n'

if not disable:
    os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_name}')
    os.system(f'chmod a+x {bash_dir}{bash_file_name}')
else:
    print(bash_content)

# Submission file
if not disable:
    os.system(f'touch {sub_dir}{sub_file_name}')

if bool(args.run_gpu):
    additional_string = 'request_gpus = 1\n'
else:
    additional_string = ''

sub_content = f'executable = {bash_dir}{bash_file_name}\n' \
              f'arguments = \$(ClusterId)\$(ProcId)\n' \
              f'output = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).out\n' \
              f'error = {bash_dir}{file_name}.\$(ClusterId)\$(ProcId).err\n' \
              f'log = {bash_dir}{file_name}.\$(ClusterId).log\n' \
              f'+JobFlavour = \\"{args.flavour}\\"\n' \
              f'queue'

sub_content = additional_string + sub_content

if not disable:
    os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_name}')
    os.system(f'chmod a+x {sub_dir}{sub_file_name}')

    os.system(f'condor_submit {sub_dir}{sub_file_name}')