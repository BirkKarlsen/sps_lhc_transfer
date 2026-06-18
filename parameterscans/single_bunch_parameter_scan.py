'''
File launching parameter scans of LHC parameters.

Author: Birk Emil Karlsen-Baeck
'''


# Arguments -----------------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description='Script to launch a parameter scan defined by a yaml file.',
                                 add_help=True)

parser.add_argument(
    '--scan_name', '-sn', type=str,
    default='single_bunch_persistent_oscillations_2.3e11_2024_mini.yaml',
    help='Name of the parameter scan to turn.'
)
parser.add_argument(
    '--beam_process', '-bp', type=str, choices=['inj', 'fltbttm', 'ramp', 'flttp'],
    default='inj',
    help='Choose the part of the LHC cycle to simualte.'
)
parser.add_argument(
    '--run_gpu', '-gpu', type=int, default=0,
    help='Option to run the simulation on a GPU; default is False (0)'
)
parser.add_argument(
    '--permutations', '-pm', type=int, default=1,
    help='Option to choose to every permutation of the scanned parameters (grid scan)'
         'or to have the values correlated with each other; default is to do the permutations'
         '(True, 1)'
)
parser.add_argument(
    '--memory', '-m', type=str,
    help='Option to request more memory for the simulation; default is no request.'
)

args = parser.parse_args()


# imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import itertools
from datetime import date

from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml, make_and_write_yaml

from lxplus_setup.parsers import parse_arguments_from_dictonary
from parameterscans.param_scan_utils import (generate_sequential_scan,
                                             generate_random_scan,
                                             generate_grid_scan,
                                             generate_correlated_scan)

# Directories ---------------------------------------------------------------------------------------------------------
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
LXPLUS = True
if not 'cern.ch' in os.getcwd():
    lxdir = '../'
    LXPLUS = False
    print('\nRunning locally...')
else:
    print('\nRunning in lxplus...')

sub_dir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/submission_files/'

# Launching scans -----------------------------------------------------------------------------------------------------
param_dict = fetch_from_yaml(args.scan_name, lxdir + f'parameterscans/scan_programs/')
params = list(param_dict.keys())

# Find parameters to be scanned
scans = []
reg_params = {}
for param in params:
    if type(param_dict[param]) is dict or type(param_dict[param]) is list:
        scans.append(param)
    else:
        reg_params[param] = param_dict[param]

# Make parameters to scan over
n_d = len(scans)

scan_dict = {}
for param in scans:
    if type(param_dict[param]) is not dict:
        scan_vals = param_dict[param]
    elif 'mean' in param_dict[param]:
        scan_vals = generate_random_scan(
            param_dict[param]['mean'],
            param_dict[param]['std'],
            param_dict[param]['n_samples'],
        )
    else:
        scan_vals = generate_sequential_scan(
            param_dict[param]['start'],
            param_dict[param]['stop'],
            param_dict[param]['steps'],
            param_dict[param]['scale']
        )

    scan_dict[param] = scan_vals

# Launch scripts
fixed_arguments = parse_arguments_from_dictonary(reg_params)
sim_folder_name = args.scan_name[:-5] + '/'

today = date.today()
save_to = lxdir + f'simulation_results/{today.strftime("%Y-%m-%d")}/{sim_folder_name}'

if LXPLUS:
    os.makedirs(f'{lxdir}submission_files/{sim_folder_name}', exist_ok=True)
    os.makedirs(save_to, exist_ok=True)
    os.system(f'which python3')


if args.permutations:
    configurations = generate_grid_scan(
        scan_dict, scans, reg_params, sim_folder_name, save_to, LXPLUS
    )
else:
    configurations = generate_correlated_scan(
        scan_dict, scans, reg_params, sim_folder_name, save_to, LXPLUS
    )

if LXPLUS:
    os.system(f'touch {sub_dir}{sim_folder_name}configs.txt')

    configs_file = open(f'{sub_dir}{sim_folder_name}configs.txt', 'w')
    for config in configurations:
        configs_file.write(config + '\n')

    configs_file.close()

# Bash file
if args.beam_process == 'inj':
    script_name = 'single_bunch_injection'
elif args.beam_process == 'fltbttm':
    script_name = 'single_bunch_flatbottom'
else:
    raise RuntimeError('Ramp and flat top is not yet implemented')

# f'export EOS_MGM_URL=root://eosuser.cern.ch\n' \ at second line
# f'{stage_data}\n' \ after source .bashrc

bash_content = f'#!/bin/bash\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'which /afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 --version\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 ' \
               f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/input_files/{script_name}.py ' \
               f'--config \$1 -dte {today.strftime("%Y-%m-%d")} \n\n'

if LXPLUS:
    os.system(f'echo "{bash_content}" > {sub_dir}{sim_folder_name}execute_sim.sh')
    os.system(f'chmod a+x {sub_dir}{sim_folder_name}execute_sim.sh')
else:
    print(bash_content)

# Submission file
if LXPLUS:
    os.system(f'touch {sub_dir}{sim_folder_name}condor_submission.sub')

if bool(args.run_gpu):
    additional_string = 'request_gpus = 1\n'
else:
    additional_string = ''

if args.memory is not None:
    additional_string += f'request_memory = {args.memory}\n'

sub_content = f'executable = {sub_dir}{sim_folder_name}execute_sim.sh\n' \
              f'arguments = {save_to}\$(config)\n' \
              f'output = {sub_dir}{sim_folder_name}\$(ClusterId)\$(ProcId).out\n' \
              f'error = {sub_dir}{sim_folder_name}\$(ClusterId)\$(ProcId).err\n' \
              f'log = {sub_dir}{sim_folder_name}\$(ClusterId)\$(ProcId).log\n' \
              f'transfer_input_files = {save_to}\$(config)\n' \
              f'+JobFlavour = \\"{reg_params["flavour"]}\\"\n' \
              f'queue config from {sub_dir}{sim_folder_name}configs.txt'

sub_content = additional_string + sub_content

if LXPLUS:
    os.system(f'echo "{sub_content}" > {sub_dir}{sim_folder_name}condor_submission.sub')
    os.system(f'chmod a+x {sub_dir}{sim_folder_name}condor_submission.sub')

    os.system(f'condor_submit {sub_dir}{sim_folder_name}condor_submission.sub')