'''
File launching parameter scans of LHC parameters.

Author: Birk Emil Karlsen-Baeck
'''


# Arguments -----------------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description='Script to launch a parameter scan defined by a yaml file.',
                                 add_help=True, prefix_chars='~')

parser.add_argument('~~scan_name', '~sn', type=str,
                    default='single_bunch_persistent_oscillations_2.3e11_2024_mini.yaml',
                    help='Name of the parameter scan to turn.')
parser.add_argument('~~run_gpu', '~gpu', type=int, default=0,
                    help='Option to run the simulation on a GPU; default is False (0)')

args = parser.parse_args()


# imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import itertools
from datetime import date

from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml, make_and_write_yaml
from beam_dynamics_tools.analytical_functions.mathematical_functions import to_linear

from lxplus_setup.parsers import parse_arguments_from_dictonary

# Directories ---------------------------------------------------------------------------------------------------------
lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
LXPLUS = True
if 'birkkarlsen-baeck' in os.getcwd():
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
    elif param_dict[param]['scale'] == 'dB':
        scan_vals_dB = np.linspace(param_dict[param]['start'], param_dict[param]['stop'],
                                   param_dict[param]['steps'])
        scan_vals = to_linear(scan_vals_dB)
    else:
        scan_vals = np.linspace(float(param_dict[param]['start']), float(param_dict[param]['stop']),
                                param_dict[param]['steps'])

    scan_dict[param] = scan_vals

# Launch scripts
fixed_arguments = parse_arguments_from_dictonary(reg_params)
sim_folder_name = args.scan_name[:-5] + '/'

today = date.today()
save_to = lxdir + f'simulation_results/{today.strftime("%b-%d-%Y")}/{sim_folder_name}'

if LXPLUS:
    os.makedirs(f'{lxdir}submission_files/{sim_folder_name}', exist_ok=True)
    os.makedirs(save_to, exist_ok=True)
    os.system(f'which python3')

configurations = []
for arguments in itertools.product(*scan_dict.values()):
    sim_name_i = 'sim'
    sim_arg_i = ''
    config_i = {}

    for i, param in enumerate(scans):
        if type(arguments[i]) is str:
            sim_name_i += f'_{param}{arguments[i]}'
        else:
            sim_name_i += f'_{param}{arguments[i]:.3e}'
        sim_arg_i += f'~~{param} {arguments[i]} '

        try:
            config_i[param] = arguments[i].item()
        except:
            config_i[param] = arguments[i]

    config_i['simulation_name'] = sim_name_i
    config_i = config_i | reg_params
    configurations.append(sim_name_i + '/config.yaml')

    if LXPLUS:
        os.makedirs(save_to + sim_name_i, exist_ok=True)
        make_and_write_yaml('config.yaml', save_to + sim_name_i + '/', config_i)
    else:
        print(sim_arg_i)
        print(save_to)
        print(sim_name_i)

if LXPLUS:
    os.system(f'touch {sub_dir}{sim_folder_name}configs.txt')

    configs_file = open(f'{sub_dir}{sim_folder_name}configs.txt', 'w')
    for config in configurations:
        configs_file.write(config + '\n')

    configs_file.close()

# Bash file
script_name = 'single_bunch_injection'

# f'export EOS_MGM_URL=root://eosuser.cern.ch\n' \ at second line
# f'{stage_data}\n' \ after source .bashrc

bash_content = f'#!/bin/bash\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'which /afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 --version\n' \
               f'/afs/cern.ch/user/b/bkarlsen/pythonpackages/p3.11.8/bin/python3 ' \
               f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/input_files/{script_name}.py ' \
               f'~~cfg \$1 ~dte {today.strftime("%b-%d-%Y")} \n\n'

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

sub_content = f'executable = {sub_dir}{sim_folder_name}execute_sim.sh\n' \
              f'arguments = {save_to}\$(config)\n' \
              f'output = {sub_dir}{sim_folder_name}\$(config).out\n' \
              f'error = {sub_dir}{sim_folder_name}\$(config).err\n' \
              f'log = {sub_dir}{sim_folder_name}\$(config).log\n' \
              f'transfer_input_files = {save_to}\$(config)\n' \
              f'+JobFlavour = \\"{reg_params["flavour"]}\\"\n' \
              f'queue config from {sub_dir}{sim_folder_name}configs.txt'

sub_content = additional_string + sub_content

if LXPLUS:
    os.system(f'echo "{sub_content}" > {sub_dir}{sim_folder_name}condor_submission.sub')
    os.system(f'chmod a+x {sub_dir}{sim_folder_name}condor_submission.sub')

    os.system(f'condor_submit {sub_dir}{sim_folder_name}condor_submission.sub')