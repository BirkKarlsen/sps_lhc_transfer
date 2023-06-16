'''
File launching parameter scans of LHC parameters.

Author: Birk Emil Karlsen-Baeck
'''

# Arguments -----------------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description='Script to launch a parameter scan defined by a yaml file.',
                                 add_help=True, prefix_chars='~')

parser.add_argument('~~scan_name', '~sn', type=str, default='LHC_capture_voltage_2.0e11.yaml',
                    help='Name of the parameter scan to turn.')

args = parser.parse_args()

# imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import itertools

from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml
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


# Launching scans -----------------------------------------------------------------------------------------------------
param_dict = fetch_from_yaml(args.scan_name, lxdir + f'parameterscans/scan_programs/')
params = list(param_dict.keys())

# Find parameters to be scanned
scans = []
reg_params = {}
for param in params:
    if type(param_dict[param]) is dict:
        scans.append(param)
    else:
        reg_params[param] = param_dict[param]

# Make parameters to scan over
n_d = len(scans)

scan_dict = {}
for param in scans:
    if param_dict[param]['scale'] == 'dB':
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

if LXPLUS:
    os.system(f'mkdir {lxdir}bash_files/{sim_folder_name}')
    os.system(f'mkdir {lxdir}submission_files/{sim_folder_name}')
    os.system(f'which python')

for arguments in itertools.product(*scan_dict.values()):
    sim_name_i = sim_folder_name + 'sim'
    sim_arg_i = ''
    for i, param in enumerate(scans):
        sim_name_i += f'_{param}{arguments[i]:.3e}'
        sim_arg_i += f'~~{param} {arguments[i]} '

    launch_string = f'python {lxdir}lxplus_setup/launch_simulation.py ' \
                    f'~ma lhc ~sm {sim_name_i} {sim_arg_i}{fixed_arguments}'

    if LXPLUS:
        os.system(launch_string)
    else:
        print(sim_arg_i)
