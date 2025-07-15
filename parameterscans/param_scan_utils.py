"""Useful functions for parameter scans

Author: Birk Emil Karlsen-Bæck
"""

import numpy as np
import numpy.random as npr
import itertools
import os

from beam_dynamics_tools.analytical_functions.mathematical_functions import to_linear
from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml, make_and_write_yaml


def generate_sequential_scan(start: float, stop: float, steps: int, scale: str = 'linear'):
    '''Generate a sequence of parameters to be scanned.

    Args:
        start (float):
            Start point of sequence.
        stop (float):
            End point of sequence.
        steps (int):
            Number of steps in sequence.
        scale (str):
            Option to do either a linear or logarithmic scan.

    Returns:
        scan_vals (NDArray):
            Numpy-array with values to be scanned.
    '''
    scan_vals = np.linspace(float(start), float(stop), int(steps))

    if scale == 'dB':
        scan_vals = to_linear(scan_vals)

    return scan_vals


def generate_random_scan(mean: float, std: float, n_samples: int):
    '''Generate a sequence of parameters to be scanned.

    Args:
        mean (float):
            Mean of random distribution.
        std (float):
            Standard deviation of random distribution.
        n_samples (int):
            The number of samples to draw from the distribution

    Returns:
        Numpy-aarray with values to be scanned.
    '''

    return npr.normal(mean, std, n_samples)


def generate_grid_scan(
        scan_dict: dict,
        scans: list,
        reg_params: dict,
        simulation_folder_name: str,
        save_to: str,
        lxplus: bool = True
    ) -> list[str]:
    '''Generate scan with every permutation of the parameters to be scanned.

    Args:
        scan_dict (dict):
            The dictionary of all parameters to be scanned and how.
        scans (list):
            List of parameters to be scanned.
        reg_params (dict):
            Dictionary of parameters which are not to be scanned over.
        simulation_folder_name (str):
            The name of the folder to save all results of the grid scan to.
        save_to (str):
            The over all directory to save the files to.
        lxplus (bool):
            Flag to see whether the function is called locally or on lxplus.

    Returns:
        configurations (list):
            List of all the parameter configurations to be simulated.
    '''

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
            sim_arg_i += f'--{param} {arguments[i]} '

            try:
                config_i[param] = arguments[i].item()
            except:
                config_i[param] = arguments[i]

        config_i['simulation_name'] = simulation_folder_name + sim_name_i
        config_i = config_i | reg_params
        config_i.pop('flavour', None)
        configurations.append(sim_name_i + '/config.yaml')

        if lxplus:
            os.makedirs(save_to + sim_name_i, exist_ok=True)
            make_and_write_yaml('config.yaml', save_to + sim_name_i + '/', config_i)
        else:
            print(sim_arg_i)
            print(config_i['simulation_name'])

    return configurations


def generate_correlated_scan(
        scan_dict: dict,
        scans: list,
        reg_params: dict,
        simulation_folder_name: str,
        save_to: str,
        lxplus: bool = True
    ) -> list[str]:

    n_sims = len(scan_dict[list(scan_dict.keys())[0]])

    configurations = []
    for i in range(n_sims):
        sim_name_i = 'sim'
        sim_arg_i = ''
        config_i = {}

        for j, param in enumerate(scans):
            if type(scan_dict[param][i]) is str:
                sim_name_i += f'_{param}{scan_dict[param][i]}'
            else:
                sim_name_i += f'_{param}{scan_dict[param][i]:.3e}'
            sim_arg_i += f'--{param} {scan_dict[param][i]} '

            try:
                config_i[param] = scan_dict[param][i].item()
            except:
                config_i[param] = scan_dict[param][i]

        config_i['simulation_name'] = simulation_folder_name + sim_name_i
        config_i = config_i | reg_params
        config_i.pop('flavour', None)
        configurations.append(sim_name_i + '/config.yaml')

        if lxplus:
            os.makedirs(save_to + sim_name_i, exist_ok=True)
            make_and_write_yaml('config.yaml', save_to + sim_name_i + '/', config_i)
        else:
            print(sim_arg_i)
            print(config_i['simulation_name'])

    return configurations