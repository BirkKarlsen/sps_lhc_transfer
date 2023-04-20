'''
Functions to load the correct data for lxplus simulations.

Author: Birk Emil Karlsen-Bæck
'''

import os
from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml


def convert_afs_to_eos(afs_dir, username):
    r'''
    Generates an eos address from an afs directory for a given username.

    :param afs_dir: directory in AFS
    :param username: username
    :return: string with the eos directory
    '''
    return f'/eos/user/{username[0]}/{username}/sps_lhc_simulation_data/' + afs_dir.split('sps_lhc_transfer/')[-1]


def load_and_stage_simulation_data(data_dir, data_name, target_dir):
    r'''
    Generates a directory and loads the data from eos into that file.

    :param data_dir: directory with the eos data
    :param data_name: the filename of the data in eos
    :param target_dir: the target directory
    :return: string with console commands
    '''

    console_command = ''

    if not os.path.isdir(target_dir):
        console_command += f'mkdir -p {target_dir}\n'

    if not os.path.isfile(f'{data_dir}{data_name}'):
        console_command += f'eos cp {data_dir}{data_name} {target_dir}.\n'

    return console_command


def stage_data_for_beam_generation(custom_beam, usr):
    r'''
    Generates the console commands necessary to stage data for generating beams using HTCondor.

    :param custom_beam: directory of the data to laod form
    :param usr: username
    :return: string containing the console commands
    '''

    eos_dir = convert_afs_to_eos(custom_beam, username=usr)
    beam_params = ['bunch_lengths.npy', 'exponents.npy', 'bunch_intensities.npy', 'bunch_positions.npy']
    console_command = ''

    for param in beam_params:
        if os.path.isfile(eos_dir + param):
            console_command += load_and_stage_simulation_data(eos_dir, param, custom_beam)

    return console_command


def stage_beams(beam_dir, eos_beam_dir, simulated):
    r'''
    Generates the console commands necessary to stage the macroparticle beam for simulation in HTCondor.

    :param beam_dir: AFS directory of the beam
    :param eos_beam_dir: EOS directory of the beam to load from
    :param simulated: bool to determine if the beam has been simulated at SPS flattop or not
    :return: string containing the console commands
    '''

    console_commands = ''

    if simulated:
        sim_stat = 'simulated'
    else:
        sim_stat = 'generated'

    if not os.path.isfile(beam_dir + f'{sim_stat}_beam.npy'):
        console_commands += load_and_stage_simulation_data(eos_beam_dir, f'{sim_stat}_beam.npy', beam_dir)

    return console_commands


def stage_data_for_simulation(beam_name, get_from, save_to, usr,
                              sps=True, simulated=False, inj_sch='single_injection.yaml', lxplus=True):
    r'''
    Generating console commands for the bash file to correctly stage in the necessary data from eos to perform
    the simulation.

    :param beam_name: name of initial beam
    :param get_from: directory to get data from
    :param save_to: directory to save data to
    :param usr: usernam
    :param sps: bool if the simulation is in the SPS (True) or in the LHC (False)
    :param simulated: bool if the initial beam has been simulated at SPS flattop (True) or not (False)
    :param inj_sch: name of the yaml-file containing the LHC injection scheme for this simulation.
    :return: a string containing the console commands
    '''
    if lxplus:
        lxdir = f'/afs/cern.ch/work/{usr[0]}/{usr}/sps_lhc_transfer/'
    else:
        lxdir = '../'

    beam_dir = lxdir + f'generated_beams/{beam_name}/'
    eos_beam_dir = convert_afs_to_eos(beam_dir, usr)

    console_commands = ''

    # Generating input and output directories
    if not os.path.isdir(beam_dir):
        console_commands += f'mkdir -p {beam_dir}\n'

    if not os.path.isdir(get_from) and get_from != '':
        console_commands += f'mkdir -p {get_from}\n'

    if not os.path.isdir(save_to) and save_to != '':
        console_commands += f'mkdir -p {save_to}\n'

    # Loading the generating settings used for the beam to be used in simulation
    if not os.path.isfile(beam_dir + 'generation_settings.yaml'):
        console_commands += load_and_stage_simulation_data(eos_beam_dir, 'generation_settings.yaml', beam_dir)

    # staging the macroparticle model of the beam
    if sps:
        console_commands += stage_beams(beam_dir, eos_beam_dir, simulated=False)

    else:
        console_commands += stage_beams(beam_dir, eos_beam_dir, simulated=simulated)

        # if multiple injections into the LHC are simulated then these beams are also loaded into afs
        inj_dict = fetch_from_yaml(inj_sch, lxdir + 'injection_schemes/')
        inj_beam_ids = list(inj_dict.keys())

        for beam_id in inj_beam_ids:
            if beam_id == 'no injections':
                pass
            else:
                # AFS directory for beam i
                beam_dir_i = lxdir + f'generated_beams/{beam_id}/'
                # EOS directory to load beam i from
                eos_beam_dir_i = convert_afs_to_eos(beam_dir_i, usr)
                # Bool if beam has been simulated at SPS flattop or not
                simulated_i = bool(inj_dict[beam_id][2])
                # Add the necessary console commands
                console_commands += stage_beams(beam_dir_i, eos_beam_dir_i, simulated=simulated_i)

    return console_commands


def stage_out_simulation_results(save_to, usr):

    console_command = ''
    eos_save_to = convert_afs_to_eos(save_to, usr)
    console_command += f'eos mkdir -p {eos_save_to}\n'
    console_command += f'eos mv -r {save_to} {eos_save_to}.\n'

    return console_command



