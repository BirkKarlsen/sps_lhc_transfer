from __future__ import annotations

import numpy as np
import pandas as pd

import os
from datetime import date

from blond.beam.beam import Beam, Proton
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.distributions import matched_from_distribution_function
from blond.impedances.impedance_sources import InputTable
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.llrf.beam_feedback import BeamFeedback
from blond.llrf.rf_noise import FlatSpectrum
from SPS.impedance_scenario import scenario, impedance2blond


class SPSGeneration:
    # SPS Machine Parameters --------------------------------------------------------------------------------------
    C = 2 * np.pi * 1100.009  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    h = 4620  # Harmonic number [-]
    gamma_t = 18.0  # Transition gamma [-]
    alpha = 1. / gamma_t / gamma_t  # First order mom. comp. factor [-]

    dphi = 0  # 200 MHz Phase modulation/offset [rad]
    dphi_800 = np.pi  # 800 MHz Phase modulation/offset [rad]

    # BLonD Objects
    profile = None
    induced_voltage = None
    rf_tracker = None
    full_tracker = None
    exponent = 1.5
    bunch_length = 1.6

    def __init__(self, args):
        print("Setting SPS as machine")
        self.N_p = args.intensity * 1e11         # Bunch intensity [p/b]
        self.N_m = args.n_macroparticles

        # SPS Machine Parameters --------------------------------------------------------------------------------------
        V = args.voltage_200 * 1e6          # 200 MHz RF voltage [V]
        V_800 = args.voltage_800 * V        # 800 MHz RF voltage [V]

        # Set up Ring and RF Station
        # SPS Ring
        self.ring = Ring(self.C, self.alpha, self.p_s, Proton(), n_turns=1)

        # RF Station
        self.rfstation = RFStation(
            self.ring, [self.h, 4 * self.h],
            [V, V_800], [self.dphi, self.dphi_800],
            n_rf=2
        )

        self.beam = Beam(self.ring, self.N_m, self.N_p)

    def set_profile(self, args):

        self.profile = Profile(self.beam,
                               CutOptions((-1.5) * self.rfstation.t_rf[0, 0],
                                          (2.5) * self.rfstation.t_rf[0, 0],
                                          4 * (2 ** 7)),
                               FitOptions(fit_option='fwhm'))
        self.profile.track()

    def generate_bunch(self, n_iterations, args, impedance_model="futurePostLS2_SPS_f1.txt"):
        self.exponent = args.exponent
        self.bunch_length = args.bunchlength * 1e-9

        print("Adding SPS impedance model")
        imp_scenario = scenario(impedance_model)
        imp_model = impedance2blond(imp_scenario.table_impedance)
        freq_res = 1 / self.rfstation.t_rev[0]

        imp_freq = InducedVoltageFreq(self.beam, self.profile, imp_model.impedanceList, freq_res)
        impedance_table = InputTable(imp_freq.freq, imp_freq.total_impedance.real * self.profile.bin_size,
                                     imp_freq.total_impedance.imag * self.profile.bin_size)

        impedance_freq = InducedVoltageFreq(self.beam, self.profile, [impedance_table],
                                            frequency_resolution=freq_res)
        self.induced_voltage = TotalInducedVoltage(self.beam, self.profile, [impedance_freq])

        # Initialize the RF tracker
        self.rf_tracker = RingAndRFTracker(self.rfstation, self.beam,
                                           TotalInducedVoltage=self.induced_voltage,
                                           Profile=self.profile)

        # Initialize the Full Ring and RF tracker
        self.full_tracker = FullRingAndRF([self.rf_tracker])

        print(f"Generating a {self.bunch_length * 1e9:.3f} ns bunch")
        matched_from_distribution_function(self.beam, self.full_tracker,
                                           TotalInducedVoltage=self.induced_voltage,
                                           bunch_length=self.bunch_length,
                                           bunch_length_fit="fwhm",
                                           distribution_type="binomial",
                                           distribution_exponent=self.exponent,
                                           n_iterations=n_iterations)


class LHCInjection:
    # LHC Machine Parameters --------------------------------------------------------------------------------------
    C = 26658.883  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    h = 35640  # Harmonic number [-]
    dphi = 0  # Phase modulation/offset [rad]

    beam = None
    beam_feedback = None
    profile = None
    induced_voltage = None
    rf_tracker = None
    intra_beam_scattering = None

    emit_x = None
    emit_y = None

    def __init__(self, args, lxdir: str):
        # LHC Machine Parameters --------------------------------------------------------------------------------------
        gamma_t = args.gamma_t              # Transition gamma [-]
        alpha = 1. / gamma_t / gamma_t      # First order mom. comp. factor [-]
        V = args.voltage * 1e6              # RF voltage [V]

        self.lxdir = lxdir
        self.N_t = args.number_of_turns

        # Set up Ring and RF Station
        # LHC Ring
        self.ring = Ring(self.C, alpha, self.p_s, Proton(), n_turns=1)

        # RF Station
        self.rfstation = RFStation(
            self.ring, [self.h],
            [V], [self.dphi],
            n_rf=1
        )

    def set_profile(self):
        self.profile = Profile(self.beam,
                               CutOptions(-1.5 * self.rfstation.t_rf[0, 0],
                                          2.5 * self.rfstation.t_rf[0, 0],
                                          4 * (2 ** 7)),
                               FitOptions(fit_option='fwhm'))
        self.profile.track()

    def inject_beam(self, beam: Beam, injection_shift: float):
        self.beam = Beam(self.ring, beam.n_macroparticles, beam.intensity)

        self.beam.dE[:] = beam.dE[:]
        self.beam.dt[:] = beam.dt[:] + injection_shift

    def set_injection_errors(self, energy_error: float, phase_error: float):
        print(f"Add injection error of {energy_error:.3f} MeV and {phase_error:.3f} degrees...")
        self.beam.dE += energy_error * 1e6
        self.beam.dt += phase_error / 360 * self.rfstation.t_rf[0, 0]

    def set_induced_voltage(self, model_str: str):
        f_r = 5e9
        freq_res = 1 / self.rfstation.t_rev[0]
        imp_data = np.loadtxt(self.lxdir + 'impedance/' + model_str, skiprows=1)
        imp_ind = imp_data[:, 0] < 2 * f_r
        impedance_table = InputTable(imp_data[imp_ind, 0], imp_data[imp_ind, 1], imp_data[imp_ind, 2])

        impedance_freq = InducedVoltageFreq(self.beam, self.profile, [impedance_table],
                                            frequency_resolution=freq_res)
        self.induced_voltage = TotalInducedVoltage(self.beam, self.profile, [impedance_freq])

    def set_beam_feedback(self, args):
        print("Adding beam control to the simulation...")

        if args.pl_gain is None:
            PL_gain = 1 / (5 * self.ring.t_rev[0])
        else:
            PL_gain = args.pl_gain

        if args.sl_gain is None:
            SL_gain = PL_gain / 10
        else:
            SL_gain = args.sl_gain

        bl_config = {'machine': 'LHC',
                     'PL_gain': PL_gain,
                     'SL_gain': SL_gain}

        self.beam_feedback = BeamFeedback(self.ring, self.rfstation, self.profile, bl_config)

    def set_rf_noise(self):
        # TODO: check implementation with Helga
        print("Adding RF noise...")

        rf_noise = FlatSpectrum(self.ring, self.rfstation, delta_f=1.12455000e-02, fmin_s0=0,
                                fmax_s0=1.1, seed1=1234, seed2=7564,
                                initial_amplitude=1.11100000e-07)
        rf_noise.generate()

        self.rfstation.phi_noise = np.array(rf_noise.dphi, ndmin=2)

    def construct_tracker(self):
        print("Constructing tracker")
        # Initialize the RF tracker
        self.rf_tracker = RingAndRFTracker(self.rfstation, self.beam,
                                           BeamFeedback=self.beam_feedback,
                                           TotalInducedVoltage=self.induced_voltage,
                                           Profile=self.profile)

    def compute_losses(self):
        pass

    def compute_induced_voltage(self):
        pass

    def track(self):
        self.rf_tracker.track()
        self.profile.track()


def main():
    # Parse Arguments --------------------------------------------------------------------------------------------------
    from lxplus_setup.parsers import single_bunch_simulation_parser

    parser = single_bunch_simulation_parser(add_help=True)
    args = parser.parse_args()

    # Options ----------------------------------------------------------------------------------------------------------
    lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
    LXPLUS = True
    if 'birkkarlsen-baeck' in os.getcwd():
        lxdir = '../'
        LXPLUS = False
        print('\nRunning locally...')
    else:
        print('\nRunning in lxplus...')

    # Make simulation output folder
    if args.date is None:
        today = date.today()
        save_to = lxdir + f'simulation_results/{today.strftime("%b-%d-%Y")}/{args.simulation_name}/'
        if not os.path.isdir(save_to):
            os.makedirs(save_to)
    else:
        save_to = lxdir + f'simulation_results/{args.date}/{args.simulation_name}/'
        if not os.path.isdir(save_to):
            os.makedirs(save_to)

    sps_generation = SPSGeneration(args)
    sps_generation.set_profile(args)
    sps_generation.generate_bunch(
        n_iterations=50, args=args
    )

    lhc_injection = LHCInjection(args, lxdir=lxdir)
    injection_shift = ...
    lhc_injection.inject_beam(sps_generation.beam, injection_shift)
    lhc_injection.set_injection_errors(args.energy_error, args.phase_error)

    lhc_injection.set_profile()

    # Adding an impedance model
    if args.include_impedance:
        lhc_injection.set_induced_voltage(args.impedance_model)

    # Adding the beam feedback
    if args.include_global:
        lhc_injection.set_beam_feedback(args)

    # Constructing the tracker
    lhc_injection.construct_tracker()

    dt_int = args.dt_int
    dt_cont = args.dt_cont
    dt_beam = args.dt_beam

    indx = 0

    evolution = {
        'time': np.zeros(lhc_injection.N_t // dt_cont),
        'tau': np.zeros(lhc_injection.N_t // dt_cont),
        'intensity': np.zeros(lhc_injection.N_t // dt_cont),
        'dt_rms': np.zeros(lhc_injection.N_t // dt_cont),
        'dE_rms': np.zeros(lhc_injection.N_t // dt_cont),
        'dt_mean': np.zeros(lhc_injection.N_t // dt_cont),
        'dE_mean': np.zeros(lhc_injection.N_t // dt_cont),
        'rms_emittance': np.zeros(lhc_injection.N_t // dt_cont),
    }

    for i in range(lhc_injection.N_t):
        lhc_injection.track()

        if i % dt_int == 0:
            lhc_injection.compute_induced_voltage()

        if i % dt_cont == 0:
            lhc_injection.compute_losses()

            evolution['time'][indx] = (i - 1) * lhc_injection.rfstation.t_rev[lhc_injection.rfstation.counter[0]]
            evolution['tau'][indx] = lhc_injection.profile.bunchLength
            evolution['intensity'][indx] = lhc_injection.beam.ratio * lhc_injection.beam.n_macroparticles_alive

            lhc_injection.beam.statistics()
            evolution['dt_rms'][indx] = lhc_injection.beam.sigma_dt
            evolution['dE_rms'][indx] = lhc_injection.beam.sigma_dE
            evolution['dt_mean'][indx] = lhc_injection.beam.mean_dt
            evolution['dE_mean'][indx] = lhc_injection.beam.mean_dE
            evolution['rms_emittance'][indx] = lhc_injection.beam.epsn_rms_l

            indx += 1

        if i % dt_beam == 0:
            df = pd.DataFrame(evolution)
            df.to_hdf(save_to + 'output.b5', 'Beam')


if __name__ == "__main__":
    main()
