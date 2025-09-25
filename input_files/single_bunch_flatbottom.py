from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from datetime import date

from blond.beam.beam import Beam, Proton
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.distributions import matched_from_distribution_function
from blond.impedances.impedance_sources import InputTable, Resonators
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.llrf.beam_feedback import BeamFeedback
from blond.llrf.rf_noise import FlatSpectrum
import blond.utils.bmath as bm
from blond.intra_beam_scattering.intra_beam_scattering import NagaitsevScattering
from blond.intra_beam_scattering.load_lattice import TwissParameters

from beam_dynamics_tools.analytical_functions.transfer_functions import H_a, H_d, Z_cl
from beam_dynamics_tools.analytical_functions.potential_wells import (single_rf_potential,
                                                                      compute_bunch_length)


def prepare_gpu_simulation(obj):
    import cupy as cp

    dev = cp.cuda.Device(0)
    comp_capability = dev.compute_capability
    print(f"Compute capability: {comp_capability}")
    bm.use_gpu()

    def convert_to_gpu(obj) -> None:
        gpu_call = getattr(obj, "to_gpu", None)
        if callable(gpu_call):
            gpu_call()

    objects_in_sim = obj.__dict__
    for key, value in objects_in_sim.items():
        convert_to_gpu(value)


class LHCGeneration:
    # SPS Machine Parameters --------------------------------------------------------------------------------------
    C = 26658.883  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    h = 35640  # Harmonic number [-]
    dphi = 0  # Phase modulation/offset [rad]

    # BLonD Objects
    profile = None
    induced_voltage = None
    rf_tracker = None
    full_tracker = None
    exponent = 1.5
    bunch_length = 1.6
    emittance = 0.58

    def __init__(self, args, lxdir):
        print("Setting LHC as machine")
        self.N_p = args.intensity * 1e11         # Bunch intensity [p/b]
        self.N_m = args.n_macroparticles

        print(f'Setting the LHC voltage to {args.voltage:.2f} MV...')
        # LHC Machine Parameters --------------------------------------------------------------------------------------
        gamma_t = args.gamma_t  # Transition gamma [-]
        alpha = 1. / gamma_t / gamma_t  # First order mom. comp. factor [-]
        V = args.voltage * 1e6  # RF voltage [V]

        self.lxdir = lxdir
        self.N_t = args.number_of_turns

        # Set up Ring and RF Station
        # LHC Ring
        self.ring = Ring(self.C, alpha, self.p_s, Proton(), n_turns=self.N_t)

        # RF Station
        self.rfstation = RFStation(
            self.ring, [self.h],
            [V], [self.dphi],
            n_rf=1
        )

        self.beam = Beam(self.ring, self.N_m, self.N_p)

    def set_profile(self, args):

        self.profile = Profile(self.beam,
                               CutOptions((-1.5) * self.rfstation.t_rf[0, 0],
                                          (2.5) * self.rfstation.t_rf[0, 0],
                                          4 * (2 ** 7)),
                               FitOptions(fit_option='fwhm'))
        self.profile.track()

    def set_induced_voltage(
            self,
            model_str: str,
            effective: bool = False,
            z_over_n: float = 0.07,
            f_cutoff: float = 5e9
    ):
        print(f'Adding induced voltage...')
        f_r = 5e9
        freq_res = 1 / self.rfstation.t_rev[0] / 1

        imp_data = np.loadtxt(self.lxdir + 'impedance/' + model_str, skiprows=1)
        imp_ind = imp_data[:, 0] < 2 * f_r
        impedance_table = InputTable(imp_data[imp_ind, 0], imp_data[imp_ind, 1], imp_data[imp_ind, 2])

        impedance_list = [impedance_table]

        if "noRF" in model_str:
            print("Added RF cavities...")
            G_a = 6.79e-6  # Analog FB gain [A/V]
            G_d = 10  # Digital FB gain [-]
            tau_loop = 650e-9  # Overall loop delay [s]
            tau_a = 170e-6  # Analog FB delay [s]
            tau_d = 400e-6  # Digital FB delay [s]
            Q_L = 20000  # Loaded Quality factor [-]
            delta_f = -3480

            cavity = Resonators(45 * Q_L, self.rfstation.omega_rf[0, 0] / (2 * np.pi) + delta_f, Q_L)

            ind_freq = InducedVoltageFreq(
                self.beam, self.profile,
                [cavity],
                frequency_resolution=freq_res
            )

            freq = ind_freq.freq
            ind_freq.sum_impedances(freq)

            freq = freq - self.rfstation.omega_rf[0, 0] / (2 * np.pi)
            h_d = lambda f: H_d(f, G_a, G_d, tau_d, 0)
            h_a = lambda f: H_a(f, G_a, tau_a)
            z_cav = ind_freq.total_impedance * self.profile.bin_size
            z_cl = Z_cl(freq, h_a, h_d, z_cav, tau_loop)
            freq = ind_freq.freq

            input_table = InputTable(freq, 8 * z_cl.real, 8 * z_cl.imag)
            impedance_list.append(input_table)

        if effective:
            effective_broadband = Resonators(
                R_S=f_cutoff * z_over_n / self.ring.f_rev[0],
                frequency_R=f_cutoff,
                Q=1
            )

            impedance_list = [effective_broadband]

        impedance_freq = InducedVoltageFreq(
            self.beam, self.profile,
            impedance_list,
            frequency_resolution=freq_res
        )

        self.induced_voltage = TotalInducedVoltage(self.beam, self.profile, [impedance_freq])

    def prepare_generation_bunch(self, args, model_str: str):
        self.exponent = args.exponent
        self.bunch_length = args.bunchlength * 1e-9
        self.emittance = args.emittance

        if self.emittance is not None:
            rf_potential = single_rf_potential(
                V=args.voltage * 1e6,
                harmonic=self.h,
                gamma_t=args.gamma_t,
                C=self.C,
                p_s=self.p_s
            )
            self.bunch_length = compute_bunch_length(
                self.emittance, rf_potential, [0.1e-9, 2.4e-9]
            )

        print("Adding LHC impedance model")
        if bool(args.include_impedance):
            self.set_induced_voltage(model_str)

        # Initialize the RF tracker
        self.rf_tracker = RingAndRFTracker(self.rfstation, self.beam,
                                           TotalInducedVoltage=self.induced_voltage,
                                           Profile=self.profile)

        # Initialize the Full Ring and RF tracker
        self.full_tracker = FullRingAndRF([self.rf_tracker])

    def run_generation(self, n_iterations, tol=0.001e-9):

        if self.emittance is not None:
            print(f"Generating a {self.emittance:.3f} eVs bunch")
            print(f"Equivalent to {self.bunch_length * 1e9:.3f} ns")
        else:
            print(f"Generating a {self.bunch_length * 1e9:.3f} ns bunch")

        n_iter = 20
        iter_num = 0

        while abs(self.bunch_length - self.profile.bunchLength) > tol:
            matched_from_distribution_function(
                self.beam, self.full_tracker,
                TotalInducedVoltage=self.induced_voltage,
                bunch_length=self.bunch_length,
                bunch_length_fit="fwhm",
                distribution_type="binomial",
                distribution_exponent=self.exponent,
                n_iterations=n_iterations,
                n_points_potential=1e4,
                n_points_grid=int(1e3),
                dt_margin_percent=0.40,
            )
            self.profile.track()
            print(f"Generated bunch had a length of {self.profile.bunchLength * 1e9:.3f} ns")
            iter_num += 1

            if iter_num > n_iter:
                break

    def simulate_on_gpu(self) -> None:
        prepare_gpu_simulation(self)


class LHCFlatBottom:
    # LHC Machine Parameters --------------------------------------------------------------------------------------
    C = 26658.883  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    h = 35640  # Harmonic number [-]
    dphi = 0  # Phase modulation/offset [rad]

    beam = None
    beam_feedback = None
    profile = None
    profile_sigma = None
    induced_voltage = None
    rf_tracker = None
    scattering = None

    emit_x = None
    emit_y = None

    def __init__(self, args, lxdir: str):
        print(f'Setting the LHC voltage to {args.voltage:.2f} MV...')
        # LHC Machine Parameters --------------------------------------------------------------------------------------
        gamma_t = args.gamma_t              # Transition gamma [-]
        alpha = 1. / gamma_t / gamma_t      # First order mom. comp. factor [-]
        V = args.voltage * 1e6              # RF voltage [V]

        self.lxdir = lxdir
        self.N_t = args.number_of_turns

        # Set up Ring and RF Station
        # LHC Ring
        self.ring = Ring(self.C, alpha, self.p_s, Proton(), n_turns=self.N_t)

        # RF Station
        self.rfstation = RFStation(
            self.ring, [self.h],
            [V], [self.dphi],
            n_rf=1
        )

        self.track = self.track_without_scattering

    def set_profile(self):
        self.profile = Profile(self.beam,
                               CutOptions(-1.5 * self.rfstation.t_rf[0, 0],
                                          2.5 * self.rfstation.t_rf[0, 0],
                                          4 * (2 ** 6)),
                               FitOptions(fit_option='fwhm'))
        self.profile.track()

        self.profile_sigma = Profile(self.beam,
                                     CutOptions(-1.5 * self.rfstation.t_rf[0, 0],
                                                2.5 * self.rfstation.t_rf[0, 0],
                                                4 * (2 ** 6)),
                                     FitOptions(fit_option='rms'))
        self.profile_sigma.track()

    def inject_beam(self, beam: Beam, injection_shift: float = 0):
        print(f'Injected beam with {beam.n_macroparticles} macro particles and {beam.intensity} protons')
        self.beam = Beam(self.ring, beam.n_macroparticles, beam.intensity)

        self.beam.dE[:] = beam.dE[:]
        self.beam.dt[:] = beam.dt[:] + injection_shift

    def set_induced_voltage(
            self,
            model_str: str,
            effective: bool = False,
            z_over_n: float = 0.07,
            f_cutoff: float = 5e9
    ):
        print(f'Adding induced voltage...')
        f_r = 5e9
        freq_res = 1 / self.rfstation.t_rev[0] / 1

        imp_data = np.loadtxt(self.lxdir + 'impedance/' + model_str, skiprows=1)
        imp_ind = imp_data[:, 0] < 2 * f_r
        impedance_table = InputTable(imp_data[imp_ind, 0], imp_data[imp_ind, 1], imp_data[imp_ind, 2])

        impedance_list = [impedance_table]

        if "noRF" in model_str:
            print("Added RF cavities...")
            G_a = 6.79e-6  # Analog FB gain [A/V]
            G_d = 10  # Digital FB gain [-]
            tau_loop = 650e-9  # Overall loop delay [s]
            tau_a = 170e-6  # Analog FB delay [s]
            tau_d = 400e-6  # Digital FB delay [s]
            Q_L = 20000  # Loaded Quality factor [-]
            delta_f = -3480

            cavity = Resonators(45 * Q_L, self.rfstation.omega_rf[0, 0] / (2 * np.pi) + delta_f, Q_L)

            ind_freq = InducedVoltageFreq(
                self.beam, self.profile,
                [cavity],
                frequency_resolution=freq_res
            )

            freq = ind_freq.freq
            ind_freq.sum_impedances(freq)

            freq = freq - self.rfstation.omega_rf[0, 0] / (2 * np.pi)
            h_d = lambda f: H_d(f, G_a, G_d, tau_d, 0)
            h_a = lambda f: H_a(f, G_a, tau_a)
            z_cav = ind_freq.total_impedance * self.profile.bin_size
            z_cl = Z_cl(freq, h_a, h_d, z_cav, tau_loop)
            freq = ind_freq.freq

            input_table = InputTable(freq, 8 * z_cl.real, 8 * z_cl.imag)
            impedance_list.append(input_table)

        if effective:
            effective_broadband = Resonators(
                R_S=f_cutoff * z_over_n / self.ring.f_rev[0],
                frequency_R=f_cutoff,
                Q=1
            )

            impedance_list = [effective_broadband]

        impedance_freq = InducedVoltageFreq(
            self.beam, self.profile,
            impedance_list,
            frequency_resolution=freq_res
        )

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

    def set_intra_beam_scattering(self, args):
        twiss = TwissParameters()
        twiss.prepare_from_madx(self.lxdir + 'twiss/' + args.twiss_file)

        emit_x = args.emittance_x / self.beam.gamma / self.beam.beta
        emit_y = args.emittance_y / self.beam.gamma / self.beam.beta

        self.scattering = NagaitsevScattering(
            self.beam, self.profile_sigma, self.rfstation, twiss, emit_x, emit_y
        )

        self.track = self.track_with_scattering
        print("Added intra-beam scattering to tracking...")

    def construct_tracker(self):
        print("Constructing tracker")
        # Initialize the RF tracker
        self.rf_tracker = RingAndRFTracker(self.rfstation, self.beam,
                                           BeamFeedback=self.beam_feedback,
                                           TotalInducedVoltage=self.induced_voltage,
                                           Profile=self.profile,
                                           interpolation=True)

    def compute_losses(self):
        self.beam.losses_separatrix(self.ring, self.rfstation)
        self.beam.losses_longitudinal_cut(
            self.rfstation.bucket_center(0) - self.rfstation.t_rf[0, self.rfstation.counter[0]],
            self.rfstation.bucket_center(0) + self.rfstation.t_rf[0, self.rfstation.counter[0]]
        )

    def compute_induced_voltage(self):
        self.induced_voltage.induced_voltage_sum()

    def compute_scattering_params(self):
        self.scattering.update_kick_strength()

    def save_distribution(self, save_to):
        np.savez(
            save_to + f'distribution_{self.rf_tracker.counter[0]}.npz',
            dt=self.beam.dt,
            dE=self.beam.dE
        )

    def track_without_scattering(self):
        self.rf_tracker.track()
        self.profile.track()

    def track_with_scattering(self):
        self.rf_tracker.track()
        self.profile.track()
        self.profile_sigma.track()
        self.scattering.track()

    def simulate_on_gpu(self) -> None:
        prepare_gpu_simulation(self)


def main():
    # Parse Arguments --------------------------------------------------------------------------------------------------
    from lxplus_setup.parsers import single_bunch_simulation_parser

    parser = single_bunch_simulation_parser(add_help=True)

    args = parser.parse_args()

    # Options ----------------------------------------------------------------------------------------------------------
    lxdir = f'/afs/cern.ch/work/b/bkarlsen/sps_lhc_transfer/'
    LXPLUS = True
    if 'Users' in os.getcwd():
        lxdir = '../'
        LXPLUS = False
        print('\nRunning locally...')
    else:
        print('\nRunning in lxplus...')

    # Make simulation output folder
    if args.date is None:
        today = date.today()
        save_to = lxdir + f'simulation_results/{today.strftime("%Y-%m-%d")}/{args.simulation_name}/'
        if not os.path.isdir(save_to):
            os.makedirs(save_to)
    else:
        save_to = lxdir + f'simulation_results/{args.date}/{args.simulation_name}/'
        if not os.path.isdir(save_to):
            os.makedirs(save_to)

    # Generate SPS bunch
    lhc_generation = LHCGeneration(args, lxdir)
    lhc_generation.set_profile(args)
    # Adding an impedance model
    lhc_generation.prepare_generation_bunch(args=args, model_str=args.impedance_model)
    lhc_generation.run_generation(n_iterations=30)

    # LHC injection
    lhc_flatbottom = LHCFlatBottom(args, lxdir=lxdir)
    lhc_flatbottom.inject_beam(lhc_generation.beam)

    lhc_flatbottom.set_profile()

    # Adding an impedance model
    if bool(args.include_impedance):
        lhc_flatbottom.set_induced_voltage(args.impedance_model)

    # Adding the beam feedback
    if bool(args.include_global):
        lhc_flatbottom.set_beam_feedback(args)

    # Adding the beam feedback
    if bool(args.intra_beam):
        lhc_flatbottom.set_intra_beam_scattering(args)

    # Constructing the tracker
    lhc_flatbottom.construct_tracker()

    # Converting all the code run on a GPU
    if bool(args.run_gpu):
        lhc_flatbottom.simulate_on_gpu()

    dt_int = args.dt_int
    dt_cont = args.dt_cont
    dt_beam = args.dt_beam
    dt_ld = args.dt_ld
    dt_ibs = args.update_ibs

    indx = 0

    evolution = {
        'time': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'tau': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'tau_rms': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'intensity': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'dt_rms': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'dE_rms': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'dt_mean': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'dE_mean': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'rms_emittance': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'emittance_x': np.zeros(lhc_flatbottom.N_t // dt_cont),
        'emittance_y': np.zeros(lhc_flatbottom.N_t // dt_cont)
    }

    if lhc_flatbottom.scattering is not None:
        lhc_flatbottom.compute_scattering_params()
        print(f'Initial longitudinal growth rate {lhc_flatbottom.scattering.t_z:.6f} s')

    for i in tqdm(range(lhc_flatbottom.N_t), disable=LXPLUS):
        if i % dt_int == 0 and lhc_flatbottom.induced_voltage is not None:
            lhc_flatbottom.compute_induced_voltage()

        if i % dt_ibs == 0 and lhc_flatbottom.scattering is not None:
            lhc_flatbottom.compute_scattering_params()

        lhc_flatbottom.track()

        if (i - 1) % dt_cont == 0:
            lhc_flatbottom.compute_losses()

            evolution['time'][indx] = (i - 1) * lhc_flatbottom.rfstation.t_rev[lhc_flatbottom.rfstation.counter[0]]
            evolution['tau'][indx] = lhc_flatbottom.profile.bunchLength
            evolution['tau_rms'][indx] = lhc_flatbottom.profile_sigma.bunchLength
            evolution['intensity'][indx] = lhc_flatbottom.beam.ratio * lhc_flatbottom.beam.n_macroparticles_alive

            lhc_flatbottom.beam.statistics()
            evolution['dt_rms'][indx] = lhc_flatbottom.beam.sigma_dt
            evolution['dE_rms'][indx] = lhc_flatbottom.beam.sigma_dE
            evolution['dt_mean'][indx] = lhc_flatbottom.beam.mean_dt
            evolution['dE_mean'][indx] = lhc_flatbottom.beam.mean_dE
            evolution['rms_emittance'][indx] = lhc_flatbottom.beam.epsn_rms_l

            if lhc_flatbottom.scattering is not None:
                evolution['emittance_x'][indx] = lhc_flatbottom.scattering.emittance_x \
                                                 * lhc_flatbottom.beam.beta * lhc_flatbottom.beam.gamma
                evolution['emittance_y'][indx] = lhc_flatbottom.scattering.emittance_y \
                                                 * lhc_flatbottom.beam.beta * lhc_flatbottom.beam.gamma

            indx += 1

        if i % dt_beam == 0:
            df = pd.DataFrame(evolution)
            df.to_hdf(save_to + 'output.h5', 'Beam')

        if i % dt_ld == 0:
            lhc_flatbottom.save_distribution(save_to)

    df = pd.DataFrame(evolution)
    df.to_hdf(save_to + 'output.h5', 'Beam')
    lhc_flatbottom.save_distribution(save_to)


if __name__ == "__main__":
    main()
