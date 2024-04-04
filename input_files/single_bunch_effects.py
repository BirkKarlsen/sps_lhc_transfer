r'''
Simulation of single-bunch effects at flat bottom in the LHC.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
print('\nImporting...')
import numpy as np
import os
import yaml
from datetime import date

from beam_dynamics_tools.simulation_functions.diagnostics.lhc_diagnostics import LHCDiagnostics
from beam_dynamics_tools.simulation_functions.machine_beam_processes import fetch_momentum_program
from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml


from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.beam_feedback import BeamFeedback
from blond.llrf.rf_noise import FlatSpectrum
from blond.impedances.impedance_sources import InputTable
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage

from intra_beam_scattering.IBSfunctions import NagaitsevIBS
from intra_beam_scattering.loadlattice import prepareTwiss

from SPS.impedance_scenario import scenario, impedance2blond


class SingleBunch:

    def __init__(self, args, lxdir, turns):
        # Beam parameters
        self.N_p = args.intensity * 1e11         # Bunch intensity [p/b]
        self.N_m = args.n_macroparticles
        self.exponent = args.exponent
        self.bunch_length = args.bunchlength * 1e-9
        self.injection_shift = 0

        self.lxdir = lxdir
        self.N_t = turns

        # BLonD Objects
        self.ring = None
        self.rfstation = None
        self.beam = None
        self.beam_feedback = None
        self.profile = None
        self.induced_voltage = None
        self.rf_tracker = None
        self.full_tracker = None
        self.diagnostics = None
        self.intra_beam_scattering = None

        self.emit_x = None
        self.emit_y = None

    def set_sps_machine(self, args):
        # SPS Machine Parameters --------------------------------------------------------------------------------------
        C = 2 * np.pi * 1100.009            # Machine circumference [m]
        p_s = 450e9                         # Synchronous momentum [eV/c]
        h = 4620                            # Harmonic number [-]
        gamma_t = 18.0                      # Transition gamma [-]
        alpha = 1./gamma_t/gamma_t          # First order mom. comp. factor [-]
        V = args.voltage_200 * 1e6          # 200 MHz RF voltage [V]
        V_800 = args.voltage_800 * V        # 800 MHz RF voltage [V]
        dphi = 0                            # 200 MHz Phase modulation/offset [rad]
        dphi_800 = np.pi                    # 800 MHz Phase modulation/offset [rad]

        # Set up Ring and RF Station
        # SPS Ring
        self.ring = Ring(C, alpha, p_s, Proton(), n_turns=1)

        # RF Station
        self.rfstation = RFStation(self.ring, [h, 4 * h], [V, V_800], [dphi, dphi_800], n_rf=2)

    def set_lhc_machine(self, args):
        # LHC Machine Parameters --------------------------------------------------------------------------------------
        C = 26658.883                       # Machine circumference [m]
        p_s = 450e9                         # Synchronous momentum [eV/c]
        h = 35640                           # Harmonic number [-]
        gamma_t = args.gamma_t              # Transition gamma [-]
        alpha = 1. / gamma_t / gamma_t      # First order mom. comp. factor [-]
        V = args.voltage * 1e6              # RF voltage [V]
        dphi = 0                            # Phase modulation/offset [rad]

        # LHC ramp
        if args.ramp is not None:
            ramp = fetch_momentum_program(self.lxdir + 'momentum_programs/' + args.momentum_program,
                                          C=C, particle_mass=Proton().mass, target_momentum=args.ramp * 1e9)
            cycle = np.concatenate((np.linspace(p_s, p_s, self.N_t), ramp))
            self.N_t = len(cycle) - 1
            print(f'Ramp length is {len(ramp)} turns')
        else:
            cycle = np.linspace(p_s, p_s, self.N_t + 1)

        if self.rfstation is not None:
            self.injection_shift = self.rfstation.t_rf[0, 0] / 2

        # LHC ring
        self.ring = Ring(C, alpha, cycle, Proton(), n_turns=self.N_t)

        # 400MHz RF station
        self.rfstation = RFStation(self.ring, [h], [V], [dphi])

        if self.injection_shift != 0:
            self.injection_shift -= self.rfstation.t_rf[0, 0] / 2

    def set_beam(self):
        beam_tmp = None

        # Store already existing beam temporarily
        if self.beam is not None:
            beam_tmp = self.beam

        # Make new beam object with the new ring
        self.beam = Beam(self.ring, self.N_m, self.N_p)

        # Inject the beam if there was existing beam from before
        if beam_tmp is not None:
            self.beam.dt = beam_tmp.dt
            self.beam.dE = beam_tmp.dE

        self.profile = Profile(self.beam, CutOptions((-0.5) * self.rfstation.t_rf[0, 0],
                                                     (1.5) * self.rfstation.t_rf[0, 0],
                                                     2 * (2**7)))

    def construct_tracker(self):
        # Initialize the RF tracker
        self.rf_tracker = RingAndRFTracker(self.rfstation, self.beam,
                                           BeamFeedback=self.beam_feedback,
                                           TotalInducedVoltage=self.induced_voltage,
                                           Profile=self.profile)

        # Initialize the Full Ring and RF tracker
        self.full_tracker = FullRingAndRF([self.rf_tracker])

    def generate_bunch(self, n_iterations):

        matched_from_distribution_function(self.beam, self.full_tracker,
                                           TotalInducedVoltage=self.induced_voltage,
                                           bunch_length=self.bunch_length,
                                           bunch_length_fit="fwhm",
                                           distribution_type="binomial",
                                           distribution_exponent=self.exponent,
                                           n_iterations=n_iterations)

    def set_induced_voltage(self, model_str, machine="SPS"):
        if machine == "SPS":
            imp_scenario = scenario(model_str)
            imp_model = impedance2blond(imp_scenario.table_impedance)
            freq_res = 1 / self.rfstation.t_rev[0]

            imp_freq = InducedVoltageFreq(self.beam, self.profile, imp_model.impedanceList, freq_res)
            impedance_table = InputTable(imp_freq.freq, imp_freq.total_impedance.real * self.profile.bin_size,
                                         imp_freq.total_impedance.imag * self.profile.bin_size)
        else:
            f_r = 5e9
            freq_res = 1 / self.rfstation.t_rev[0]

            imp_data = np.loadtxt(self.lxdir + 'impedance/' + model_str, skiprows=1)
            imp_ind = imp_data[:, 0] < 2 * f_r
            impedance_table = InputTable(imp_data[imp_ind, 0], imp_data[imp_ind, 1], imp_data[imp_ind, 2])

        impedance_freq = InducedVoltageFreq(self.beam, self.profile, [impedance_table],
                                            frequency_resolution=freq_res)
        self.induced_voltage = TotalInducedVoltage(self.beam, self.profile, [impedance_freq])

    def set_beam_feedback(self, args):

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

    def set_rf_noise(self, noise_file):
        # TODO: check implementation with Helga

        rf_noise = FlatSpectrum(self.ring, self.rfstation, delta_f=1.12455000e-02, fmin_s0=0,
                                fmax_s0=1.1, seed1=1234, seed2=7564,
                                initial_amplitude=1.11100000e-07)
        rf_noise.generate()

        self.rfstation.phi_noise = np.array(rf_noise.dphi, ndmin=2)

    def set_intra_beam_scattering(self, twiss_file, emit_x, emit_y):

        twiss = prepareTwiss(twiss_file)
        twiss['slip'] = self.rfstation.eta_0[0]

        self.emit_x = emit_x / self.beam.gamma / self.beam.beta
        self.emit_y = emit_y / self.beam.gamma / self.beam.beta

        self.intra_beam_scattering = NagaitsevIBS()
        self.intra_beam_scattering.set_beam_parameters(self.beam)
        self.intra_beam_scattering.set_optic_functions(twiss)

    def set_injection_errors(self, energy_error, phase_error):
        self.beam.dE += energy_error * 1e6
        self.beam.dt += phase_error / 360 * self.rfstation.t_rf[0, 0]

    def set_simulation_diagnostics(self, args, save_to):
        # Fetch injection scheme
        injection_scheme = fetch_from_yaml("single_injection.yaml", self.lxdir + 'injection_schemes/')

        self.diagnostics = LHCDiagnostics(self.rf_tracker, self.profile, self.induced_voltage, LHCCavityLoop=None,
                                          Ring=self.ring, save_to=save_to, get_from=self.lxdir, n_bunches=1,
                                          setting=args.diag_setting, dt_cont=args.dt_cont, dt_prfl=args.dt_prfl,
                                          dt_cl=args.dt_cl, dt_beam=args.dt_beam, dt_ld=args.dt_ld,
                                          injection_scheme=injection_scheme)

    def update_intra_beam_scatter(self):
        self.intra_beam_scattering.calculate_longitudinal_kick(self.emit_x, self.emit_y, self.beam)

    def track(self):
        if self.intra_beam_scattering is not None:
            self.intra_beam_scattering.track(self.profile, self.beam)
            self.emit_x, self.emit_y = self.intra_beam_scattering.emittance_evolution_2D(self.emit_x,
                                                                                         self.emit_y,
                                                                                         1 / self.ring.f_rev[0])

        self.rf_tracker.track()
        self.diagnostics.track()

        if self.induced_voltage is not None:
            self.induced_voltage.induced_voltage_sum()


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

    single_bunch = SingleBunch(args, lxdir, args.number_of_turns)

    # First generate the bunch
    if args.generated_in == 'SPS':
        single_bunch.set_sps_machine(args)
        single_bunch.set_beam()
        if args.include_impedance:
            single_bunch.set_induced_voltage(freq_res=43.3e3, model_str="futurePostLS2_SPS_f1.txt",
                                             machine=args.generated_in)
    else:
        single_bunch.set_lhc_machine(args)
        single_bunch.set_beam()
        if args.include_impedance:
            single_bunch.set_induced_voltage(model_str=args.impedance_model,
                                             machine=args.generated_in)

    single_bunch.construct_tracker()
    single_bunch.generate_bunch(n_iterations=50)

    # Set up the particle tracking
    single_bunch.set_lhc_machine(args)
    single_bunch.set_beam()

    # Introducing injection errors
    single_bunch.set_injection_errors(args.energy_error, args.phase_error)

    # Adding an impedance model
    if args.include_impedance:
        single_bunch.set_induced_voltage(model_str=args.impedance_model,
                                         machine='LHC')

    # Adding the beam feedback
    single_bunch.set_beam_feedback(args)

    # Adding RF noise
    single_bunch.set_rf_noise(args.noise)

    # Adding intra-beam scattering
    if args.intra_beam:
        single_bunch.set_intra_beam_scattering(twiss_file=...,
                                               emit_x=args.emittance_x,
                                               emit_y=args.emittance_y)

    # Constructing the tracker
    single_bunch.construct_tracker()

    # Setting up the diagnostics for the simulation
    single_bunch.set_simulation_diagnostics(args, args.save_to)

    for i in range(args.number_of_turns):
        single_bunch.track()

        if args.intra_beam and i % args.update_ibs == 0:
            single_bunch.update_intra_beam_scatter()


if __name__ == "__main__":
    main()
