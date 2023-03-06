# SPS-LHC Transfer
Repository of input files to simulate beams in the SPS and the behavior when injected into the LHC

## Simulation Setup
The folder ```input_files/``` contains scripts to generate beams in the SPS (```generate_sps_beams.py```), 
simulate beams at SPS flattop (```sps_flattop.py```) and injection into LHC (```lhc_injection.py```) using ```blond```.

### Generating an SPS Beam
First you generate a beam. The result will then be saved in the folder in the ```generated_beams/```-folder. 
This folder will contain a ```.yaml```-file with the parameters that were used to generate the beam and a 
```generated_beam.npy```-file with the generated beam. The user can choose the name of this folder and if no name is 
given, the folder name will be generated in a standard way. 

### Simulating at SPS flattop
When a beam is generated you can either send it straight to the LHC for simulations at injection or first simulate the
beam at SPS flattop. The simulation at SPS flattop will use the information in the ```.yaml```-file to recreate the
conditions the beam was generated in and continue simulating this. When the simulation is done the beam will be saved as
```simulated_beam.npy``` and the number of turns the beam was simulated for in the SPS will be saved to 
```generation_settings.yaml```. 

### Simulations at LHC Injection
Lastly, you can simulate the beam at LHC injection. This script will take beam either from the ```generated_beam.npy```
or ```simulated_beam.npy```, depending on what the user wants. 

## Simulating in lxplus
The simulations are possible to do in ```lxplus``` using this repository. Both beam generation and simulations can be 
set up using the scripts in ```lxplus_setup/```.

## What you need
To run the simulations in this repository you need:
 - The python packages ```numpy```, ```scipy``` and ```yaml```.
 - The ```blond``` simulation suit.
 - The repository ```beam_dynamics_tools```