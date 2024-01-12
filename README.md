# Repository 'mesoscale_simulations_cebc'
Code for model, simulations and analysis of classical eyeblink conditioning with a spiking neural network, E-GLIF neurons and spike-driven plasticity, as described in Geminiani et al., biorXiv, 2022 - doi: https://doi.org/10.1101/2023.06.20.545667


# Requirements:
- NEST simulator v 2.18
- <a href="https://github.com/dbbs-lab/cereb-nest">cereb-nest</a>  NEST extension module

# Repository content:
- `configuration` contains the configuration file of the network model with construction (<a href="https://github.com/dbbs-lab/bsb">BSB</a> v3.8+) and simulation parameters
- `utils` includes the code for simulation parameter setting and functions
- `sim_run.py` and `sim_analysis.py` are scripts to run and ananlyse simulations, respectively
- `sim_run.sh` contains the .slurm instructions to run simulations on Piz Daint supercomputer
