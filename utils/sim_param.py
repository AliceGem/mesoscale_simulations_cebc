"""
Created on Sept 2019

@author: Alice Geminiani - alice.geminiani@unipv.it
olivocerebellar spiking neural network with E-GLIF neurons and PLASTICITY, for classical eyeblink conditioning simulations
Publication: Geminiani et al., biorXiv, 2022 - doi: https://doi.org/10.1101/2023.06.20.545667

NEST CEBC simulation parameters from scaffold configuration file .json
"""

import json
import pickle

file_config_Zneg = "configuration/mouse_cerebellum_Z-.json"
file_config_Zpos = "configuration/mouse_cerebellum_Z+.json"
file_network = 'balanced_DCN_IO.hdf5'
f = open(file_config_Zneg)
f1 = open(file_config_Zpos)
config = json.load(f)
config1 = json.load(f1)

with open('labelled_ids.pkl', 'rb') as f_label:  # Python 3: open(..., 'rb')
	pos, neg = pickle.load(f_label)

# Cell type ID (can be changed without constraints)
cell_types = list(config['cell_types'].keys())

connection_types = list(config['simulations']['stim_on_MFs']['connection_models'].keys())		#


# PARAMETERS
"""
CEREBELLAR NEURONS
"""

# In E-GLIF, 3 synaptic receptos per neuron: the first is always associated to exc, the second to inh, the third to remaining synapse type; the third (tau_exc2) is assigned to 1.0
param_neurons = {}

for ct in cell_types:
	if ct in config['simulations']['stim_on_MFs']['cell_models'].keys():		# Check if the cell_type is among the cell_models
		if 'parameters' in config['simulations']['stim_on_MFs']['cell_models'][ct].keys():			# Check if the parameters of the current cell are defined (not defined cos not needed for parrot neurons)
			if ct in pos.keys():
				param_neurons[ct] = {}
				param_neurons[ct]['neg'] = config['simulations']['stim_on_MFs']['cell_models'][ct]['parameters']
				param_neurons[ct]['neg'].update(config['simulations']['stim_on_MFs']['cell_models'][ct]['eglif_cond_alpha_multisyn'])
				param_neurons[ct]['pos'] = config1['simulations']['stim_on_MFs']['cell_models'][ct]['parameters']
				param_neurons[ct]['pos'].update(config1['simulations']['stim_on_MFs']['cell_models'][ct]['eglif_cond_alpha_multisyn'])
			else:
				param_neurons[ct] = config['simulations']['stim_on_MFs']['cell_models'][ct]['parameters']
				param_neurons[ct].update(config['simulations']['stim_on_MFs']['cell_models'][ct]['eglif_cond_alpha_multisyn'])

# Synapse parameters (alpha conductances in NEST have the same rise and decay time; we use ref decay time from literature)
Erev_exc = 0.0		# [mV]	#[Cavallari et al, 2014]
Erev_inh = -80.0		# [mV]

"""
CEREBELLAR CONNECTIVITY
"""

# Connections
DEFAULT_PLAST = 0.0		# Default plasticity parameter (LTP and LTD)
param_connections = {}

for ct in connection_types:
	print(ct)
	# weight and delay
	if config['connection_types'][ct]['to_cell_types'][0]['type'] in pos.keys():			# Microzone differentiation
		param_connections[ct] = {}
		# Z-
		param_connections[ct]['neg'] = config['simulations']['stim_on_MFs']['connection_models'][ct]['connection']
		param_connections[ct]['neg'] = config['simulations']['stim_on_MFs']['connection_models'][ct]['connection']
		param_connections[ct]['neg'].update({'sender': config['connection_types'][ct]['from_cell_types'][0]['type'], 'receiver': config['connection_types'][ct]['to_cell_types'][0]['type']})
		if param_connections[ct]['neg']['receiver'] in param_neurons.keys():			# In this case the neuron is an eglif_cond_alpha_multisyn and needs receptor specification
			param_connections[ct]['neg'].update({'receptor': param_neurons[param_connections[ct]['neg']['receiver']]['receptors'][param_connections[ct]['neg']['sender']]})
		# Z+
		param_connections[ct]['pos'] = config1['simulations']['stim_on_MFs']['connection_models'][ct]['connection']
		param_connections[ct]['pos'] = config['simulations']['stim_on_MFs']['connection_models'][ct]['connection']
		param_connections[ct]['pos'].update({'sender': config['connection_types'][ct]['from_cell_types'][0]['type'], 'receiver': config['connection_types'][ct]['to_cell_types'][0]['type']})
		if param_connections[ct]['pos']['receiver'] in param_neurons.keys():			# In this case the neuron is an eglif_cond_alpha_multisyn and needs receptor specification
			param_connections[ct]['pos'].update({'receptor': param_neurons[param_connections[ct]['pos']['receiver']]['receptors'][param_connections[ct]['neg']['sender']]})
	else:
		param_connections[ct] = config['simulations']['stim_on_MFs']['connection_models'][ct]['connection']
		param_connections[ct].update({'sender': config['connection_types'][ct]['from_cell_types'][0]['type'], 'receiver': config['connection_types'][ct]['to_cell_types'][0]['type']})
		if param_connections[ct]['receiver'] in param_neurons.keys():			# In this case the neuron is an eglif_cond_alpha_multisyn and needs receptor specification
			param_connections[ct].update({'receptor': param_neurons[param_connections[ct]['receiver']]['receptors'][param_connections[ct]['sender']]})

	if ct == 'parallel_fiber_to_purkinje' :
		param_connections[ct]['neg'].update({'plastic': True, 'hetero': True, 'model_plast': list(config['simulations']['stim_on_MFs']['connection_models'][ct]['synapse'].keys())[1],\
		'ltd': DEFAULT_PLAST-0.1, 'ltp': DEFAULT_PLAST+0.01,'teaching': config['simulations']['stim_on_MFs']['connection_models'][ct]['teaching']})
		param_connections[ct]['pos'].update({'plastic': True, 'hetero': True, 'model_plast': list(config['simulations']['stim_on_MFs']['connection_models'][ct]['synapse'].keys())[1],\
		'ltd': DEFAULT_PLAST-0.1, 'ltp': DEFAULT_PLAST+0.01,'teaching': config['simulations']['stim_on_MFs']['connection_models'][ct]['teaching']})
	elif ct == 'parallel_fiber_to_basket' or ct == 'parallel_fiber_to_stellate':
		param_connections[ct].update({'plastic': True, 'hetero': True, 'model_plast': list(config['simulations']['stim_on_MFs']['connection_models'][ct]['synapse'].keys())[1],\
		'ltd': DEFAULT_PLAST-0.01, 'ltp': DEFAULT_PLAST+0.1,'teaching': config['simulations']['stim_on_MFs']['connection_models'][ct]['teaching']})
	
	# non-plastic connections
	else:
		if type(param_connections[ct]) is dict:
			param_connections[ct]['neg'].update({'plastic': False, 'hetero': None, 'model_plast': None,\
			'ltd': None, 'ltp': None,'teaching': None})
			param_connections[ct]['pos'].update({'plastic': False, 'hetero': None, 'model_plast': None,\
			'ltd': None, 'ltp': None,'teaching': None})
		else:
			param_connections[ct].update({'plastic': False, 'hetero': None, 'model_plast': None,\
			'ltd': None, 'ltp': None,'teaching': None})

sd_iomli = 10.0				# IO-MLI weights set as normal distribution to reproduce the effect of spillover-based transmission
min_iomli = 40.0

"""
STIMULI, RECORDING and SIMULATION
"""

# EBC protocol
ratio = 50		
ntrial = 20*ratio
onlyCS_ntrial = 30	
probe_ntrial = 100
td = 1260.0		# [ms]
TOT_DURATION = (ntrial)*td  # [ms]
num_blocks = 10

# Within each trial:
param_stimuli = {\
'CS': {'model_stim':'poisson_generator', 'start': 500.,'end': 760., 'freq': 10.},\
'US': {'model_stim': 'spike_generator', 'start': 750.,'end': 760., 'freq': 500.},\
'background': {'model_stim':'poisson_generator', 'start': 0.,'end': TOT_DURATION, 'freq': 4.},\
'learned_cosp': {'model_stim': 'spike_generator', 'start': 588.,'end': 598., 'freq': 400.}}

# Recording
sample = {'granule_cell': 0.01,'golgi_cell':1,'glomerulus':0.1,'mossy_fibers':1,'purkinje_cell':1, 'stellate_cell':1, 'basket_cell': 0.1,'dcn_cell_glut_large':1,'dcn_cell_GABA':1,'dcn_cell_Gly-I':1,'io_cell':1}
