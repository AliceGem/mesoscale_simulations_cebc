"""
Created on Sept 2019

@author: Alice Geminiani - alice.geminiani@unipv.it
olivocerebellar spiking neural network with E-GLIF neurons and PLASTICITY, for classical eyeblink conditioning simulations
Publication: Geminiani et al., biorXiv, 2022 - doi: https://doi.org/10.1101/2023.06.20.545667

NEST CEBC simulation functions
"""


# Function to create neural populations
def create_neural_pop(name, ids, neu_param, model, Erev_exc, Erev_inh):
    """
    Create neural populations
    Input arguments:
    # name = name of the neural population (e.g. golgi)
    # ids = neural population ids extracted from scaffold hdf5 filename
    # neu_param = neural population parameters (e.g. Cm etc)
    # model = NEST neuron model
    # Erev_exc and Erev_inh = reversal potentials of exc and inh synapses
    """
    import nest

    if name not in nest.Models():
        nest.CopyModel(model, name)
        nest.SetDefaults(name, {'t_ref': neu_param['t_ref'],  # ms
                                'C_m': neu_param['C_m'],  # pF
                                'E_L': neu_param['E_L'],  # mV
                                'V_reset': neu_param['V_reset']})  # mV
        if model == 'eglif_cond_alpha_multisyn':
            nest.SetDefaults(name, {'tau_m': neu_param['tau_m'],  # ms
                                    'V_th': neu_param['V_th'],  # mV
                                    'Vinit': neu_param['Vinit'],  # mV
                                    'lambda_0' : neu_param['lambda_0'],
                                    'tau_V' : neu_param['tau_V'],
                                    'I_e': neu_param['I_e'],  # pA # tonic ~9-10 Hz  ;previous = 36.0 pA
                                    'kadap' :  neu_param['kadap'],
                                    'k1' : neu_param['k1'],
                                    'k2' : neu_param['k2'],
                                    'A1' : neu_param['A1'],
                                    'A2' : neu_param['A2'],
                                    'tau_syn1': neu_param['tau_syn1'],		# glom-goc (mf-goc)
                                    'tau_syn2': neu_param['tau_syn2'],		# goc-goc
                                    'E_rev1': Erev_exc,
                                    'E_rev2' : Erev_inh})
            if 'Vmin' in neu_param.keys():
                nest.SetDefaults(name, {'Vmin': neu_param['Vmin']})
            if 'tau_syn3' in neu_param.keys():
                nest.SetDefaults(name, {'tau_syn3': neu_param['tau_syn3'], 'E_rev3': Erev_exc})			# pf-goc (grc-goc)
        else:                               # iaf_cond_alpha
            nest.SetDefaults(name, {'g_L': neu_param['C_m'] / neu_param['tau_m'],
                                    'V_th': neu_param['V_th'],  # mV
									'I_e': neu_param['Ie_iaf'],  # pA
									'tau_syn_ex': neu_param['tau_syn1'],  # glom-goc (mf-goc)
									'tau_syn_in': neu_param['tau_syn2'],  # goc-goc
									'E_ex': Erev_exc,
									'E_in': Erev_inh})
    neural_pop_ids = nest.Create(name, ids[1])

    return neural_pop_ids


# Function for random initialization of Vm (between EL-Vreset and EL+half; being half the half of the range between EL and Vth)
def random_init_vm(neural_pop_ids, name, neu_param):
    #print("imports in random_init_vm")
    import nest
    import random
    #print("imported in random_init_vm")

    for x in range(1,len(neural_pop_ids),2):
    	nest.SetStatus(neural_pop_ids[x-1:x],{'V_m':neu_param['E_L']+random.randint(neu_param['V_reset']-neu_param['E_L'],int((neu_param['V_th']-neu_param['E_L'])/2))})


# Function to create CONNECTIONS
def create_connections(name, matrix, neurons, conn_param, plastic_conn, weight_rec, vt, WR):
    #print("imports in create_connections")
    import nest
    #import sim_param
    import numpy as np
    #print("imported in create_connections")

    conn_dict = {'rule': 'one_to_one'}
    pre = matrix[:,0]
    post = matrix[:,1]

    if conn_param['plastic']==True:                # PLASTIC CONNECTIONS
        if WR:
            # Weight recorders
            weight_rec[name] = nest.Create('weight_recorder', params={"to_memory": False,
                                                                            "to_file": True,
                                                                            "senders": neurons[conn_param['sender']],
                                                                            "targets": neurons[conn_param['receiver']]})
        else:
            weight_rec[name] = None
        plastic_conn.append(name)
        name_plast = 'plast'+name
        print('checked plastic')
        if conn_param['hetero']==True:                # heterosynaptic plasticity
            print('checked heterosyn')

            # Volume transmitter
            vt[conn_param['receiver']] = nest.Create("volume_transmitter_alberto",len(neurons[conn_param['receiver']]))
            print("Created vt: ",conn_param['receiver'])
            for n,vti in enumerate(vt[conn_param['receiver']]):
                nest.SetStatus([vti],{"deliver_interval" : 2})  # TO CHECK
                nest.SetStatus([vti],{"vt_num" : n})

            nest.CopyModel(conn_param['model_plast'],name_plast)
            nest.SetDefaults(name_plast,{"A_minus":   conn_param['ltd'],   # double - Amplitude of weight change for depression
        								 "A_plus":    conn_param['ltp'],   # double - Amplitude of weight change for facilitation
        								 "Wmin":      0.0,    # double - Minimal synaptic weight
        								 "Wmax":      10.0,     # double - Maximal synaptic weight
        								 "vt": vt[conn_param['receiver']][0]})
            if WR:
                nest.SetDefaults(name_plast,{"weight_recorder": weight_rec[name][0]})
            syn_param = {"model": name_plast, "weight": abs(conn_param['weight']), "delay": conn_param['delay'], "receptor_type":conn_param['receptor']}


            for vt_num, post_cell in enumerate(np.unique(post)):
                syn_param["vt_num"] = float(vt_num)
                indexes = np.where(post == post_cell)[0]
                pre_neurons = np.array(pre)[indexes]
                post_neurons = np.array(post)[indexes]
                nest.Connect(list(pre_neurons),list(post_neurons),conn_dict, syn_param)

        else:           # homosynaptic plasticity
            print('checked homosyn')
            nest.SetDefaults(conn_param['model_plast'],{"tau_plus": 30.0,
        									            "lambda": conn_param['ltp'],
        									            "alpha": conn_param['ltd']/conn_param['ltp'],
        									            "mu_plus": 0.0,  # Additive STDP
        									            "mu_minus": 0.0, # Additive STDP
        									            "Wmax": 4000.0})
            if WR:
                nest.SetDefaults(conn_param['model_plast'],{"weight_recorder": weight_rec[name][0]})

            syn_param = {"model": conn_param['model_plast'], "weight": abs(conn_param['weight']), "delay": conn_param['delay'], "receptor_type":conn_param['receptor']}
            nest.Connect(list(pre), list(post), conn_dict, syn_param)

    else:           # STATIC CONNECTIONS
        if name == 'io_to_basket' or name == 'io_to_stellate':              # Spillover-mediated synapses
            syn_param = {"model": "static_synapse", "weight": abs(conn_param['weight']),
            "delay": conn_param['delay'],"receptor_type":conn_param['receptor']}
        elif name == "mossy_to_glomerulus":
            syn_param = {"model": "static_synapse", "weight": abs(conn_param['weight']), "delay": conn_param['delay']}
        elif name == 'dcn_GABA_to_io':
            syn_param = {"model": "static_synapse", "weight": conn_param['weight'], "delay": conn_param['delay'], "receptor_type":conn_param['receptor']}
        else:
            syn_param = {"model": "static_synapse", "weight": abs(conn_param['weight']), "delay": conn_param['delay'], "receptor_type":conn_param['receptor']}
        nest.Connect(list(pre), list(post), conn_dict, syn_param)

    for plast in plastic_conn:                            # If the connection is also a teaching connection, the volume transmitter should be connected
        if conn_param[plast]['teaching'] == name:
            post = [x - neurons[conn_param[plast]['receiver']][0] + vt[conn_param[plast]['receiver']][0] for x in post]
            nest.Connect(list(pre), list(post), conn_dict, {"model": "static_synapse", "weight": 0.0, "delay": 1.0})


    return plastic_conn, weight_rec

def write_weights(connection,file_weight):
    #print("imports in write_weights")
    import nest
    #print("imported in write_weights")

    weights = nest.GetStatus(connection, 'weight')
    for w in weights:
        file_weight.write(str(w)+" ")
    file_weight.write("\n")
