# -*- coding: utf-8 -*-
"""
Created on Sept 2019

@author: Alice Geminiani - alice.geminiani@unipv.it
olivocerebellar spiking neural network with E-GLIF neurons and PLASTICITY, for classical eyeblink conditioning simulations
Publication: Geminiani et al., biorXiv, 2022 - doi: https://doi.org/10.1101/2023.06.20.545667

MAIN SIMULATION file to be used for running simulations
"""

def simulate_network(plasticity_parameters, path_saving_files):
    import time
    import numpy as np
    import h5py
    import nest
    import random
    from utils import sim_functions
    import pickle
    


    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Initializing kernel for simulation...")

    ''' VARIABLES INITIALIZATION '''
    SIM_RES = 1.0      
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({
                          'resolution': SIM_RES,
                          'overwrite_files': True,
    					  "data_path": path_saving_files})

    try:
        nest.Install("cerebmodule")
    except Exception as e:
        print("cerebmodule already installed")



    NEU_MOD = 'eglif_cond_alpha_multisyn'  # Neuron model name

    # Control variables
    WEIGHT_RECORDING = True            # To record weights with weight_recorder device. ATT: may cause memory issues!
    RECORD_VM = False
    SAVE_LABELLED = True


    start_netw = time.time()
    print(start_netw)


    # Set plasticity parameters imported from function (plasticity_parameters) to the conn_param
    sim_param.param_connections['parallel_fiber_to_purkinje']['ltp'] = plasticity_parameters[0]
    sim_param.param_connections['parallel_fiber_to_purkinje']['ltd'] = -plasticity_parameters[1]
    sim_param.param_connections['parallel_fiber_to_basket']['ltp'] = plasticity_parameters[2]
    sim_param.param_connections['parallel_fiber_to_basket']['ltd'] = -plasticity_parameters[3]
    sim_param.param_connections['parallel_fiber_to_stellate']['ltp'] = plasticity_parameters[2]
    sim_param.param_connections['parallel_fiber_to_stellate']['ltd'] = -plasticity_parameters[3]


    """
    CEREBELLAR NEURONS
    """

    f = h5py.File(sim_param.file_network, 'r')

    # Load ids of labelled cells

    with open('labelled_ids.pkl', 'rb') as f_label:  # Python 3: open(..., 'rb')
        pos, neg = pickle.load(f_label)


    # Create a dictionary with all cell names (keys)
    # and lists that will contain nest models (values)
    neuron_models = {key: [] for key in sim_param.cell_types}
    pos_neuron_models = {key: [] for key in pos.keys()}
    neg_neuron_models = {key: [] for key in neg.keys()}

    start_scaffold_id = {}          # To be used for conn matrix mapping between scaffold and NEST
    for cell_name in neuron_models.keys(): #cell_id in sorted_nrn_types:
        if rank==0:
            print(cell_name)
        path_ids = 'cells/placement/'+cell_name+'/identifiers'
        cell_ids = np.array(f[path_ids])
        start_scaffold_id[cell_name] = cell_ids[0]
        if cell_name != 'glomerulus' and cell_name != 'mossy_fibers':
            if cell_name in pos.keys():
                neuron_models[cell_name] = sim_functions.create_neural_pop(cell_name, len(pos[cell_name]), sim_param.param_neurons[cell_name]['pos'],NEU_MOD,sim_param.Erev_exc,sim_param.Erev_inh)
                neuron_models[cell_name].extend(sim_functions.create_neural_pop(cell_name, len(neg[cell_name]), sim_param.param_neurons[cell_name]['neg'],NEU_MOD,sim_param.Erev_exc,sim_param.Erev_inh))
                sim_functions.random_init_vm(neuron_models[cell_name],cell_name,sim_param.param_neurons[cell_name])
            else:
                neuron_models[cell_name] = sim_functions.create_neural_pop(cell_name, cell_ids, sim_param.param_neurons[cell_name],NEU_MOD,sim_param.Erev_exc,sim_param.Erev_inh)
                sim_functions.random_init_vm(neuron_models[cell_name],cell_name,sim_param.param_neurons[cell_name])        # Random initialization of Vm
        # Check if neuron is labelled and save corresponding ids
        if cell_name in pos.keys():
            pos_neuron_models[cell_name] = pos[cell_name] - start_scaffold_id[cell_name] + neuron_models[cell_name][0]
            neg_neuron_models[cell_name] = neg[cell_name] - start_scaffold_id[cell_name] + neuron_models[cell_name][0]
        else:
            if cell_name not in nest.Models():
                nest.CopyModel('parrot_neuron', cell_name)				# glomeruli and mossy fibers are parrot neurons
                neuron_models[cell_name] = nest.Create(cell_name, cell_ids[1])

   
    """
    CEREBELLAR CONNECTIVITY
    """
    # Weight recorders and volume transmitters (for heterosynaptic plasticity)
    weight_rec = {}
    vt = {}


    ####################### Creation of connections #######################
    ## Att: if the plastic connections are not put at the beginning of connection definition, the vt assignment on pf-pc connections could take also
    # aa-pc connections (if we don't specify GetConnections of stpd_connection_sinexp type), giving an error on vt parameter type!
    if rank==0:
        print(sim_param.connection_types)
    plastic_conn = []           # List of active plastic connections
    connections = sim_param.connection_types
    # Order list of connections to have the potential plastic ones at the beginning
    connections.insert(0, connections.pop(connections.index('parallel_fiber_to_purkinje')))
    connections.insert(1, connections.pop(connections.index('parallel_fiber_to_basket')))
    connections.insert(2, connections.pop(connections.index('parallel_fiber_to_stellate')))
    connections.insert(3, connections.pop(connections.index('mossy_to_dcn_glut_large')))
    connections.insert(4, connections.pop(connections.index('purkinje_to_dcn_glut_large')))
    for conn in connections:
        if rank==0:
            print("connecting...",conn)
        conn_matrix = np.array(f['cells/connections/'+conn])
        # Mapping to NEST ids
        pre_neuron = sim_param.param_connections[conn]['sender']
        post_neuron = sim_param.param_connections[conn]['receiver']
        if post_neuron in pos.keys():
            # Z+ conns
            conn_matrix_pos = conn_matrix[np.isin(conn_matrix[:, 1], pos), :]
            conn_matrix_pos[:,0] = conn_matrix_pos[:,0] - start_scaffold_id[pre_neuron] + neuron_models[pre_neuron][0]
            conn_matrix_pos[:,1] = conn_matrix_pos[:,1] - start_scaffold_id[post_neuron] + neuron_models[post_neuron][0]
            conn_matrix_pos = conn_matrix_pos.astype(int)
            plastic_conn, weight_recorders = sim_functions.create_connections(conn, conn_matrix_pos, neuron_models, sim_param.param_connections[conn]['pos'], plastic_conn, weight_rec, vt, WEIGHT_RECORDING)
            # Z- conns
            conn_matrix_neg = conn_matrix[np.isin(conn_matrix[:, 1], neg), :]
            conn_matrix_neg[:,0] = conn_matrix_neg[:,0] - start_scaffold_id[pre_neuron] + neuron_models[pre_neuron][0] + len(pos[cell_name])
            conn_matrix_neg[:,1] = conn_matrix_neg[:,1] - start_scaffold_id[post_neuron] + neuron_models[post_neuron][0] + len(pos[cell_name])
            conn_matrix_neg = conn_matrix_neg.astype(int)
            plastic_conn, weight_recorders = sim_functions.create_connections(conn, conn_matrix_neg, neuron_models, sim_param.param_connections[conn]['neg'], plastic_conn, weight_rec, vt, WEIGHT_RECORDING)
        else:
            conn_matrix[:,0] = conn_matrix[:,0] - start_scaffold_id[pre_neuron] + neuron_models[pre_neuron][0]
            conn_matrix[:,1] = conn_matrix[:,1] - start_scaffold_id[post_neuron] + neuron_models[post_neuron][0]
            conn_matrix = conn_matrix.astype(int)
            plastic_conn, weight_recorders = sim_functions.create_connections(conn, conn_matrix, neuron_models, sim_param.param_connections[conn], plastic_conn, weight_rec, vt, WEIGHT_RECORDING)

    end_netw = time.time()  # end of network creation (placement and connectome)
    print("netw_time: ", end_netw - start_netw)


    """
    STIMULI, RECORDING and SIMULATION
    """
    mf_num = len(neuron_models['mossy_fibers'])
    io_num = len(neuron_models['io_cell'])

    # Background noise to all mossy fibers
    background = nest.Create('poisson_generator',params={'rate':sim_param.param_stimuli['background']['freq'], 'start': sim_param.param_stimuli['background']['start'], 'stop': sim_param.param_stimuli['background']['end']})		# Rancz
    nest.Connect(background,neuron_models['mossy_fibers'])			# Input to parrot neurons


    # CS (Conditioned Stimulus) as spike generator to avoid different spike patterns in each trial
    spike_nums_CS = np.int(np.round((sim_param.param_stimuli['CS']['freq'] * (sim_param.param_stimuli['CS']['end'] - sim_param.param_stimuli['CS']['start'])) / 1000.))         # Rancz
    CS_matrix_start = np.random.uniform(sim_param.param_stimuli['CS']['start'],sim_param.param_stimuli['CS']['end'],[mf_num, spike_nums_CS])

    CS_matrix = CS_matrix_start

    for t in range(1,sim_param.ntrial+sim_param.probe_ntrial):
        CS_matrix = np.concatenate((CS_matrix,CS_matrix_start+t*sim_param.td),axis=1)
     
    CS = []

    for gn in range(mf_num):
        spk_gen = nest.Create('spike_generator', params = {'spike_times': np.sort(np.round(CS_matrix[gn,:]))})
        CS.append(spk_gen[0])

    # Localized CS to avoid border effects
    r_x, r_z = 100, 50
    gloms_pos = np.array(f['cells/placement/glomerulus/positions'])
    x_c, z_c = 150., 100.

    # Find glomeruli falling into the selected volume
    target_gloms_bool = np.add(((gloms_pos[:,[0]] - x_c)**2)/r_x**2,((gloms_pos[:,[2]] - z_c)**2)/r_z**2).__lt__(1)              # ellipse equation
    target_gloms_id_scaffold = np.array(np.where(target_gloms_bool)[0] + start_scaffold_id['glomerulus'])

    # Select the corrisponding original MFs
    conn_glom_mf = np.array(f['cells/connections/mossy_to_glomerulus'])
    target_mfs_id_scaffold = conn_glom_mf[np.isin(conn_glom_mf[:, 1],target_gloms_id_scaffold), 0]
    # translate to NEST ids
    target_mfs_id_nest = target_mfs_id_scaffold - start_scaffold_id['mossy_fibers'] + neuron_models['mossy_fibers'][0]
    target_mfs_id_nest = target_mfs_id_nest.astype(int)
    print(len(target_mfs_id_nest)," mfs stimulated")
    # Obtain an ordered list of non-duplicates
    id_stim = sorted(list(set(target_mfs_id_nest)))
    n = len(id_stim)
    print(n, " stimulated mfs")
    nest.Connect(list(CS[:n]), id_stim, {'rule': 'one_to_one'})


    # US (Unconditioned Stimulus) not as Poisson to avoid that some IO do not fire:
    spike_nums = np.int(np.round((sim_param.param_stimuli['US']['freq'] * (sim_param.param_stimuli['US']['end'] - sim_param.param_stimuli['US']['start'])) / 1000.))
    US_array = []
    for t in range(sim_param.ntrial):
        US_array.extend(np.round(np.linspace(t*sim_param.td+sim_param.param_stimuli['US']['start'], t*sim_param.td+sim_param.param_stimuli['US']['end']-2.0, spike_nums)))			# US_array = np.random.sample(range(US_START, US_END), spike_nums) 		#
    
    # Generate array of CS-US trials in the probe session
    probe_trials = np.sort(random.sample(list(range(sim_param.ntrial,sim_param.ntrial+sim_param.probe_ntrial)), k=(sim_param.probe_ntrial-sim_param.onlyCS_ntrial)))
    print("probe_trials: ", probe_trials)
    with open('./spikes/probe_trials.pkl', "wb") as f_probe:
        pickle.dump(probe_trials, f_probe)
    for pt in probe_trials:
        US_array.extend(np.round(np.linspace(pt*sim_param.td+sim_param.param_stimuli['US']['start'], pt*sim_param.td+sim_param.param_stimuli['US']['end']-2.0, spike_nums)))			# US_array = np.random.sample(range(US_START, US_END), spike_nums) 		#

    US = ()
    for ii in range(int(io_num/2)):
        US_new = nest.Create('spike_generator')
        nest.SetStatus(US_new, {'spike_times': US_array})
        US = US + US_new

    # US to 1st microcomplex IO neurons
    syn_param = {"model": "static_synapse", "weight":55.0, "delay": SIM_RES,"receptor_type":1}

    nest.Connect(US,neuron_models['io_cell'][:int(io_num/2)],{'rule':'one_to_one'},syn_param)


    # PC CoSp with linearly-increasing probability (up to 43%) to occurr as learning proceeds, with latency 88 ms (CIO)
    spike_nums = np.int(np.round((sim_param.param_stimuli['learned_cosp']['freq'] * (sim_param.param_stimuli['learned_cosp']['end'] - sim_param.param_stimuli['learned_cosp']['start'])) / 1000.))
    cosp_array = []
    for t in range(sim_param.ntrial+sim_param.probe_ntrial):
        # Probability check (linearly increasing with the number of trials)
        if random.random() < (t/sim_param.ntrial)*0.43:
            cosp_array.extend(np.round(np.linspace(t*sim_param.td+sim_param.param_stimuli['learned_cosp']['start'], t*sim_param.td+sim_param.param_stimuli['learned_cosp']['end']-2.0, spike_nums)))				#

    print(cosp_array)
    cosp = ()
    for ii in range(int(io_num/4)):
        cosp_new = nest.Create('spike_generator')
        nest.SetStatus(cosp_new, {'spike_times': cosp_array})
        cosp = cosp + cosp_new

    syn_param = {"model": "static_synapse", "weight":55.0, "delay": SIM_RES,"receptor_type":1}

    nest.Connect(cosp,neuron_models['io_cell'][:int(io_num/4)],{'rule':'one_to_one'},syn_param)



    ################################################# RECORDING ###################################################################
    if rank==0:
        print('Create recording devices')


    spikes = {}
    if SAVE_LABELLED:
        pos_spikes = {}
        neg_spikes = {}
    vm = {}
    ## Record spikes from all cells
    for cell_name in neuron_models.keys():
        spikes[cell_name] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "to_file": True, "label": cell_name+"_spikes"})
        nest.Connect(random.sample(neuron_models[cell_name],int(round(sim_param.sample[cell_name]*len(neuron_models[cell_name])))), spikes[cell_name])

        # Save ALL labelled neuron spikes in different spike recorders
        if SAVE_LABELLED and cell_name in pos.keys():
            pos_spikes[cell_name] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "to_file": True, "label": "pos_"+cell_name+"_spikes"})
            nest.Connect(tuple(pos_neuron_models[cell_name]), pos_spikes[cell_name])
            neg_spikes[cell_name] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "to_file": True, "label": "neg_"+cell_name+"_spikes"})
            nest.Connect(tuple(neg_neuron_models[cell_name]), neg_spikes[cell_name])
            print("Connecting labelled spike detectors")

        if RECORD_VM and cell_name!="glomerulus" and cell_name!="mossy_fibers":
            print("vm ", cell_name)
            vm[cell_name] = nest.Create("multimeter", params={'withtime': True, 'interval': 0.1,'record_from': ['V_m','I_adap','I_dep','G1','G2','G3'], 'to_file': True, 'label': cell_name+'_vm'})
            nest.Connect(vm[cell_name],random.sample(neuron_models[cell_name],int(0.1*round(sim_param.sample[cell_name]*len(neuron_models[cell_name])))))
    if rank==0:
        print("Plastic conn: ",plastic_conn)


################################################# SIMULATION  ###################################################################
    start_sim = time.time()
    print('Simulation start')
    if WEIGHT_RECORDING:            
        sim_subsampled_dur = sim_param.TOT_DURATION/sim_param.num_blocks
        for k in range(sim_param.num_blocks):
            for plastic in plastic_conn:
                nest.SetStatus(weight_recorders[plastic],{'start': k * sim_subsampled_dur,'stop': k * sim_subsampled_dur+sim_param.td, "label": plastic+'_block_'+str(k)})
            nest.Simulate(sim_subsampled_dur)   
    else:
        nest.Simulate(sim_param.TOT_DURATION)
    end_sim = time.time()
    print('Simulation finished')
    print('Sim time: ',end_sim-start_sim)

    spikes['mli'] = spikes['basket_cell']+spikes['stellate_cell']

    # Extract spike matrices from spike detectors: first column neuron ID, second column spike time instant
    spike_mat = {}
    for cell_type in spikes.keys():
        dSD = nest.GetStatus(spikes[cell_type],keys="events")[0]
        evs = dSD["senders"]
        ts = dSD["times"]
        spike_mat[str(cell_type)] = np.array([evs,ts])

    return spike_mat['purkinje_cell'], spike_mat['mli'], spike_mat['dcn_cell_glut_large']


if __name__ == '__main__':
    import sys
    from utils import sim_param


    # Plasticity learning rule parameters
    
    LTP_PFPC =  0.01/sim_param.ratio
    LTD_PFPC = 1.0/sim_param.ratio
    LTP_PFMLI = 0.01/sim_param.ratio
    LTD_PFMLI = 0.001/sim_param.ratio


    plasticity_parameters = [LTP_PFPC, LTD_PFPC, LTP_PFMLI, LTD_PFMLI]

    print("Initial plasticity parameters: ",plasticity_parameters)

    simulate_network(plasticity_parameters, "/spikes")
