"""
Created on Sept 2019

@author: Alice Geminiani - alice.geminiani@unipv.it
ANALYSIS of classical eyeblink conditioning simulations
Publication: Geminiani et al., biorXiv, 2022 - doi: https://doi.org/10.1101/2023.06.20.545667

"""


from cmath import nan
from doctest import OutputChecker
from _plotly_utils.utils import split_multichar
from matplotlib.pyplot import figimage, xlabel, ylabel
import numpy as np
import sim_param
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import glob
import h5py
import pickle
import math
from os.path import exists


data_path = "spikes/"



SAVING_DATA = True
saving_filename = "rates_info.txt"

filename_sdf = "sdf.pickle"
filename_norm_sdf = "norm_division_sdf.pickle"    
filename_neg_sdf = "neg_sdf.pickle"
filename_neg_norm_sdf = "neg_norm_division_sdf.pickle"  
filename_baseline = "baseline.pickle"
filename_ss = "simple_spikes.pickle"
filename_cs = "complex_spikes.pickle"
filename_ss_neg = "neg_simple_spikes.pickle"
filename_cs_neg = "neg_complex_spikes.pickle"
filename_motor_output = "motor_output.pickle"
filename_cr = "cr.pickle"

cell_to_plot = [
    "mossy_fibers",
    "granule_cell",
    "golgi_cell",
    "basket_cell",
    "stellate_cell",
    "purkinje_cell",
    "dcn_cell_glut_large",
    "io_cell",
]

cell_baselines = [
    "mossy_fibers",
    "glomerulus",
    "granule_cell",
    "golgi_cell",
    "stellate_cell",
    "basket_cell",
    "purkinje_cell",
    "dcn_cell_glut_large",
    "dcn_cell_GABA",
    "io_cell",
    "mli",
]


window_filter = 180
cutoff = 50  # cutoff [ms] for plots and SDF/motor output baseline computations
threshold_cr = 0.5  # Threshold to monitor if there is a CR or not
nblocks = 10

# For sdf change division
baseline_window=[100, 500]
change_window=[550, 750]
baseline_trial = 0
            
# SDF plot info
cell_sdf = [
    "mli",
    "purkinje_cell-ss",
    "purkinje_cell-cs",
    "dcn_cell_glut_large",
    "io_cell",
]  # Cells for which we compute the sdf

rc = {
    "mli": [1, 1],
    "purkinje_cell-ss": [2, 1],
    "purkinje_cell-cs": [3, 1],
    "dcn_cell_glut_large": [4, 1],
    "io_cell": [5, 1],
}  # row-column position
gw = {
    "purkinje_cell-ss": 41,
    "mli": 41,
    "dcn_cell_glut_large": 10,     #41,     # 41  only for DCN smoothed SDF
    "io_cell": 5,
    "purkinje_cell-cs": 5
}  # Gaussian window of the SDF kernel
subtract_baseline = True  # Activate to plot SDF normalized wrt baseline
threshold_change = {
    "purkinje_cell-ss": -5,
    "mli": 200,
    "dcn_cell_glut_large": 45,
    "io_cell": 0.1,
    "purkinje_cell-cs": 1,
}

trial_cut = 250  #  [ms] to cut before CS start and after US start to refer plot to CS start as the 0 time point
yranges_all_trials = {
    "mli": [0.7, 1.3], 
    "purkinje_cell-ss": [0.75, 1.4], 
    "purkinje_cell-cs": [0.75, 1.4],  
    "dcn_cell_glut_large": [0.75, 1.4], 
    "io_cell": [0.75, 1.4],
}
yranges_last_trial = {
    "mli": [0.75, 1.5],
    "purkinje_cell-ss": [0.75, 1.5], 
    "purkinje_cell-cs": [0.75, 1.5], 
    "dcn_cell_glut_large": [0.5, 1.5], 
    "io_cell":[0.75, 1.5], 
}
step_plot_sdf = 2

# Raster and PSTH plot info
raster_add_cell = []  # "granule_cell"       # Cell to be added only to raster
to_subsample = [
    "mossy_fibers",
    "granule_cell",
    "golgi_cell",
    "basket_cell",
    "stellate_cell",
    "purkinje_cell",
    "dcn_cell_glut_large"
]  
psth_yrange = {
    "mossy_fibers": [0, 60],
    "glomerulus": [0, 60],
    "golgi_cell": [0, 150],
    "granule_cell": [0, 40],
    "purkinje_cell": [0, 600],
    "stellate_cell": [0, 300],
    "basket_cell": [0, 300],
    "dcn_cell_glut_large": [0, 300],
    "dcn_cell_GABA": [0, 100],
    "dcn_cell_Gly-I": [0, 100],
    "io_cell": [0, 500],
}
trial_to_plot = [1, sim_param.ntrial]

bw = 5  # Histogram bin width
subsample_factor_gr = 0.05  # Subsample factor for GrC raster plot
subsample_factor = 0.5  # Subsample factor for other raster plot
ALL_SIM = True  # Take all simulations in PSTH plot
PLOT_LABELLED = True
trial_cut_spikes = 500  #  [ms] to cut before CS start and after US start to refer plot to CS start as the 0 time point

# From Gamma2 mutant experiments (wild type data)
mean_per_cr_exp = [9, 27, 52, 71, 85, 78, 83, 90, 84, 81]
std_per_cr_exp = [4, 7, 10, 9, 4, 7, 6, 3, 6, 6]

f = h5py.File(sim_param.file_network, "r")
with open("labelled_ids.pkl", "rb") as f_label:  # Python 3: open(..., 'rb')
    pos, neg = pickle.load(f_label)

# Loading functions
def load_spikes(cell, file_to_search):
    """
    load_spikes load .gdf files for the selected cell

    :param cell: name of the current cell
    :type cell: string

    :param file_to_search: name of the file to search for the current cell spikes
    :type file_to_search: string

    :return spks: list of spikes for the current cell type
    :return first_spikes_num: number of spikes for the current cell in the first simulation

    """
    files = glob.glob(data_path + "/" + file_to_search)
    for f in range(len(files)):
        content = np.loadtxt(files[f], delimiter=" ", usecols=[0, 1])
        # Check empty file
        if len(content) == 0:
            content = np.zeros((1, 2))
        # Check if we are in first or following simulations
        
        if f == 0:  # First simulation
            spks_array = content
            if len(content.shape) > 1:
                first_spikes_num = len(
                    spks_array[:, 1]
                )  # number of spikes in the first simulation to select first simulation
            else:
                first_spikes_num = 0
        else:
            spks_array = np.append(spks_array, content, axis=0)

    if cell == "mossy_fibers":
        print("min max mfs ", min(spks_array[:, 0]), max(spks_array[:, 0]))
        global sim_num
        sim_num = len(files)

    return spks_array, first_spikes_num


# Processing functions
def extract_current_trial_spikes(spks, t=1):
    # Extract current trial spikes; spks is a 2D array derived from .gdf NEST files with first column neuron ID, second column spike time
    current_spks = spks[spks[:, 1] > (sim_param.td * t), :]
    current_spks = current_spks[current_spks[:, 1] < (sim_param.td * (t + 1)), :]
    return current_spks


def extract_current_neuron_spikes(spks, current_neuron):
    # Extract spike times of the selected neuron
    return spks[spks[:, 0] == current_neuron, 1]


def compute_spike_rate(spks, window=[0, 500]):
    # extracts the firing rate as number of spikes in time window [ms] provided, for each cell. Default window is the baseline [0, 500] of the first trial
    # It returns a list with the same length as the number of neurons
    cell_ids = np.unique(spks[:, 0])
    spike_rate = []
    spike_rate_isi = []
    for ci in cell_ids:
        current_spikes = extract_current_neuron_spikes(spks, ci)
        current_window_spikes = current_spikes[current_spikes > window[0]]
        current_window_spikes = current_window_spikes[current_window_spikes < window[1]]
        rate = len(current_window_spikes) / ((window[1] - window[0]) * 0.001)
        if len(current_window_spikes) == 0:
            rate_isi = 0
        else:
            rate_isi = np.mean(np.diff(current_window_spikes))
        spike_rate.append(rate)
        spike_rate_isi.append(rate_isi)
    return spike_rate, spike_rate_isi


def separate_complex_spikes(spks, cutoff_ratio_burst=2, cutoff_ratio_pause=0.1, baseline_window=[0, 500]):
    # From PC spikes, remove the complex spikes identified as bursts with ISI cutoff_ratio times lower the average baseline (computed in baseline_window)
    # The complex spikes are substituted with simple spikes at the same rate as the baseline
    cell_ids = np.unique(spks[:, 0])
    complex_spikes = np.empty([1, 2])
    simple_spikes = np.empty([1, 2])
    for ci in cell_ids:
        cs_avg_spikes = []
        ind_to_remove = []
        to_add = []
        current_spikes = extract_current_neuron_spikes(
            spks, ci
        )  # 1-D vector of spike times for the current cell ci
        # Select only first 10 trials
        baseline_spikes = current_spikes[current_spikes > baseline_window[0]]
        baseline_spikes = baseline_spikes[baseline_spikes < baseline_window[1]]
        baseline_isi_avg = np.mean(np.diff(baseline_spikes))
        current_isi = np.diff(current_spikes)
        # Find all indices of CS ISIs
        cs_isi_index = list(np.where(current_isi < (baseline_isi_avg / cutoff_ratio_burst)))[
            0
        ]
        if len(cs_isi_index) > 0:
            cs_timings = current_spikes[np.array(cs_isi_index)]
            # print("cs_timings", cs_timings)
            differential_cs = np.diff(cs_timings)
            # print("differential cs ", differential_cs)
            # Find first index of each CS
            cs_cut_index = list(
                np.where(differential_cs > (baseline_isi_avg / cutoff_ratio_burst))
            )[0]
            # print("cs_cut_index", cs_cut_index)
            # print([cs_timings[0],current_spikes[cs_isi_index[0]+1]])
            
            # First CS
            # Check for pause after the burst
            if len(cs_cut_index) > 0:
                # print("More than one CS!")
                # print("First starts at ", current_spikes[cs_isi_index[0]])
                # Check that we are in the proper window
                start_cs_in_trial = current_spikes[cs_isi_index[0]]%sim_param.td            # Start of the CoSp in the current trial
                # print("Start CS in trial of length ", sim_param.td, ": ", start_cs_in_trial)
                if (start_cs_in_trial > sim_param.param_stimuli['learned_cosp']['start'] and start_cs_in_trial < sim_param.param_stimuli['learned_cosp']['start']+20) or (start_cs_in_trial > sim_param.param_stimuli['US']['start'] and start_cs_in_trial < sim_param.param_stimuli['US']['start']+20): 
                    if (
                        current_spikes[cs_isi_index[cs_cut_index[0]] + 2]
                        - current_spikes[cs_isi_index[cs_cut_index[0]] + 1]
                    ) > baseline_isi_avg * cutoff_ratio_pause:
                        cs_avg_spikes.append(
                            np.mean(
                                np.append(
                                    cs_timings[0 + cs_cut_index[0]],
                                    current_spikes[cs_isi_index[cs_cut_index[0]] + 1],
                                )
                            )
                        )
                        ind_to_remove.extend(
                            list(
                                np.append(
                                    [0 + cs_cut_index[0]], cs_isi_index[cs_cut_index[0]] + 1
                                )
                            )
                        )
                        # Add simple spikes in the interval left by complex spikes
                        complex_spike_interval = (
                            current_spikes[cs_isi_index[cs_cut_index[0]] + 2]
                            - current_spikes[cs_isi_index[0] - 1]
                        )
                        # print("cosp start 1", current_spikes[cs_isi_index[0]-1], " end ", current_spikes[cs_isi_index[cs_cut_index[0]]+2])
                        num_to_add = int(complex_spike_interval / baseline_isi_avg)
                        to_add.extend(
                            list(
                                np.linspace(
                                    current_spikes[cs_isi_index[0] - 1],
                                    current_spikes[cs_isi_index[cs_cut_index[0]] + 2],
                                    num_to_add,
                                )
                            )
                        )
                        # print("to_remove", ind_to_remove)
                        # print("to add ", to_add)
                ind = 1
            else:  # There is only one complex spike
                # print("One CS!")
                start_cs_in_trial = cs_timings[0 + cs_cut_index[0]]%sim_param.td            # Start of the CoSp in the current trial  
                if (start_cs_in_trial > sim_param.param_stimuli['learned_cosp']['start'] and start_cs_in_trial < sim_param.param_stimuli['learned_cosp']['start']+20) or (start_cs_in_trial > sim_param.param_stimuli['US']['start'] and start_cs_in_trial < sim_param.param_stimuli['US']['start']+20): 
                    if (
                        current_spikes[cs_isi_index[-1] + 2]
                        - current_spikes[cs_isi_index[-1] + 1]
                    ) > baseline_isi_avg * cutoff_ratio_pause:
                        cs_avg_spikes.append(
                            np.mean(
                                np.append(cs_timings, current_spikes[cs_isi_index[-1] + 1])
                            )
                        )
                        ind_to_remove.extend(
                            list(np.append(cs_isi_index, cs_isi_index[-1] + 1))
                        )
                        # Add simple spikes in the interval left by complex spikes
                        complex_spike_interval = (
                            current_spikes[cs_isi_index[-1] + 2]
                            - cs_timings[0 + cs_cut_index[0]]
                        )
                        #    print("cosp start 2", cs_timings[0+cs_cut_index[0]], " end ", current_spikes[cs_isi_index[-1]+2])
                        num_to_add = int(complex_spike_interval / baseline_isi_avg)
                        to_add.extend(
                            list(
                                np.linspace(
                                    cs_timings[0 + cs_cut_index[0]],
                                    current_spikes[cs_isi_index[-1] + 2],
                                    num_to_add,
                                )
                            )
                        )
                    # print("to_remove", ind_to_remove)
                #    print("to add ", to_add)
                #  print("to_remove", ind_to_remove)

            # print("avg: ",cs_avg_spikes)
            # Intermediate CSs
            for cs in range(1, len(cs_cut_index)):
                # print("current cs", cs, "starting", current_spikes[cs_isi_index[cs_cut_index[cs - 1] + 1]])
                start_cs_in_trial = current_spikes[cs_isi_index[cs_cut_index[cs - 1] + 1]]%sim_param.td            # Start of the CoSp in the current trial  
                # print("Start CS in trial ",current_spikes[cs_isi_index[cs_cut_index[cs - 1] + 1]]/sim_param.td  , ": ", start_cs_in_trial)
                if (start_cs_in_trial > sim_param.param_stimuli['learned_cosp']['start'] and start_cs_in_trial < sim_param.param_stimuli['learned_cosp']['start']+20) or (start_cs_in_trial > sim_param.param_stimuli['US']['start'] and start_cs_in_trial < sim_param.param_stimuli['US']['start']+20):        
                    # print("It's in time window")    # Check for pause after the burst
                    if (
                        current_spikes[cs_isi_index[cs_cut_index[cs]] + 2]
                        - current_spikes[cs_isi_index[cs_cut_index[cs]] + 1]
                    ) > baseline_isi_avg * cutoff_ratio_pause:
                        # print("Verify pause")
                        # print("first spikes of CS ", current_spikes[cs_isi_index[cs_cut_index[cs-1]+1]:cs_isi_index[cs_cut_index[cs]]+1])
                        # print("cur spk ",current_spikes[cs_isi_index[cs_cut_index[cs]]+1])
                        first_spks_current_cs = current_spikes[
                            cs_isi_index[cs_cut_index[cs - 1] + 1] : cs_isi_index[
                                cs_cut_index[cs]
                            ]
                            + 1
                        ]

                        # print("array ", (np.append(first_spks_current_cs,current_spikes[cs_isi_index[cs_cut_index[cs]]+1])))
                        cs_avg_spikes.append(
                            np.mean(
                                np.append(
                                    first_spks_current_cs,
                                    current_spikes[cs_isi_index[cs_cut_index[cs]] + 1],
                                )
                            )
                        )
                        # print("cur avg ", np.mean(np.append(first_spks_current_cs,current_spikes[cs_isi_index[cs_cut_index[cs+1]]+1])))
                        ind_to_remove.extend(
                            list(
                                range(
                                    cs_isi_index[cs_cut_index[cs - 1] + 1],
                                    cs_isi_index[cs_cut_index[cs]] + 2,
                                )
                            )
                        )
                        # print("index remove ",cs+ind+1, cs_cut_index[ind+1]+1, cs_cut_index[ind+1])
                        # print("ind ", ind)
                        # print("current ind_to_remove intermediate", list(range(cs_isi_index[cs_cut_index[cs-1]+1], cs_isi_index[cs_cut_index[cs]]+2)))
                        # Add simple spikes in the interval left by complex spikes
                        complex_spike_interval = (
                            current_spikes[cs_isi_index[cs_cut_index[cs]] + 2]
                            - current_spikes[cs_isi_index[cs_cut_index[cs - 1] + 1] - 1]
                        )      # From spike before to spike after the burst
                        # print("cosp start 3", current_spikes[cs_isi_index[cs_cut_index[cs-1]+1]-1], " end ", current_spikes[cs_isi_index[cs_cut_index[cs]]+2])
                        num_to_add = round(complex_spike_interval / baseline_isi_avg)
                        # print("num_to_add",num_to_add, complex_spike_interval, baseline_isi_avg, current_spikes[cs_isi_index[cs_cut_index[cs-1]+1]-1], current_spikes[cs_isi_index[cs_cut_index[cs]]+2])
                        to_add.extend(
                            list(
                                np.linspace(
                                    current_spikes[
                                        cs_isi_index[cs_cut_index[cs - 1] + 1] - 1
                                    ],
                                    current_spikes[cs_isi_index[cs_cut_index[cs]] + 2],
                                    num_to_add,
                                )
                            )
                        )
                        # print("to_remove", ind_to_remove)
                        # print("to add ", to_add)
                    ind += 1
            # Last CS
            if len(cs_cut_index) > 1:
                # print("Last CS")
                if len(current_spikes) > cs_isi_index[-1] + 2:
                    start_cs_in_trial = current_spikes[cs_isi_index[cs_cut_index[-1] + 1] - 1]%sim_param.td            # Start of the CoSp in the current trial 
                    # print("Last CoSp start in trial: ", start_cs_in_trial) 
                    if (start_cs_in_trial > sim_param.param_stimuli['learned_cosp']['start'] and start_cs_in_trial < sim_param.param_stimuli['learned_cosp']['start']+20) or (start_cs_in_trial > sim_param.param_stimuli['US']['start'] and start_cs_in_trial < sim_param.param_stimuli['US']['start']+20):        
                        # print("detected burst cosp...")
                        if (
                            current_spikes[cs_isi_index[-1] + 2]
                            - current_spikes[cs_isi_index[-1] + 1]
                        ) > baseline_isi_avg * cutoff_ratio_pause:
                            # print("Checked also pause")
                            cs_avg_spikes.append(
                                np.mean(
                                    np.append(
                                        cs_timings[cs_cut_index[-1] + 1 :],
                                        current_spikes[cs_isi_index[-1] + 1],
                                    )
                                )
                            )
                            ind_to_remove.extend(
                                list(
                                    np.append(
                                        cs_isi_index[cs_cut_index[-1] + 1 :],
                                        cs_isi_index[-1] + 1,
                                    )
                                )
                            )
                            # Add simple spikes in the interval left by complex spikes
                            complex_spike_interval = (
                                current_spikes[cs_isi_index[-1] + 2]
                                - current_spikes[cs_isi_index[cs_cut_index[-1] + 1] - 1]
                            )
                            # print("cosp start 4", current_spikes[cs_isi_index[cs_cut_index[-1]+1]-1], " end ", current_spikes[cs_isi_index[-1]+2])
                            num_to_add = int(complex_spike_interval / baseline_isi_avg)
                            to_add.extend(
                                list(
                                    np.linspace(
                                        current_spikes[
                                            cs_isi_index[cs_cut_index[-1] + 1] - 1
                                        ],
                                        current_spikes[cs_isi_index[-1] + 2],
                                        num_to_add,
                                    )
                                )
                            )
                        # print("to_remove", ind_to_remove)
                        # print("to add ", to_add)
        #                 print("to remove last ",ind_to_remove)
        # print("avg: ",cs_avg_spikes, " for neuron ", ci)
        #  Add avg for last CS of current neuron
        current_complex_spikes = zip(np.repeat(ci, len(cs_avg_spikes)), cs_avg_spikes)
        current_complex_spikes = np.array(list(current_complex_spikes))
        # print("current_complex_spikes: ", current_complex_spikes, " for neuron ", ci)
        if len(current_complex_spikes > 0):
            print("cell ",ci, " has current_complex_spikes num", current_complex_spikes.shape)
            complex_spikes = np.array(complex_spikes)
            # print("shapes ",complex_spikes,current_complex_spikes.shape)
            complex_spikes = np.array(
                np.vstack((complex_spikes, current_complex_spikes))
            )
            # print(len(current_complex_spikes)," complex spikes for neuron ", ci, current_complex_spikes)
            # print("all complex spikes: ", complex_spikes)

        # print("to_remove final", ind_to_remove)
        # print("to add final", to_add)
        # print("current spikes around first spike to remove", current_spikes[ind_to_remove[0]-5:ind_to_remove[0]+5])
        current_spikes = np.delete(current_spikes, ind_to_remove)
        current_spikes = np.sort(np.append(current_spikes, to_add))
        current_simple_spikes = np.array(
            list(zip(np.repeat(ci, len(current_spikes)), current_spikes))
        )
        # print("current_simple_spikes ", current_simple_spikes)
        simple_spikes = np.array(simple_spikes)
        simple_spikes = np.array(np.vstack((simple_spikes, current_simple_spikes)))
        # print("simple spikes ", simple_spikes)
    # Remove first element that is initialization
    complex_spikes = np.delete(complex_spikes, 0, axis=0)
    simple_spikes = np.delete(simple_spikes, 0, axis=0)
    # print("final complex_spikes",complex_spikes)

    return simple_spikes, complex_spikes



def compute_sdf(spks, g_size):
    # Extract NEST ids of neurons that are firing
    neurons = np.unique(spks[:, 0])

    # Compute firing rate as spike density function, from spikes using convolution with a sliding Gaussian window.
    # Method based on Dyan Abbott book and [Ten Brinke et al., 2015]: "Spike density functions (SDFs) were computed for all trials by
    # convolving simple and complex spike occurrences across 1-ms bins with a
    # 41-ms Gaussian kernel (Figure S2)."
    # For each neuron neu, sdf (spike density function) contains the value of the firing rate in each time instant
    # t in each trial (dimension 3)
    sdf = np.empty([len(neurons), int(sim_param.td), sim_param.ntrial])
    normalized_sdf = np.empty([len(neurons), int(sim_param.td), sim_param.ntrial])
    for trial in range(sim_param.ntrial):
        # Extract current trial spikes
        current_spks = extract_current_trial_spikes(spks, t=trial)
        for neu in range(len(neurons)):
            # Spike times in the current trial, in the range 0 to td [ms]
            spike_times = (
                extract_current_neuron_spikes(current_spks, neurons[neu])
                - sim_param.td * trial
            )

            for t in range(int(sim_param.td)):
                tau = t - spike_times
                sdf[neu, t, trial] = sum(
                    1
                    / (math.sqrt(2 * math.pi) * g_size)
                    * np.exp(-np.power(tau, 2) / (2 * (g_size**2)))
                ) * (10**3)

           # normalized_sdf[neu, :, trial] = sdf[neu, :, trial] - np.mean(
           #     sdf[neu, cutoff : int(sim_param.param_stimuli["CS"]["start"]), trial]
           # )
            
            avg_baseline_sdf = np.mean(sdf[neu,baseline_window[0]:baseline_window[1], trial])
            if avg_baseline_sdf > 0.5:
                normalized_sdf[neu, :, trial] = sdf[neu,:, trial]/avg_baseline_sdf
            else:
                normalized_sdf[neu, :, trial] = np.nan*sdf[neu,:, trial]
                
    return sdf, normalized_sdf


def compute_sdf_change(
    sdf,
    baseline_trial,
    change_trial,
    baseline_window=[0, 500],
    change_window=[550, 750],
):
    # Compute the SDF change in the change_window with respect to avg baseline SDF for each cell, in the selected trial.
    # It returns the absolute change in Hz and the % change
    sdf_change = []
    per_sdf_change = []
    for neuron in range(sdf.shape[0]):
        avg_baseline_sdf = np.mean(
            sdf[neuron, baseline_window[0] : baseline_window[1], baseline_trial]
        )
        current_sdf_change = np.sum(
            sdf[neuron, int(change_window[0]) : int(change_window[1]), change_trial]
            / avg_baseline_sdf - 1,
            axis=None,
        ) / (change_window[1] - change_window[0])
        current_per_sdf_change = (current_sdf_change / avg_baseline_sdf) * 100
        sdf_change.extend([current_sdf_change])
        per_sdf_change.extend([current_per_sdf_change])
    return sdf_change, per_sdf_change


def compute_motor_output_delta(spk_pos, spk_neg, wf=20):
    # Compute the motor output of the cerebellum from the average DCN SDF of positive and negative cells for each trial, applying a moving average filter
    motor_output = []
    normalized_motor_output = []
    mean_noise = 0
    sdf_smooth_pos = compute_sdf(spk_pos, 25)[0]
    sdf_smooth_neg = compute_sdf(spk_neg, 25)[0]
    for trial in range(sim_param.ntrial):
        cumsum_pos = np.cumsum(np.mean(sdf_smooth_pos[:, :, trial], axis=0))
        cumsum_pos[wf:] = cumsum_pos[wf:] - cumsum_pos[:-wf]
        cumsum_neg = np.cumsum(np.mean(sdf_smooth_neg[:, :, trial], axis=0))
        cumsum_neg[wf:] = cumsum_neg[wf:] - cumsum_neg[:-wf]
        current_trial_motor_output = (cumsum_pos[wf - 1 :] / float(wf)) - (
            cumsum_neg[wf - 1 :] / float(wf)
        )
        # noise = np.random.normal(mean_noise, np.sqrt(0.5), len(current_trial_motor_output))
        # current_trial_motor_output += noise
        motor_output.append(current_trial_motor_output)

    # Normalize wrt current trial baseline and peak
    peak = 39.91  # np.amax(motor_output)
    print("Peak ", peak)
    for trial in range(sim_param.ntrial):
        baseline = np.mean(
            motor_output[trial][cutoff : int(sim_param.param_stimuli["CS"]["start"])]
        )
        # peak = np.amax(motor_output[trial])
        print("baseline", baseline)
        normalized_motor_output.append(
            list(((np.array(motor_output[trial])) - baseline) / (peak - baseline))
        )
        print(
            "After normalization: ",
            np.mean(
                normalized_motor_output[trial][
                    cutoff : int(sim_param.param_stimuli["CS"]["start"])
                ]
            ),
        )
    # # Normalize between 0 at baseline and 1 at peak computed in the first trial where the effect of plasticity is not there yet
    # baseline = np.mean(motor_output[0][cutoff:int(sim_param.param_stimuli['CS']['start'])])
    # peak = np.amax(motor_output)
    # normalized_motor_output = ((np.array(motor_output))-baseline)/(peak-baseline)
    print("peak: ", peak)
    return motor_output  # normalized_motor_output


def compute_max_eye_closure(motor_output):
    delta = []
    for trial in range(sim_param.ntrial):
        baseline = np.mean(
            motor_output[trial][cutoff : int(sim_param.param_stimuli["CS"]["start"])]
        )
        peak = np.amax(motor_output[trial])
        delta.append(peak - baseline)
    max_closure = np.amax(delta)

    return max_closure


def compute_per_cr(motor_output, input_cr_threshold=None):
    cr = []
    cr_timing = []
    cr_peak_timing = []
    normalized_motor_output = []
    max_amplitude =  compute_max_eye_closure(motor_output)      
    for trial in range(sim_param.ntrial):
        if input_cr_threshold is None:
            # The threshold is computed in each trial as mean+10SD of the baseline in the current trial

            baseline = np.mean(
                motor_output[trial][
                    cutoff : int(sim_param.param_stimuli["CS"]["start"])
                ]
            )
            threshold = (
                baseline
                + 0.05 * baseline
                + 2.5
                * np.std(
                    motor_output[trial][
                        cutoff : int(sim_param.param_stimuli["CS"]["start"])
                    ]
                )
            )  
            norm_threshold = (threshold - baseline) / max_amplitude 
            normalized_motor_output.append(
                list(((np.array(motor_output[trial])) - baseline) / max_amplitude)
            )
            cr_threshold = max(0.2, norm_threshold)
        else:
            cr_threshold = input_cr_threshold

        above_threshold = np.where(
            np.array(normalized_motor_output[trial]) > cr_threshold
        )
        if len(above_threshold[0]) > 0:
            threshold_crossing = min(above_threshold[0])
        else:
            threshold_crossing = 0.0
        print(
             "In trial ",
             trial,
             " crossing threshold ",
             threshold,
             " normalized ",
             cr_threshold,
             "at ",
             threshold_crossing,
        )
        if threshold_crossing > (
            sim_param.param_stimuli["CS"]["start"] - window_filter + cutoff
        ) and threshold_crossing < (
            sim_param.param_stimuli["US"]["start"] - window_filter - 10
        ):
            cr.append(True)
            cr_timing.append(
                threshold_crossing
                + window_filter
                - sim_param.param_stimuli["CS"]["start"]
            )
            cr_peak_timing.append(
                np.argmax(normalized_motor_output[trial])
                + window_filter
                - sim_param.param_stimuli["CS"]["start"]
            )
        else:
            cr.append(False)
            cr_timing.append(nan)
            cr_peak_timing.append(nan)

    return cr, cr_timing, cr_peak_timing, normalized_motor_output


# Plotting functions
def add_stimulation_lines(subplot_handle, r, c, tw=[0, sim_param.TOT_DURATION]):
    # Add stimulus and trial lines in the subplot_handle plot, at row and column [r, c], within the time window tw which is the first trial by default
    subplot_handle.add_vline(
        x=tw[0] + sim_param.param_stimuli["CS"]["start"],
        line_dash="dash",
        line_color="black",
        row=r,
        col=c,
    )  # line CS start
    subplot_handle.add_vline(
        x=tw[0] + sim_param.param_stimuli["US"]["start"],
        line_dash="dash",
        line_color="magenta",
        row=r,
        col=c,
    )  # line US starting
    subplot_handle.add_vline(
        x=tw[0] + sim_param.param_stimuli["CS"]["end"],
        line_dash="dash",
        line_color="black",
        row=r,
        col=c,
    )  # line stim end
    subplot_handle.add_vline(
        x=tw[0] + sim_param.td, line_color="black", row=r, col=c
    )  # line trial end


def plot_histogram(spks, cell_num, sim_num=1, window=[0, sim_param.TOT_DURATION], bw=5):
    """
    plot_histogram represents the histogram of spikes given as input

    :param spks: dictionary of spike times per each neuronal population
    :param window: the time window to represent the histogram
    :param bw: bin width in [ms]

    """

    subplots_fig = make_subplots(
        cols=1,
        rows=len(spks.keys()),
        subplot_titles=list(spks.keys()),
        x_title="Time [ms]",
        y_title="Population firing rate [Hz]",
    )
    r = 1
    for cell in spks.keys():
        # create the bins
        counts, bins = np.histogram(
            spks[cell][:, 1], bins=range(int(window[0]), int(window[1]), bw)
        )
        bins = 0.5 * (bins[:-1] + bins[1:])

        subplots_fig.add_trace(
            go.Bar(
                x=bins,
                y=counts / ((bw * 0.001) * cell_num[cell] * sim_num),
                marker_color=color[cell],
                name=label[cell],
            ),
            row=r,
            col=1,
        )
        add_stimulation_lines(subplots_fig, r, 1, tw=window)
        subplots_fig.update_yaxes(range=psth_yrange[cell], row=r, col=1)

        r += 1

    return subplots_fig


def plot_raster(spks, window=[0, sim_param.TOT_DURATION]):
    """
    plot_raster represents the raster plot of spikes given as input

    :param spks: dictionary of spike times per each neuronal population
    :param window: the time window to represent the raster

    """

    subplots_fig = make_subplots(
        cols=1,
        rows=len(spks.keys()),
        subplot_titles=list(spks.keys()),
        x_title="Time [ms]",
        y_title="Neuron id",
    )

    r = 1
    for cell in spks.keys():
        subplots_fig.add_trace(
            go.Scatter(
                x=spks[cell][:, 1],
                y=spks[cell][:, 0],
                mode="markers",
                marker_color=color[cell],
                name=label[cell],
            ),
            row=r,
            col=1,
        )
        add_stimulation_lines(subplots_fig, r, 1, tw=window)

        r += 1
    subplots_fig.update_xaxes(range=window)
    return subplots_fig


def plot_sdf_over_trials(sdf_dict, color_dict, microzone):
    fig = make_subplots(
        cols=1,
        rows=5,
        subplot_titles=[key + " - " + microzone for key in sdf_dict.keys()],
        x_title="Time [ms]",
        y_title="SDF [Hz]",
    )
    for trial in range(0, sim_param.ntrial, step_plot_sdf):
        for cell in sdf_dict.keys():
            color_key = cell.rpartition('-')[0]
            if color_key == '':
                color_key = cell
            # Extract SDF across cells
            avg_sel_sdf_dict = np.mean(sdf_dict[cell][:, :, trial], axis=0)
            for sub_trial in range(trial + 1, trial + step_plot_sdf):
                avg_sel_sdf_dict = np.vstack(
                    (avg_sel_sdf_dict, np.mean(sdf_dict[cell][:, :, sub_trial], axis=0))
                )
            fig.add_trace(
                go.Scatter(
                    y=np.array(np.mean(avg_sel_sdf_dict, axis=0)),
                    mode="lines",
                    line=dict(color=color_dict[color_key], width=1),
                    opacity=0.1 + 0.9 * (trial / sim_param.ntrial),
                ),
                row=rc[cell][0],
                col=rc[cell][1],
            )
    # Add lines of stimuli
    for r in [1, 2, 3, 4]:
        for c in [1]:
            add_stimulation_lines(fig, r, c)
            fig.update_yaxes(
                range=yranges_all_trials[list(sdf_dict.keys())[r - 1]], row=r, col=c
            )

    # Cut axes before and after stimuli in each trial
    fig.update_yaxes(tickfont=dict(family="Arial", size=16))
    fig.update_xaxes(
        range=[
            sim_param.param_stimuli["CS"]["start"] - trial_cut,
            sim_param.param_stimuli["US"]["start"] + trial_cut + 1,
        ]
    )
    fig.update_xaxes(
        tickfont=dict(family="Arial", size=16),
        tickmode="array",
        tickvals=list(range(trial_cut, trial_cut * 4, trial_cut)),
        ticktext=[
            "-" + str(trial_cut),
            "0",
            str(
                sim_param.param_stimuli["US"]["start"]
                - sim_param.param_stimuli["CS"]["start"]
            ),
            str(
                sim_param.param_stimuli["US"]["start"]
                - sim_param.param_stimuli["CS"]["start"]
                + trial_cut
            ),
        ],
    )
    fig.update_annotations(font_size=16)
    fig.update_layout(showlegend=True)

    return fig


def plot_sdf_last_trial(sdf_dict, significant_ind_dict, color_dict, microzone):
    fig = make_subplots(
        cols=1,
        rows=5,
        subplot_titles=[key + " - "+ microzone for key in sdf_dict.keys()],
        x_title="Time [ms]",
        y_title="SDF [Hz]",
    )
    for cell in cell_sdf:
        color_key = cell.rpartition('-')[0]
        print("partition: ", cell.rpartition('-'))
        if color_key == '':
            color_key = cell
        # Extract number of cells firing
        num = sdf_dict[cell].shape[0]
        print("num: ", num, range(num), cell)
        for n in range(num):
            
            if n in list(significant_ind_dict[cell][0]):
                fig.add_trace(
                    go.Scatter(
                        y=sdf_dict[cell][n, :, -1],
                        mode="lines",
                        line=dict(color=light_color[color_key], width=1),
                    ),
                    row=rc[cell][0],
                    col=rc[cell][1],
                )
            
        
        fig.add_trace(
            go.Scatter(
                y=np.nanmean(
                    sdf_dict[cell][significant_ind_dict[cell][0], :, -1], axis=0
                ),
                mode="lines",
                line=dict(color=color_dict[color_key], width=2),
            ),
            row=rc[cell][0],
            col=rc[cell][1],
        )
        
    # Add lines of stimuli
    for r in [1, 2, 3, 4]:
        for c in [1]:
            add_stimulation_lines(fig, r, c)
            fig.update_yaxes(
                range=yranges_last_trial[list(sdf_dict.keys())[r - 1]], row=r, col=c
            )

    # Cut axes before and after stimuli in each trial
    fig.update_xaxes(
        range=[
            sim_param.param_stimuli["CS"]["start"] - trial_cut,
            sim_param.param_stimuli["US"]["start"] + trial_cut + 1,
        ]
    )
    fig.update_xaxes(
        tickfont=dict(family="Arial", size=16),
        tickmode="array",
        tickvals=list(range(trial_cut, trial_cut * 4, trial_cut)),
        ticktext=[
            "-" + str(trial_cut),
            "0",
            str(
                sim_param.param_stimuli["US"]["start"]
                - sim_param.param_stimuli["CS"]["start"]
            ),
            str(
                sim_param.param_stimuli["US"]["start"]
                - sim_param.param_stimuli["CS"]["start"]
                + trial_cut
            ),
        ],
    )
    fig.update_yaxes(tickfont=dict(family="Arial", size=16), range=yranges_last_trial[cell])
    fig.update_annotations(font_size=16)
    fig.update_layout(showlegend=True)
    return fig


# Extract cell properties
neuron_models = {key: [] for key in sim_param.cell_types}
cell_num = {}
cell_num_pos = {}
cell_num_neg = {}
color = {}
label = {}
for cell_name in neuron_models.keys():  # cell_id in sorted_nrn_types:
    path_ids = "cells/placement/" + cell_name + "/identifiers"
    cell_ids = np.array(f[path_ids])
    cell_num[cell_name] = cell_ids[1] * sim_param.sample[cell_name]
    if PLOT_LABELLED and cell_name in pos.keys():
        cell_num_pos[cell_name] = len(pos[cell_name])
        cell_num_neg[cell_name] = len(neg[cell_name])
    color[cell_name] = sim_param.config["cell_types"][cell_name]["plotting"]["color"]
    if cell_name == "mossy_fibers":
        label[cell_name] = "mossy fibers"
    else:
        label[cell_name] = sim_param.config["cell_types"][cell_name]["plotting"][
            "display_name"
        ]
color["mli"] = "#ff6200"
light_color = {
    "purkinje_cell": "lightgreen",
    "mli": "#f6be00",
    "dcn_cell_glut_large": "#708090",
    "io_cell": "mediumpurple",
}

# Load spikes
spks = {}
first_spikes_num = {}
if PLOT_LABELLED:
    pos_spks = {}
    neg_spks = {}
    pos_first_spikes_num = {}
    neg_first_spikes_num = {}
for cell in cell_baselines:
    print("Loading ", cell)
    if cell == "mli":
        spks[cell] = np.concatenate((spks["basket_cell"], spks["stellate_cell"]))
        first_spikes_num[cell] = (
            first_spikes_num["basket_cell"] + first_spikes_num["stellate_cell"]
        )
    else:
        spks[cell], first_spikes_num[cell] = load_spikes(cell, cell + "_spikes*")
        if PLOT_LABELLED and cell in pos.keys():
            # Positive
            pos_spks[cell], pos_first_spikes_num[cell] = load_spikes(
                cell, "pos_" + cell + "_spikes*"
            )
            neg_spks[cell], neg_first_spikes_num[cell] = load_spikes(
                cell, "neg_" + cell + "_spikes*"
            )


# Select first simulation spikes
spks_first = {}
for cell in cell_baselines:
    spks_first[cell] = spks[cell][: first_spikes_num[cell], :]
if PLOT_LABELLED:
    pos_spks_first = {}
    for cell in set(cell_baselines).intersection(pos.keys()):
        pos_spks_first[cell] = pos_spks[cell][: pos_first_spikes_num[cell], :]
    neg_spks_first = {}
    for cell in set(cell_baselines).intersection(pos.keys()):
        neg_spks_first[cell] = neg_spks[cell][: neg_first_spikes_num[cell], :]


if exists(data_path + filename_ss) and exists(data_path + filename_cs) and exists(data_path + filename_ss_neg) and exists(data_path + filename_cs_neg):
   # Load
    print("Loading simple and complex spikes...")
    with open(data_path + filename_ss, "rb") as f:
        ss = pickle.load(f)
    with open(data_path + filename_cs, "rb") as f:
        cs = pickle.load(f)
    with open(data_path + filename_ss_neg, "rb") as f:
        ss_neg = pickle.load(f)
    with open(data_path + filename_cs_neg, "rb") as f:
        cs_neg = pickle.load(f)   
else:
    print("Separating complex spikes...")
    ss, cs = separate_complex_spikes(pos_spks_first["purkinje_cell"])
    ss_neg, cs_neg = separate_complex_spikes(neg_spks_first["purkinje_cell"])


pos_spks_first["purkinje_cell"] = ss
neg_spks_first["purkinje_cell"] = ss_neg
# pos_spks_first['io_cell'] = cs

# Compute baseline firing rate for the first simulation
baseline_rate = {}
baseline_rate_isi = {}
pos_baseline_rate = {}
neg_baseline_rate = {}
pos_baseline_rate_isi = {}
neg_baseline_rate_isi = {}
for cell in cell_baselines:
    baseline_rate[cell], baseline_rate_isi[cell] = compute_spike_rate(
        spks_first[cell], window=[cutoff, 500]
    )
    if cell != "mli" and len(baseline_rate[cell]) < cell_num[cell]:
        print(
            "Less ",
            cell,
            " are firing than total in network ",
            len(baseline_rate[cell]),
            cell_num[cell],
        )
    if cell in pos.keys():
        pos_baseline_rate[cell], pos_baseline_rate_isi[cell] = compute_spike_rate(
            pos_spks_first[cell], window=[cutoff, 500]
        )
    if cell in pos.keys():
        neg_baseline_rate[cell], neg_baseline_rate_isi[cell] = compute_spike_rate(
            neg_spks_first[cell], window=[cutoff, 500]
        )

# print("baselines: ", baseline_rate, pos_baseline_rate, neg_baseline_rate)


if exists(data_path + filename_sdf) and exists(data_path + filename_norm_sdf):
    # Load
    print("Loading SDF...")
    with open(data_path + filename_sdf, "rb") as f:
        sdf = pickle.load(f)
    with open(data_path + filename_norm_sdf, "rb") as f_norm:
        norm_sdf = pickle.load(f_norm)
    # Load
    with open(data_path + filename_neg_sdf, "rb") as f:
        neg_sdf = pickle.load(f)
    with open(data_path + filename_neg_norm_sdf, "rb") as f_norm_neg:
        neg_norm_sdf = pickle.load(f_norm_neg)
    # Recompute if necessary
    for cell in to_recompute_sdf:
        print("Computing SDF...")
        print("...of ", cell)
        if cell == "mli":
            sdf[cell], norm_sdf[cell] = compute_sdf(
                np.concatenate(
                    (spks_first["basket_cell"], spks_first["stellate_cell"]), axis=0
                ),
                gw[cell],
            )
            neg_sdf[cell] = sdf[cell]
            neg_norm_sdf[cell] = norm_sdf[cell]
        elif cell == "purkinje_cell-ss":
            sdf[cell], norm_sdf[cell] = compute_sdf(ss, gw[cell])
            neg_sdf[cell], neg_norm_sdf[cell] = compute_sdf(ss_neg, gw[cell])
        elif cell == "purkinje_cell-cs":
            sdf[cell], norm_sdf[cell] = compute_sdf(cs, gw[cell])
            neg_sdf[cell], neg_norm_sdf[cell] = compute_sdf(cs_neg, gw[cell])
        else:
            sdf[cell], norm_sdf[cell] = compute_sdf(pos_spks_first[cell], gw[cell])
            neg_sdf[cell], neg_norm_sdf[cell] = compute_sdf(neg_spks_first[cell], gw[cell])
else:
    print("Computing SDF...")
    # Compute SDF for MLI, PC (SiSp and CoSp), DCN, IO for the first simulation
    sdf = {}
    norm_sdf = {}
    neg_sdf = {}
    neg_norm_sdf = {}
    for cell in cell_sdf:
        print("...of ", cell)
        if cell == "mli":
            sdf[cell], norm_sdf[cell] = compute_sdf(
                np.concatenate(
                    (spks_first["basket_cell"], spks_first["stellate_cell"]), axis=0
                ),
                gw[cell],
            )
            neg_sdf[cell] = sdf[cell]
            neg_norm_sdf[cell] = norm_sdf[cell]
        elif cell == "purkinje_cell-ss":
            sdf[cell], norm_sdf[cell] = compute_sdf(ss, gw[cell])
            neg_sdf[cell], neg_norm_sdf[cell] = compute_sdf(ss_neg, gw[cell])
        elif cell == "purkinje_cell-cs":
            sdf[cell], norm_sdf[cell] = compute_sdf(cs, gw[cell])
            neg_sdf[cell], neg_norm_sdf[cell] = compute_sdf(cs_neg, gw[cell])
        else:
            sdf[cell], norm_sdf[cell] = compute_sdf(pos_spks_first[cell], gw[cell])
            neg_sdf[cell], neg_norm_sdf[cell] = compute_sdf(neg_spks_first[cell], gw[cell])


# Compute SDF change for each cell type
change = {}
per_change = {}
neg_change = {}
neg_per_change = {}
#  and extract indices of neurons undergoing significant changes in the LAST trial (sim_param.ntrial-1)
significant_ind = {}
neg_significant_ind = {}
for cell_type in sdf.keys():
    if cell_type == "io_cell":
        change[cell_type], per_change[cell_type] = compute_sdf_change(
            sdf[cell_type],
            baseline_trial=0,
            change_trial=sim_param.ntrial - 1,
            change_window=[
                sim_param.param_stimuli["US"]["start"],
                sim_param.param_stimuli["US"]["end"],
            ],
        )
        neg_change[cell_type], neg_per_change[cell_type] = compute_sdf_change(
            neg_sdf[cell_type],
            baseline_trial=0,
            change_trial=sim_param.ntrial - 1,
            change_window=[
                sim_param.param_stimuli["US"]["start"],
                sim_param.param_stimuli["US"]["end"],
            ],
        )
    else:
        # Downbound
        change[cell_type], per_change[cell_type] = compute_sdf_change(
            sdf[cell_type],
            baseline_trial=sim_param.ntrial - 1,
            change_trial=sim_param.ntrial - 1,
            change_window=[
                sim_param.param_stimuli["CS"]["start"] + 50,
                sim_param.param_stimuli["US"]["start"],
            ],
        )
        # Upbound
        neg_change[cell_type], neg_per_change[cell_type] = compute_sdf_change(
            neg_sdf[cell_type],
            baseline_trial=sim_param.ntrial - 1,
            change_trial=sim_param.ntrial - 1,
            change_window=[
                sim_param.param_stimuli["CS"]["start"] + 50,
                sim_param.param_stimuli["US"]["start"],
            ],
        )
    if threshold_change[cell_type] > 0:                 # Threshold for up and downbound cells are opposite
        significant_ind[cell_type] = np.where(
            np.array(per_change[cell_type]) > threshold_change[cell_type]
        )
        neg_significant_ind[cell_type] = np.where(
            np.array(neg_per_change[cell_type]) < -threshold_change[cell_type]
        )
    else:
        significant_ind[cell_type] = np.where(
            np.array(per_change[cell_type]) < threshold_change[cell_type]
        )
        neg_significant_ind[cell_type] = np.where(
            np.array(neg_per_change[cell_type]) > -threshold_change[cell_type]
        )

# print(change, per_change, significant_ind)

# Motor output
if exists(data_path + filename_motor_output):
    with open(data_path + filename_motor_output, "rb") as f_motor:
        motor_output = pickle.load(f_motor)
else:
    print("Computing motor output...")
    # Compute motor output from DCN glut large SDF in a time window of 50 ms, filtered with a moving window
    motor_output = compute_motor_output_delta(
        pos_spks_first["dcn_cell_glut_large"],
        neg_spks_first["dcn_cell_glut_large"],
        wf=window_filter,
    )
    # print(motor_output.shape)


# Computing CR
print("Computing %CR...")
cr, cr_timing, cr_peak_timing, norm_motor = compute_per_cr(motor_output)

# print(cr, cr_timing)
trial_per_block = int(sim_param.ntrial / nblocks)
per_cr = []
for block in range(nblocks):
    per_cr.append(
        100
        * (
            sum(cr[block * trial_per_block : (block + 1) * trial_per_block])
            / trial_per_block
        )
    )
print("% CR: ", per_cr)


# Writing firing rate results on file
file_saving = open(saving_filename, "a")
file_saving.write("Baseline rates:\n")
for cell in cell_baselines:
    file_saving.write(
        "{cell_name}: {mean:.2f}  +- {std:.2f} \n".format(
            cell_name=cell,
            mean=np.mean(baseline_rate[cell]),
            std=np.std(baseline_rate[cell]),
        )
    )
    if cell in pos_baseline_rate.keys():
        file_saving.write(
            "{cell_name} pos (CS-US): {mean:.2f}  +- {std:.2f} \n".format(
                cell_name=cell,
                mean=np.mean(pos_baseline_rate[cell]),
                std=np.std(pos_baseline_rate[cell]),
            )
        )
    if cell in neg_baseline_rate.keys():
        file_saving.write(
            "{cell_name} neg (CS only): {mean:.2f}  +- {std:.2f} \n".format(
                cell_name=cell,
                mean=np.mean(neg_baseline_rate[cell]),
                std=np.std(neg_baseline_rate[cell]),
            )
        )

if SAVING_DATA:
    file_saving.write("SDF change:\n")
    for cell in cell_sdf:
        print("Significant ", np.take(change[cell], significant_ind[cell][0]))
        print("Significant % ", np.take(per_change[cell], significant_ind[cell][0]))
        file_saving.write(
            "{cell_name}: {mean:.2f}  +- {std:.2f} \n".format(
                cell_name=cell,
                mean=np.mean(np.take(change[cell], significant_ind[cell][0])),
                std=np.std(np.take(change[cell], significant_ind[cell][0])),
            )
        )

    file_saving.close()


    # Save ss and cs files downbound
    if not exists(data_path + filename_ss) and not exists(data_path + filename_cs):
        print("Saving downbound PC ss and cs")
        with open(data_path + filename_ss, "wb") as f_ss:
            pickle.dump(ss, f_ss)
        with open(data_path + filename_cs, "wb") as f_cs:
            pickle.dump(cs, f_cs)
    # Save ss and cs files upbound
    if not exists(data_path + filename_ss_neg) and not exists(data_path + filename_cs_neg):
        print("Saving upbound PC ss and cs")
        with open(data_path + filename_ss_neg, "wb") as f_ss_neg:
            pickle.dump(ss_neg, f_ss_neg)
        with open(data_path + filename_cs_neg, "wb") as f_cs_neg:
            pickle.dump(cs_neg, f_cs_neg)

    # Save motor output file
    if not exists(data_path + filename_motor_output):
        with open(data_path + filename_motor_output, "wb") as f_motor:
            pickle.dump(motor_output, f_motor)


    # Save CR onset and peak timing file
    if not exists(data_path + filename_cr):
        with open(data_path + filename_cr, "wb") as f_timing:
            pickle.dump([cr, cr_timing, cr_peak_timing], f_timing)


    # Save baseline and SDF files
    if not exists(data_path + filename_baseline):
        with open(data_path + filename_baseline, "wb") as f:
            pickle.dump([baseline_rate, pos_baseline_rate, neg_baseline_rate], f)


    if not exists(data_path + filename_sdf): 
        with open(data_path + filename_sdf, "wb") as f:
            pickle.dump(sdf, f)
    if not exists(data_path + filename_norm_sdf):
        with open(data_path + filename_norm_sdf, "wb") as f_norm:
            pickle.dump(norm_sdf, f_norm)

    if not exists(data_path + filename_neg_sdf) and not exists(data_path + filename_neg_norm_sdf):
        with open(data_path + filename_neg_sdf, "wb") as f_neg:
            pickle.dump(neg_sdf, f_neg)
        with open(data_path + filename_neg_norm_sdf, "wb") as f_neg_norm:
            pickle.dump(neg_norm_sdf, f_neg_norm)


    print("Saved in ", data_path)




#  PLOTTING RESULTS of ANALYSIS
# Plot last trial SDF for each neuron in the population
print('plotting sdf...')
subplots_fig_norm_sdf_last = plot_sdf_last_trial(norm_sdf, significant_ind, color, microzone='downbound')
subplots_fig_norm_sdf_last.show()
subplots_fig_norm_sdf_last.write_image("sdf_last.svg", scale=1, width=400, height=800, engine="orca")
subplots_fig_norm_sdf_last.write_image("sdf_last.png", scale=1, width=400, height=800, engine="orca")

subplots_fig_norm_sdf_last_neg = plot_sdf_last_trial(neg_norm_sdf, neg_significant_ind, color, microzone='upbound')
subplots_fig_norm_sdf_last_neg.show()
subplots_fig_norm_sdf_last_neg.write_image("neg_sdf_last.svg", scale=1, width=400, height=800, engine="orca")
subplots_fig_norm_sdf_last_neg.write_image("neg_sdf_last.png", scale=1, width=400, height=800, engine="orca")

# Plot SDF average throughout trials
norm_sdf_to_plot = {}
neg_norm_sdf_to_plot = {}
for cell in cell_sdf:
    norm_sdf_to_plot[cell] = norm_sdf[cell][significant_ind[cell][0], :, :]
    neg_norm_sdf_to_plot[cell] = neg_norm_sdf[cell][neg_significant_ind[cell][0], :, :]
subplots_fig_norm_sdf = plot_sdf_over_trials(norm_sdf_to_plot, color, microzone='downbound')
subplots_fig_norm_sdf_neg = plot_sdf_over_trials(neg_norm_sdf_to_plot, color, microzone='upbound')


subplots_fig_norm_sdf.show()
subplots_fig_norm_sdf_neg.show()
subplots_fig_norm_sdf.write_image("sdf_avg.svg", scale=1, width=400, height=800, engine="orca")
subplots_fig_norm_sdf.write_image("sdf_avg.png", scale=1, width=400, height=800, engine="orca")
subplots_fig_norm_sdf_neg.write_image("neg_sdf_avg.svg", scale=1, width=400, height=800, engine="orca")
subplots_fig_norm_sdf_neg.write_image("neg_sdf_avg.png", scale=1, width=400, height=800, engine="orca")

# Plot motor output
print('plotting behavior...')

motor_norm_fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Cerebellar motor output normalized"),
        font=dict(family="Arial", size=16),
    )
)
for trial in range(sim_param.ntrial):
    motor_norm_fig.add_trace(
        go.Scatter(
            x=list(range(int(window_filter), int(sim_param.td))),
            y=norm_motor[trial][:],
            mode="lines",
            line=dict(color="black", width=1),
            opacity=0.05 + 0.95 * (trial / sim_param.ntrial),
        )
    )

add_stimulation_lines(motor_norm_fig, 1, 1)
to_cut = window_filter + gw["dcn_cell_glut_large"]
motor_norm_fig.update_xaxes(
    range=[to_cut + window_filter, sim_param.td - to_cut], title="Time [ms]"
)
motor_norm_fig.update_yaxes(range=[-0.4, 1.2], title="Normalized output")
motor_norm_fig.update_xaxes(
    range=[
        sim_param.param_stimuli["CS"]["start"] - trial_cut,
        sim_param.param_stimuli["US"]["start"] + trial_cut + 1,
    ]
)
motor_norm_fig.update_xaxes(
    tickfont=dict(family="Arial", size=16),
    tickmode="array",
    tickvals=list(range(trial_cut, trial_cut * 4, trial_cut)),
    ticktext=[
        "-" + str(trial_cut),
        "0",
        str(
            sim_param.param_stimuli["US"]["start"]
            - sim_param.param_stimuli["CS"]["start"]
        ),
        str(
            sim_param.param_stimuli["US"]["start"]
            - sim_param.param_stimuli["CS"]["start"]
            + trial_cut
        ),
    ],
)

motor_norm_fig.update_layout(showlegend=False)
motor_norm_fig.show()
motor_norm_fig.write_image(
    "motor_output.svg", scale=1, width=400, height=400)     #, engine="orca")


# Plot %CR
cr_fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Conditioned Responses"),
        font=dict(family="Arial", size=16),
    )
)

cr_fig.add_trace(
    go.Scatter(
        x=list(range(1, nblocks + 1)),
        y=per_cr,
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(color="white", size=6, line=dict(color="black", width=8)),
    )
)
cr_fig.update_xaxes(title="Block #", tickfont=dict(family="Arial", size=16))
cr_fig.update_yaxes(
    title="% CR",
    range=[-5, 105],
    tickfont=dict(family="Arial", size=16),
)


# Add reference experimental values
cr_fig.add_trace(
    go.Scatter(
        x=list(range(1, nblocks + 1)),
        y=mean_per_cr_exp,
        error_y=dict(type="data", array=std_per_cr_exp, visible=True),
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(color="white", size=6, line=dict(color="blue", width=8)),
    )
)
cr_fig.update_layout(showlegend=False)

cr_fig.show()

cr_fig.write_image("CRper.svg", scale=1, width=400, height=400)     #, engine="orca")

# Plot CR timing
timing_fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="CR onset timing"),
        font=dict(family="Arial", size=16),
    )
)
timing_fig.add_trace(
    go.Bar(
        x=["Healthy"],
        y=[np.nanmean(cr_timing)],
        error_y=dict(type="data", array=[np.nanstd(cr_timing)]),
    )
)
timing_fig.update_yaxes(title="Time from CS [ms]")
timing_fig.show()
timing_fig.write_image("timing.svg")
timing_fig.write_image("timing.png")
