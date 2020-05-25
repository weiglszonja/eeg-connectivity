# eeg-connectivity
Framework for computing connectivity measures using EEG data

## Description
Provides a general framework for computing connectivity analysis on EEG data.  
**Preprocess the data before useage**; 
[ICA](https://mne.tools/stable/auto_examples/preprocessing/plot_run_ica.html?highlight=ica%20epochs), 
[filtering or resampling the data](https://mne.tools/stable/auto_tutorials/preprocessing/plot_30_filtering_resampling.html#sphx-glr-auto-tutorials-preprocessing-plot-30-filtering-resampling-py),
etc.  is **not part** of the analysis pipeline.   
The current pipeline was developed to work with 
EEGLAB data (`.set`, `.fdt`) files. 
Other file formats currently are not supported. 
The subject numbers are identified from the EEGLAB file name;
the naming convention should be "`{subject_number}`_*.fdt", 
"`{subject_number}`_*.set" where `subject_number` uniquely identifies a subject.
  
Data I/O operations, connectivity calculation and visualizations are based on
[MNE](https://mne.tools/stable/index.html). For more information about 
connectivity methods refer to the documentation [here](https://mne.tools/dev/generated/mne.connectivity.spectral_connectivity.html).
Other useful tutorials about EEG data processing with MNE can be found 
[here](https://mne.tools/stable/auto_tutorials/intro/plot_10_overview.html#sphx-glr-auto-tutorials-intro-plot-10-overview-py).
## Setup
Clone and install requirements

```bash
$ git clone git@github.com:weiglszonja/eeg-connectivity.git
$ pip install -r requirements.txt
```

## Create epochs
EEGLAB data (`.set`, `.fdt`) files are read from `source_path` variable 
which defaults to `./data/`. Epochs are written to the location specified by
`target_path` variable and defaults to `./epochs_{condition}/` where 
`condition` is the condition of the data  (e.g. can be `'task'` for data recorded 
during task, or `'resting'` for data recorded during resting period).
Condition name is also used when an EEG file was recorded in two or more pieces
and the epochs are needed to be merged into one file. 

The duration of epochs are set with `EPOCH_LENGTH_IN_SECONDS` top level 
variable and it defaults to `2.0` and is defined in seconds. 
For spectral connectivity measures it is recommended to use epochs that are 
overlapping at some amount of data points; 
this is defined with `EPOCH_OVERLAP_RATIO` top level variable and defaults to
 `0.5` that corresponds to 50% overlap between the generated epochs.

Running `create_epochs_from_eeglab.py` creates `.fif` files at target location
containing the epochs created from EEGLAB data. It also generates a `.csv` file 
containing the number of epochs for each data file. 

Execute the script by running the following command:
```bash
$ python utils/create_epochs_from_eeglab.py
```

## Compute the power spectral density of epochs
The script `compute_psd_from_epochs.py` computes the power spectral density 
(PSD) of epochs generated from raw dataset.  
Epochs (`.fif`) files are read from `source_path` variable. The `condition`
variable can be defined as before to separate data by condition (e.g. resting data).
The created figures are saved to the location defined by `target_path`.  

There are two spectral estimation methods that can be used when computing the 
PSD: [Welch](https://mne.tools/stable/generated/mne.time_frequency.psd_welch.html) 
and [multitaper](https://mne.tools/stable/generated/mne.time_frequency.psd_multitaper.html) 
methods from MNE. The valid name of methods are ['welch', 'multitaper'] and is
defined by a top level variable `METHOD`. The default estimation method is the
Welch method.

The min frequency of interest is defined by `F_MIN` top level
variable and defaults to `1.0` and is defined in Hz. The max frequency of 
interest is defined by `F_MAX` that defaults to `45.0` and is defined in Hz.  

Running this script will create figures for each file in sub-directories 
under target location. The default path will save figures at 
`".result/{condition}/psd/plots"`. It will also create CSV files containing 
the calculated spectral densities averaged over the epochs for each channel.
The default location to save the files is at `".result/{condition}/psd/"`.

Execute the script by running the following command:
```bash
$ python utils/compute_psd_from_epochs.py
```

## Compute spectral connectivity between channels
Run connectivity analysis with `-h` argument to display help message   
for list of keyword arguments and possible values:
```bash
$ python compute_channel_connectivity_from_epochs.py -h

usage: compute_channel_connectivity_from_epochs.py [-h]
                                                   [--condition CONDITION]
                                                   [--method METHOD]
                                                   [--workers WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --condition CONDITION, -condition CONDITION
                        The name of the condition (default="asrt")
  --method METHOD, -method METHOD
                        The name of the connectivity estimation method
                        (default=wPLI)
  --workers WORKERS, -w WORKERS
                        The number of epochs to process in parallel
                        (default=8)
```
Where `method` is the name of connectivity measure to compute, 
for supported values visit [MNE documentation](https://mne.tools/dev/generated/mne.connectivity.spectral_connectivity.html).
Some of the supported connectivity measures are: pli, wpli, plv, coh ...  
Specifying `condition` helps to separate the data (e.g. resting data, task data).
Results are created at `./result/` where the sub-folders are structured as the following:  
`./result/{condition}/{method}`. For each connectivity estimation method 
folders are created under `./result/{condition}/` with the name of the condition.
E.g. `./result/asrt/wpli` the results of wPLI connectivity that were computed 
from asrt epochs, and similarly at `./result/asrt/plv` are the results of PLV connectivity. 