# eeg-connectivity
Framework for computing connectivity measures using EEG data

## Description
Provides a general framework for computing connectivity analysis on EEG data.
Preprocess the data before useage; ICA, filtering, etc. beforehand is *not part*
of the analysis pipeline. 
  
Data I/O operations, connectivity calculation and visualizations are based on
[MNE](https://mne.tools/stable/index.html). For more information about 
connectivity methods refer to the documentation [here](https://mne.tools/dev/generated/mne.connectivity.spectral_connectivity.html).

## Setup
Clone and install requirements

```bash
$ git clone git@github.com:weiglszonja/eeg-connectivity.git
$ pip install -r requirements.txt
```

