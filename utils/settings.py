periods = {'asrt': ['asrt_1_1', 'asrt_1_2', 'asrt_1_3', 'asrt_2'],
           'rs': ['ny_1', 'ny_2', 'ny_3', 'ny_4']}

frequency_bands = {
    'fourier':
        {
            'delta': [1, 4],
            'theta': [4, 8],
            'alpha': [8, 13],
            'beta': [13, 30]},
    'multitaper':
        {
            'gamma': [30, 45]}
}

ROI = {'frontal_left': ['F7', 'F5', 'F3', 'FC5', 'FC3'],
       'frontal_central': ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2'],
       'frontal_right': ['F4', 'F6', 'F8', 'FC4', 'FC6'],
       'temporal_left': ['FT7', 'T7', 'TP7'],
       'central': ['C3', 'Cz', 'C4'],
       'temporal_right': ['FT8', 'T8', 'TP8'],
       'parietal_left': ['CP5', 'CP3', 'P7', 'P5', 'P3'],
       'parietal_central': ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2'],
       'parietal_right': ['CP4', 'CP6', 'P4', 'P6', 'P8'],
       'occipital_left': ['PO3', 'PO7', 'O1'],
       'occipital_right': ['PO4', 'PO8', 'O2']}
