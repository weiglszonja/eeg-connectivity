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

# EEG 64 dataset
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

# EEG 128 dataset
# ROI = {'frontal_left': ['E26', 'E33', 'E27', 'E23', 'E28', 'E24'],
#        'frontal_central': ['E15', 'E18', 'E16', 'E10', 'E19', 'E11', 'E4',
#                            'E20', 'E12', 'E5', 'E118'],
#        'frontal_right': ['E2', 'E3', 'E123', 'E122', 'E124', 'E117'],
#        'temporal_left': ['E40', 'E45', 'E46', 'E50', 'E58'],
#        'central_left': ['E35', 'E29', 'E41', 'E36', 'E47', 'E42'],
#        'central': ['E6', 'E13', 'E112', 'E30', 'E7', 'E106', 'E105',
#                    'E37', 'E31', 'E80', 'E87'],
#        'central_right': ['E111', 'E110', 'E104', 'E103', 'E93', 'E98'],
#        'temporal_right': ['E109', 'E108', 'E102', 'E101', 'E96'],
#        'parietal_left': ['E53', 'E52', 'E51', 'E60', 'E59', 'E64'],
#        'parietal_central': ['E55', 'E54', 'E79', 'E61', 'E62', 'E78',
#                             'E67', 'E72', 'E77'],
#        'parietal_right': ['E97', 'E92', 'E86', 'E85', 'E91', 'E95'],
#        'occipital_left': ['E66', 'E71', 'E65', 'E70', 'E69', 'E74'],
#        'occipital_right': ['E84', 'E76', 'E90', 'E83', 'E89', 'E82']}
