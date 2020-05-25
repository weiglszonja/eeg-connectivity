import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne import read_epochs, Epochs
from mne.connectivity import spectral_connectivity

from matplotlib import pyplot as plt
import seaborn as sns

from utils.external.surface_laplacian import surface_laplacian
from utils.settings import periods, frequency_bands, ROI
import logging

logging.getLogger().setLevel('INFO')

source_path = os.path.join('')
target_path = os.path.join('result/')

# configuration shared across subjects
run_config = {}


def initialize(args: argparse.Namespace):
    """
    Initializes config variables read from keyword arguments
    and creates output folders.
    Args:
        argparse.Namespace args: keyword arguments
    """
    source = os.path.join(source_path, f'epochs_{args.condition}')
    if os.path.exists(source):
        run_config['source_path'] = source
    else:
        logging.error(f'Path {source} for condition {args.condition} '
                      f'not exists. Stopping ...')
        raise FileNotFoundError

    freq_bands = []
    for mode in list(frequency_bands.keys()):
        for freq_band in frequency_bands[mode]:
            os.makedirs(os.path.join(target_path, args.condition, args.method,
                                     freq_band, 'plots'), exist_ok=True)
        freq_bands.extend(list(frequency_bands[mode].keys()))

    run_config['freq_bands'] = freq_bands
    run_config['condition'] = args.condition
    run_config['periods'] = periods[args.condition]
    run_config['method'] = args.method
    run_config['workers'] = args.workers


def process_fif_files(fif_files: list):
    """
    Processes segmented data (Epochs).
    Args:
        list fif_files: list of file names to process
    """

    def _save_channel_connectivity(freq_index: int):
        """
        Writes channel connectivity matrix (n_channels, n_channels) to CSV and
        creates connectivity heatmap figures (.png) for each file.
        """

        conn_df = pd.DataFrame(data=conn[freq_index, ...],
                               index=channels_in_order,
                               columns=channels_in_order)

        frequency_name = run_config['freq_bands'][freq_index]
        conn_id = f'{subject}_{matched_period}_{run_config["method"]}'
        conn_path = os.path.join(target_path,
                                 run_config['condition'],
                                 run_config['method'],
                                 frequency_name)
        conn_df.to_csv(os.path.join(conn_path, f'{conn_id}_ch_conn.csv'),
                       index=True)

        fig_path = os.path.join(conn_path, 'plots', f'{conn_id}_ch_conn.png')
        plot_conn_heatmap(data=conn_df, fig_path=fig_path)

    subjects = sorted(
        list(set([file.split('_')[0] for file in fif_files])))

    channels_in_order = []
    for channels in list(ROI.values()):
        channels_in_order.extend(channels)

    # pre-define multidimensional array to store all data with dimensions of
    # (n_subjects, n_periods, n_frequencies, n_channels, n_channels)
    subjects_ch_conn = np.zeros((len(subjects),
                                 len(run_config['periods']),
                                 len(run_config['freq_bands']),
                                 len(channels_in_order),
                                 len(channels_in_order)))

    for file_name in fif_files:
        file_path = os.path.join(run_config['source_path'], file_name)
        subject = file_name.split('_')[0]
        file_name_no_extension = str(file_name.split('-')[0])
        matched_period = [period for period in run_config['periods'] if
                          period in file_name_no_extension]

        # possibility to skip those periods that are not in the config file
        if len(matched_period):
            matched_period = matched_period[0]
        else:
            continue

        epochs = read_epochs(file_path)
        epochs = epochs.reorder_channels(ch_names=channels_in_order)

        conn_ft = compute_channel_connectivity(epochs=epochs,
                                               method=run_config['method'],
                                               spectrum_mode='fourier',
                                               n_jobs=run_config['workers'])

        conn_mt = compute_channel_connectivity(epochs=epochs,
                                               method=run_config['method'],
                                               spectrum_mode='multitaper',
                                               n_jobs=run_config['workers'])

        conn = np.concatenate((conn_ft, conn_mt), axis=0)

        # save (n_frequencies, n_channels, n_channels) connectivity matrix
        # for given subject, given period (matched by the name of the file)
        subjects_ch_conn[subjects.index(subject),
                         run_config['periods'].index(matched_period)] = conn

        # save connectivity matrix into separate CSV files for each frequency
        # parallel processing to make file operations faster
        Parallel(n_jobs=run_config['workers'])(
            delayed(_save_channel_connectivity)(freq_index) for freq_index in
            range(conn.shape[0]))

    file_path = os.path.join(target_path,
                             run_config['condition'],
                             run_config['method'])

    np.save(os.path.join(file_path,
                         f'subjects_{run_config["method"]}_ch_conn.npy'),
            subjects_ch_conn)


def compute_channel_connectivity(epochs: Epochs,
                                 spectrum_mode: str,
                                 method: str,
                                 n_jobs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel level connectivity matrix from Epochs instance.
    Returns the computed connectivity matrix (n_freqs, n_signals, n_signals).

    Args:
        str spectrum_mode: Valid estimation mode 'fourier' or 'multitaper'
        Epochs epochs: Epochs extracted from a Raw instance
        str method: connectivity estimation method
        int n_jobs: number of epochs to process in parallel
    :return: np.ndarray con: The computed connectivity matrix with a shape of
    (n_freqs, n_signals, n_signals).

    See Also
    --------
    For frequency-decomposition and frequency bin reference:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfftfreq.html
    """
    # spacing between frequency bins
    spacing = epochs.info['sfreq'] / epochs.get_data().shape[-1]
    frequencies = frequency_bands[spectrum_mode].values()
    low_cutoff = tuple(band[0] for band in frequencies)
    high_cutoff = tuple(band[1] - spacing for band in frequencies)

    if method in ['coh', 'ciplv', 'ppc'] or spectrum_mode == 'multitaper':
        epochs = surface_laplacian(epochs=epochs)

    con, _, _, _, _ = spectral_connectivity(data=epochs,
                                            method=method,
                                            sfreq=epochs.info['sfreq'],
                                            mode=spectrum_mode,
                                            fmin=low_cutoff,
                                            fmax=high_cutoff,
                                            faverage=True,
                                            n_jobs=n_jobs,
                                            verbose=True)

    # from shape of (n_signals, n_signals, n_freqs) to
    # (n_freqs, n_signals, n_signals)
    con = np.transpose(con, (2, 0, 1))
    con = abs(con)
    return con


def plot_conn_heatmap(data: pd.DataFrame, fig_path: str):
    """
    Visualize heatmap of (n_connections, n_connections) connectivity data.

    Args:
        pd.DataFrame data: df to visualize on heatmap
        str fig_path: the path where to save the figure (with figure name
        and file extension)
    """
    mask = np.zeros_like(data)
    mask[data == 0.] = True

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.set(style='white', font_scale=1.4)
    sns.heatmap(data,
                mask=mask,
                square=True,
                vmin=-.6,
                vmax=.6,
                cbar=False if 'roi' in fig_path else True,
                annot=True if 'roi' in fig_path else False,
                cmap='coolwarm',
                cbar_kws={"shrink": .5},
                ax=ax)
    cax = plt.gcf().axes[0]
    cax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(fname=fig_path, dpi=200, transparent=False)

    plt.close(fig=fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--condition', '-condition',
                        help='The name of the condition '
                             '(default="asrt")',
                        type=str,
                        default='asrt')
    parser.add_argument('--method', '-method',
                        help='The name of the connectivity estimation method '
                             '(default=wPLI)',
                        type=str,
                        default='wpli')
    parser.add_argument('--workers', '-w',
                        help='The number of epochs to process in parallel '
                             '(default=8)',
                        type=int,
                        default=8,
                        required=False)

    args = parser.parse_args()

    initialize(args=args)

    fif_files = [file for file in os.listdir(run_config['source_path'])
                 if file.endswith('.fif')]

    process_fif_files(fif_files=sorted(fif_files))


if __name__ == '__main__':
    main()
