import os
from typing import Tuple

import numpy as np
from mne import read_epochs, Epochs
from mne.time_frequency import psd_welch, psd_multitaper
from utils.settings import periods

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import logging

logging.getLogger().setLevel('INFO')

condition = 'asrt'
source_path = os.path.join(f'epochs_{condition}/')
target_path = os.path.join('result/')

METHOD = 'welch'
F_MIN = 1.0
F_MAX = 45.0


def compute_psd_from_epochs(epochs: Epochs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes power spectral density (PSD) from Epochs instance using MNE.
    Returns the power spectral densities (psds) in a shape of
    (n_epochs, n_channels, n_freqs) and the frequencies (freqs) in a shape of
    (n_freqs,).

    Args:
        Epochs epochs: Epochs instance to be used for PSD estimation
    :return: np.ndarray psds, np.ndarray freqs

    See Also
    --------
    mne.time_frequency.psd_welch : Computation of PSD using Welch's method.
    mne.time_frequency.psd_multitaper : Computation of PSD using multitapers.
    """
    epoch_length = epochs.get_data().shape[-1]

    if METHOD == 'welch':
        psds, freqs = psd_welch(epochs,
                                fmin=F_MIN,
                                fmax=F_MAX,
                                n_fft=epoch_length)

    elif METHOD == 'multitaper':
        psds, freqs = psd_multitaper(epochs,
                                     fmin=F_MIN,
                                     fmax=F_MAX)

    else:
        logging.error('Not a valid method for computing PSD, '
                      'valid methods are: welch, multitaper.')
        raise

    return psds, freqs


def plot_psd(psd: np.ndarray, freqs: np.ndarray, file_name: str):
    """
    Visualizes the mean PSD in dB.
    The mean +/- 1 STD (across channels) is plotted in a grey area around the
    PSD averaged over epochs and channels.
    Creates a PNG file at target location with a specified name.

    Args:
        np.ndarray psd: Array containing the spectral densities with a
        shape of (n_epochs, n_channels, n_freqs).
        np.ndarray freqs: Array containing the frequency bins with a
        shape of (n_freqs,).
        str file_name: The name of the figure
    """
    psd = 10. * np.log10(psd)
    psd_mean = psd.mean(axis=0).mean(axis=0)
    psd_std = psd.mean(axis=0).std(axis=0)

    fig = plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, aspect='equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot(freqs, psd_mean, color='k')
    ax.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std,
                    color='k', alpha=0.5)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set(title=f'{METHOD.upper()} PSD',
           xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')

    fig.tight_layout()

    plt.savefig(os.path.join(target_path, condition, 'psd', 'plots',
                             f'{file_name}_{METHOD}_psd.png'),
                dpi=200)

    plt.close(fig)


def main():
    os.makedirs(os.path.join(target_path, condition, 'psd', 'plots'),
                exist_ok=True)

    fif_files = [file for file in os.listdir(source_path) if
                 file.endswith('.fif')]

    if not len(fif_files):
        logging.warning(f'There are no .fif files in {source_path}, '
                        f'nothing to do ...')
        return

    else:
        logging.info(f'Collected {len(fif_files)} .fif files '
                     f'from "{os.path.abspath(source_path)}"')

        subjects = sorted(
            list(set([file.split('_')[0] for file in fif_files])))
        subjects_psd = np.empty((len(subjects),
                                 len(periods[condition])),
                                dtype=object)

        for fif_file in sorted(fif_files):
            subject = fif_file.split('_')[0]
            file_name_no_extension = str(fif_file.split('-')[0])
            matched_period = [period for period in periods[condition] if
                              period in file_name_no_extension][0]
            epochs = read_epochs(os.path.join(source_path, fif_file))
            epochs.info['file_name'] = file_name_no_extension

            psd, freq = compute_psd_from_epochs(epochs=epochs)
            psd_mean_epochs = psd.mean(axis=0)
            # re-arrange axis to (n_frequencies, n_electrodes)
            psd_mean_epochs = np.transpose(psd_mean_epochs, (1, 0))

            subjects_psd[subjects.index(subject),
                         periods[condition].index(
                             matched_period)] = psd_mean_epochs

            plot_psd(psd=psd, freqs=freq, file_name=file_name_no_extension)

        np.save(os.path.join(target_path, condition, 'psd',
                             f'subjects_{METHOD}_psd.npy'), subjects_psd)


if __name__ == '__main__':
    main()
