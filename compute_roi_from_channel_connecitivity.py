import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd

from utils.settings import periods, frequency_bands, ROI
from compute_channel_connectivity_from_epochs import plot_conn_heatmap

import logging

logging.getLogger().setLevel('INFO')

source_path = os.path.join('result/')

# configuration shared across subjects
run_config = {}


def initialize(args: argparse.Namespace):
    """
    Initializes config variables read from keyword arguments
    and creates output folders.
    Args:
        argparse.Namespace args: keyword arguments
    """

    def _get_subjects_from_epochs_data(path):
        epochs_files = [file for file in os.listdir(path) if
                        file.endswith('.fif')]
        return sorted(
            list(set([file.split('_')[0] for file in epochs_files])))

    def _get_frequency_bands_from_settings():
        bands = []
        for mode in list(frequency_bands.keys()):
            for freq_band in frequency_bands[mode]:
                os.makedirs(os.path.join(source, freq_band, 'ROI', 'plots'),
                            exist_ok=True)
            bands.extend(list(frequency_bands[mode].keys()))
        return bands

    run_config['verbose'] = args.verbose
    if run_config['verbose']:
        subjects = _get_subjects_from_epochs_data(
            path=f'/epochs_{args.condition}')
        run_config['subjects'] = subjects

        freq_bands = _get_frequency_bands_from_settings()
        run_config['frequency_bands'] = freq_bands

    source = os.path.join(source_path, args.condition, args.method)
    logging.info(f'Initializing ROI averaging at {source} ...')
    run_config['source'] = source
    run_config['method'] = args.method
    run_config['condition'] = args.condition


def load_connectivity_matrix_from_path(path: str) -> np.ndarray:
    """
    Loads a channel connectivity matrix (.npy) into multidimensional array.
    Checks the shape of the matrix to match the expected number of dimensions:
    (n_subjects, n_periods, n_frequencies, n_channels, n_channels)

    Args:
    str path: the path to the channel connectivity matrix
    :return: the loaded channel matrix with a shape of
    (n_subjects, n_periods, n_frequencies, n_channels, n_channels)
    """
    ch_fname = [file for file in os.listdir(path) if
                file.endswith('ch_conn.npy')][0]

    logging.info(f'Reading matrix from {os.path.join(path, ch_fname)}\n')
    conn = np.load(os.path.join(path, ch_fname))
    assert len(conn.shape) == 5
    logging.info(f'Shape of matrix: {conn.shape}\n'
                 f'Description of dimensions: \n'
                 f'Number of subjects: {conn.shape[0]}\n'
                 f'Number of periods: {conn.shape[1]}\n'
                 f'Number of frequency bands: {conn.shape[2]}\n'
                 f'Channel connectivity matrix: '
                 f'{conn.shape[3]} x {conn.shape[4]}')

    return conn


def compute_roi_from_channel_connectivity(conn: np.ndarray):
    """
    Computes ROI averaging on multidimensional array that contains the
    connectivity estimate between channels. The data is expected to have
    the following dimensions: (n_subjects, n_periods, n_frequencies,
    n_channels, n_channels).

    The resulting matrix will have a shape of
    (n_subjects, n_periods, n_frequencies, n_roi, n_roi).

    Args:
    np.ndarray conn: the array containing the connectivity values.
    """
    # pre-define multidimensional array to store all data with dimensions of
    # (n_subjects, n_periods, n_frequencies, n_roi, n_roi)
    n_roi = len(list(ROI.keys()))
    n_subjects = conn.shape[0]
    n_periods = conn.shape[1]
    n_frequencies = conn.shape[2]
    subjects_roi_conn = np.zeros((n_subjects,
                                  n_periods,
                                  n_frequencies,
                                  n_roi,
                                  n_roi,
                                  ))

    logging.info(
        f'Averaging {conn.shape[3]}x{conn.shape[4]} {run_config["method"]} '
        f'channel connectivity into {n_roi}x{n_roi} ROIs for {n_subjects} '
        f'subjects ...')

    for subject in range(n_subjects):
        if 'subjects' in run_config:
            subject_id = run_config['subjects'][subject]

        for period in range(n_periods):
            period_id = periods[run_config['condition']][period]

            for frequency_band in range(n_frequencies):
                conn_array = conn[subject, period, frequency_band, ...]
                roi_conn = calculate_roi_averages_from_array(data=conn_array)

                subjects_roi_conn[subject, period, frequency_band] = roi_conn

                if run_config['verbose']:
                    frequency_band_name = run_config['frequency_bands'][
                        frequency_band]
                    file_name = f'{subject_id}_{period_id}_roi_conn.csv'
                    file_path = os.path.join(run_config['source'],
                                             frequency_band_name,
                                             'ROI')
                    roi_conn.to_csv(os.path.join(file_path, file_name),
                                    index=True)

                    fig_name = f'{subject_id}_{period_id}_roi_conn.png'
                    fig_path = os.path.join(file_path, 'plots', fig_name)
                    plot_conn_heatmap(data=roi_conn, fig_path=fig_path)

        logging.info(f'[{subject}/{n_subjects} ROI averaging done]')

    logging.info(f'Writing subjects_{run_config["method"]}_roi_conn.npy '
                 f'file at {run_config["source"]}')
    np.save(os.path.join(run_config["source"],
                         f'subjects_{run_config["method"]}_roi_conn.npy'),
            subjects_roi_conn)
    logging.info(f'[FINISHED]')


def calculate_roi_averages_from_array(data: np.ndarray) -> pd.DataFrame:
    """
    Calculates ROI averaging on connectivity matrix (n_channels, n_channels)
    Args:
        np.ndarray data: array containing channel connectivity data with a
        shape of (n_channels, n_channels)
    :return: ROI connectivity (n_roi, n_roi) DataFrame
    """
    channels_in_order = []
    for channels in list(ROI.values()):
        channels_in_order.extend(channels)

    ch_conn = pd.DataFrame(data=data,
                           index=channels_in_order,
                           columns=channels_in_order)

    between_roi_pairs = list(combinations(ROI.keys(), 2))
    roi_names = list(ROI.keys())
    roi_conn = pd.DataFrame(index=roi_names, columns=roi_names)
    for roi_name in roi_names:
        # locate within roi values in connectivity matrix
        conn_per_within_roi = ch_conn.loc[
            ROI[roi_name], ROI[roi_name]].replace(0., np.NaN)
        # calculate mean only for non-zero values
        within_roi_values = conn_per_within_roi.T.stack().values
        within_roi_mean = np.mean(within_roi_values)
        roi_conn.loc[roi_name, roi_name] = within_roi_mean

        # avoid duplications for roi pairs
        between_roi_names = [between_roi_name[1] for between_roi_name
                             in between_roi_pairs if
                             between_roi_name[0] == roi_name]

        for between_roi_name in between_roi_names:
            # locate between roi values in connectivity matrix
            conn_per_between_roi = ch_conn.loc[
                ROI[between_roi_name], ROI[roi_name]].replace(0, np.NaN)
            # calculate mean only for non-zero values
            between_roi_values = conn_per_between_roi.T.stack().values
            between_roi_mean = np.mean(between_roi_values)
            roi_conn.loc[
                between_roi_name, roi_name] = between_roi_mean

    roi_conn.fillna(0., inplace=True)
    return roi_conn


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
    parser.add_argument('--verbose', '-verbose',
                        help='Turn on interim processing: write to files (CSV)'
                             'and save heatmap figures (PNG) (default=False)',
                        type=bool,
                        default=False)

    args = parser.parse_args()

    initialize(args=args)

    ch_conn = load_connectivity_matrix_from_path(path=run_config['source'])
    compute_roi_from_channel_connectivity(conn=ch_conn)


if __name__ == '__main__':
    main()
