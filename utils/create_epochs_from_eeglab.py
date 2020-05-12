import os
import numpy as np
from pandas import DataFrame
from mne.io import read_raw_eeglab, Raw
from mne import make_fixed_length_events, Epochs, concatenate_epochs
from tqdm import tqdm

import logging

logging.getLogger().setLevel('INFO')

condition = 'asrt'
source_path = os.path.join('data/')
target_path = os.path.join(f'epochs_{condition}/')

EPOCH_LENGTH_IN_SECONDS = 2.0
EPOCH_OVERLAP_RATIO = 0.5


def process_set_files(set_files: list, merge_patterns: list):
    """
    Creates and saves epochs for .set files.
    list set_files: The list of .set files
    list merge_patterns: The list of patterns to be merged into one epoch
    (e.g. the EEG was recorded in two pieces) If nothing to merge set to [].
    """
    epoch_meta = DataFrame()
    epochs_to_merge = {}

    with tqdm(total=len(set_files)) as pbar:
        for set_file in set_files:
            source = os.path.join(source_path, set_file)

            raw = read_raw_eeglab_from_source(source=source,
                                              channels_to_drop=['EOG'])
            epochs = create_epochs(raw=raw, picks=None)

            file_name_no_extension = str(set_file.split('.')[0])
            subject = file_name_no_extension.split('_')[0]
            epoch_meta = epoch_meta.append(
                {'subject': subject,
                 'file_name': file_name_no_extension,
                 'n_epochs': epochs.get_data().shape[0]},
                ignore_index=True)

            epochs.info['file_name'] = file_name_no_extension
            if any(pattern in file_name_no_extension for pattern in
                   merge_patterns):
                epochs_to_merge.setdefault(subject, []).append(epochs)
            else:
                save_epochs(epochs, target_path)
                pbar.update(1)

        if epochs_to_merge:
            for merged_epochs in merge_epochs(
                    epochs_to_merge_per_subject=epochs_to_merge,
                    merge_patterns=merge_patterns):
                save_epochs(merged_epochs, target_path)
                pbar.update(int(len(merge_patterns) / 2))

    epoch_meta.to_csv(
        os.path.join(target_path, 'subjects_epoch_count.csv'),
        index=False)


def read_raw_eeglab_from_source(source: str, channels_to_drop=None) -> Raw:
    '''
    Read raw EEGLAB .set data from source

    Args:
        str source: Path to .set data
        list channels_to_drop: The list of channel names to drop (EOG channels)
    :return: Raw raw: The object containing EEGLAB .set data

    See Also
    --------
    mne.io.read_raw_eeglab : Documentation of parameters.
    '''
    try:
        raw = read_raw_eeglab(source, preload=True).set_annotations(None)
    except Exception as e:
        logging.error(f'Error while trying to read {source}, error: {e}')
        raise
    if channels_to_drop is not None:
        matched_channels = []
        for channel in channels_to_drop:
            matched_channels.extend(
                [raw_ch_name for raw_ch_name in raw.ch_names if
                 channel in raw_ch_name])

        raw = raw.drop_channels(ch_names=matched_channels)

    return raw


def create_epochs(raw: Raw, picks=None) -> Epochs:
    '''
    Create epochs from Raw instance with overlap between epochs.

    Args:
        Raw raw: Raw object containing EEGLAB .set data
        list picks: Subset of channel names to include in Epochs data,
        if None use all channels from Raw instance
    :return: Epochs epochs:  Epochs extracted from Raw instance

    See Also
    --------
    mne.io.Raw, mne.Epochs : Documentation of attribute and methods.
    '''

    def _check_epochs_are_overlapping():
        """
        Check that created epochs are overlapping and raises error if not.
        For speed concerns only checks overlap for one channel.
        """
        n_epochs = epochs.get_data().shape[0]
        overlap_data_points = int(overlap_in_seconds * raw.info['sfreq'])
        for epoch_num in range(n_epochs - 1):
            try:
                assert np.array_equal(
                    epochs.get_data()[epoch_num, 1, overlap_data_points:],
                    epochs.get_data()[epoch_num + 1, 1, :overlap_data_points])
            except AssertionError:
                logging.error('Epochs are not overlapping!')
                raise

    overlap_in_seconds = EPOCH_OVERLAP_RATIO * EPOCH_LENGTH_IN_SECONDS
    events = make_fixed_length_events(raw,
                                      id=1,
                                      first_samp=True,
                                      duration=EPOCH_LENGTH_IN_SECONDS,
                                      overlap=overlap_in_seconds)
    epochs = Epochs(raw=raw,
                    events=events,
                    picks=picks,
                    event_id=1,
                    baseline=None,
                    tmin=0.,
                    tmax=EPOCH_LENGTH_IN_SECONDS - 0.001,
                    preload=True)

    _check_epochs_are_overlapping()

    return epochs


def merge_epochs(epochs_to_merge_per_subject: dict, merge_patterns: list):
    """
    Merge a list of Epochs instances into one Epochs instance

    Args:
        dict epochs_to_merge_per_subject: list of epochs to be merged
        list merge_patterns: file name patterns to perform merge

    """
    for subject, epochs in epochs_to_merge_per_subject.items():
        matched_pattern = [pattern for pattern in merge_patterns if
                           pattern in epochs[0].info['file_name']][0]

        new_file_name = f'{subject}_{condition}_{matched_pattern[:-2]}'
        merged_epochs = concatenate_epochs(epochs_list=epochs,
                                           add_offset=True)
        merged_epochs.info['file_name'] = new_file_name

        yield merged_epochs


def save_epochs(epochs: Epochs, target: str):
    """
    Save Epochs instance in a fif file if not exists.
    Args:
        Epochs epochs: Epochs instance to save
        str target: path to save the epochs
    """

    file_name = epochs.info['file_name']
    if not os.path.exists(os.path.join(target, f'{file_name}-epo.fif')):
        epochs.save(os.path.join(target, f'{file_name}-epo.fif'))


def main():
    os.makedirs(target_path, exist_ok=True)

    set_files = [file for file in os.listdir(source_path) if
                 file.endswith('.set')]

    if not len(set_files):
        logging.warning(f'There are no .set files in {source_path}, '
                        f'nothing to do ...')
        return

    else:
        logging.info(f'Collected {len(set_files)} .set files '
                     f'from "{os.path.abspath(source_path)}"')

        process_set_files(set_files=sorted(set_files),
                          merge_patterns=['1_1_1', '1_1_2', '1_3_1', '1_3_2'])


if __name__ == '__main__':
    main()
