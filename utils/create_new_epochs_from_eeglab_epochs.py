import os
import numpy as np
from mne import read_epochs_eeglab, create_info, EpochsArray
from tqdm import tqdm
from utils.create_epochs_from_eeglab import save_epochs

import logging

logging.getLogger().setLevel('INFO')

source_path = os.path.join('data/')
target_path = os.path.join('epochs/')


def process_existing_epochs(set_files):
    with tqdm(total=len(set_files)) as pbar:
        for set_file in set_files:
            source = os.path.join(source_path, set_file)
            epochs = read_epochs_eeglab(source)

            new_epochs = create_new_epochs(epochs=epochs,
                                           duration_in_seconds=2.048,
                                           overlap_ratio=0.5)

            file_name_no_extension = str(set_file.split('.')[0])
            subject, period = file_name_no_extension.split('_')[:2]
            new_epochs.info['file_name'] = f'{subject}_asrt_{period}'
            save_epochs(epochs=new_epochs, target=target_path)

            pbar.update(1)


def create_new_epochs(epochs, duration_in_seconds, overlap_ratio):
    def _check_epochs_are_overlapping():
        """
        Check that created epochs are overlapping and raises error if not.
        For speed concerns only checks overlap for one channel.
        """
        for epoch_num in range(epochs.get_data().shape[0] - 1):
            try:
                assert np.array_equal(
                    epochs.get_data()[epoch_num, 1, n_overlap:],
                    epochs.get_data()[epoch_num + 1, 1, :n_overlap])
            except AssertionError:
                logging.error('Epochs are not overlapping!')
                raise

    logging.info(f'Creating new epochs with duration of {duration_in_seconds}s'
                 f' and {int(overlap_ratio * 100)}% overlap ...')

    epochs_data = epochs.get_data()
    epochs_data = np.transpose(epochs_data, (1, 0, 2))

    n_channels = epochs.info['nchan']
    sfreq = epochs.info['sfreq']
    duration_in_data_points = int(duration_in_seconds * sfreq)
    n_overlap = int(duration_in_data_points * overlap_ratio)
    n_new_epochs = int(
        len(epochs.selection) * (len(epochs.times) / duration_in_data_points))
    new_epochs_data = epochs_data.reshape(n_channels, n_new_epochs,
                                          duration_in_data_points)

    new_epochs_data = np.transpose(new_epochs_data, (1, 0, 2))
    for epoch_ind, epoch in enumerate(new_epochs_data):
        if epoch_ind > 0:
            overlapping_epochs_data = np.append(
                new_epochs_data[epoch_ind - 1, :, n_overlap:],
                new_epochs_data[epoch_ind, :, :n_overlap], axis=-1)
            new_epochs_data[epoch_ind] = overlapping_epochs_data

    # re-arrange data into mne's Epochs object
    tmin = epochs.tmin
    n_epochs = len(epochs.selection)
    n_times = len(epochs.times)

    events = np.zeros((n_new_epochs, 3), int)
    events[:, 0] = np.arange(tmin, n_epochs * n_times, duration_in_data_points)
    events[:, 2] = 1
    event_id = 1
    ch_names = epochs.ch_names
    info = create_info(ch_names=ch_names, sfreq=sfreq,
                       ch_types='eeg',
                       montage=None)

    epochs = EpochsArray(data=new_epochs_data,
                         info=info,
                         events=events,
                         event_id=event_id,
                         tmin=tmin,
                         on_missing='ignore')

    _check_epochs_are_overlapping()

    return epochs


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

        process_existing_epochs(set_files=sorted(set_files))


if __name__ == '__main__':
    main()
