import os
from pathlib import Path
import warnings

import numpy as np
from joblib import Parallel, delayed
from mne import Epochs, read_epochs, EpochsArray
from mne.utils import logger
from tqdm import tqdm

from compute_channel_connectivity_from_epochs import compute_channel_connectivity
from utils.settings import ROI

EPOCHS_FILE_PATH = "/Users/weian/Downloads/epochs_zsofi/"
CONDITION = "rs"
EPOCHS_FILE_POSTFIX = ".fif"

N_SURROGATES = 1000

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def shuffle_along_axis(a, axis=-1):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def parallel_conn_func(a):
    conn_ft = compute_channel_connectivity(epochs=a,
                                           method='wpli',
                                           spectrum_mode='fourier',
                                           n_jobs=8)

    conn_mt = compute_channel_connectivity(epochs=a,
                                           method='wpli',
                                           spectrum_mode='multitaper',
                                           n_jobs=8)

    conn = np.concatenate((conn_ft, conn_mt), axis=0)

    return conn


def estimate_surrogates_from_epochs(epochs: Epochs):
    data = epochs.get_data().copy()
    #  Generate reference to shuffle function
    for n_surrogate in range(N_SURROGATES):
        surrogate_data = shuffle_along_axis(a=data, axis=-1)
        surrogate = EpochsArray(data=surrogate_data,
                                info=epochs.info,
                                events=epochs.events,
                                verbose=False)

        yield surrogate


def run():
    logger.info(
        f'\nEstimating {N_SURROGATES} SURROGATES for EEG data:'
    )

    pipeline_in = Path(EPOCHS_FILE_PATH + f"/epochs_{CONDITION}")
    pipeline_out = pipeline_in.parent / 'surrogates' / 'wpli'

    if not os.path.exists(pipeline_out):
        os.makedirs(pipeline_out, exist_ok=True)

    files = sorted(list(pipeline_in.rglob(f"*{EPOCHS_FILE_POSTFIX}")))

    channels_in_order = []
    for channels in list(ROI.values()):
        channels_in_order.extend(channels)

    pbar = tqdm(sorted(files), position=0, leave=True)
    for file in pbar:
        pbar.set_description("Processing %s" % file.stem)
        file_name_no_extension = str(file.name.split('-')[0])

        if (
                pipeline_out / f"{file_name_no_extension}_surrogates_ch_conn.npy").is_file():
            pbar.update(1)
            continue

        epochs = read_epochs(pipeline_in / file, verbose=False)
        epochs = epochs.reorder_channels(ch_names=channels_in_order)

        # conn = np.zeros((N_SURROGATES,
        #                  5,
        #                  47,
        #                  47))
        # for n_surrogate in tqdm(range(N_SURROGATES)):
        #     surrogate = Parallel(n_jobs=-1, backend="multiprocessing")(
        #         delayed(shuffle_along_axis)(dat) for dat in
        #         batch(epochs.get_data(), 100)
        #     )
        #     surrogate_data = np.concatenate(surrogate, axis=0)
        #     surrogate_epochs = EpochsArray(data=surrogate_data,
        #                                    info=epochs.info,
        #                                    events=epochs.events,
        #                                    verbose=False)
        #
        #     surrogate_conn = parallel_conn_func(surrogate_epochs)
        #     conn[n_surrogate] = surrogate_conn

        #
        # surrogates = Parallel(n_jobs=-1, backend="multiprocessing")(
        #     delayed(shuffle_along_axis)(dat) for dat in tqdm(batch(epochs.get_data(), 10))
        # )
        #
        # x = np.stack(surrogates)
        #

        # for dat in batch(epochs.get_data(), 200):
        #     a = shuffle_along_axis(dat, axis=-1)

        conn = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(parallel_conn_func)(surrogate) for surrogate in
            tqdm(
                estimate_surrogates_from_epochs(epochs=epochs),
                position=0,
                leave=True,
                desc=f'{file_name_no_extension} surrogates ...'
            ))

        surrogates = np.stack(conn)
        logger.info("Saving surrogate file ...")

        np.save(
            pipeline_out / f"{file_name_no_extension}_surrogates_ch_conn.npy",
            surrogates
        )
        logger.info("Saving complete.")


if __name__ == '__main__':
    run()
