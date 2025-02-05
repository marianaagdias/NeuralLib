import os
import numpy as np
import json
import random
from NeuralLib.config import ECG_DATA_PATH, PEAKS_DATA_PATH, DATASETS_GIB01
from data_preprocessing.signal_preprocessing import resampling, split_signal
import sklearn.preprocessing as pp


X = os.path.join(DATASETS_GIB01, 'x')
Y_IDX = os.path.join(DATASETS_GIB01, 'y_idx')
Y_BIN = os.path.join(DATASETS_GIB01, 'y_bin')

os.makedirs(X, exist_ok=True)
os.makedirs(Y_IDX, exist_ok=True)
os.makedirs(Y_BIN, exist_ok=True)


def load_ecg_data():
    # Load the ECG data from the .npz file
    data = np.load(ECG_DATA_PATH)

    signals = {}
    for key in data:
        tuple_key = eval(key)  # convert the string '(1, 1)' to the tuple (1, 1)
        signals[tuple_key] = data[key]  # convert from npzfile to dictionary

    # Load the peaks data from the .json file
    with open(PEAKS_DATA_PATH, 'r') as f:
        peaks_json = json.load(f)

    peaks = {}
    for key in peaks_json:
        tuple_key = eval(key)  # Convert the string '(1, 1)' to the tuple (1, 1)
        peaks[tuple_key] = peaks_json[key]

    return signals, peaks


def inspect_data(signals):
    print("Keys in the .npz file:", signals.keys())
    for key in signals:
        print(f"Length of signal {key}: {len(signals[key])}")


def inspect_peaks(peaks):
    print("Number of peaks:", len(peaks.keys()))
    for key in peaks:
        print(f"Number of peaks of {key}: {len(peaks[key])}")


def resample_peaks_idx(peaks, target_sampling_rate, original_sampling_rate):
    for key in peaks:
        peaks[key] = (np.array(peaks[key]) * (target_sampling_rate / original_sampling_rate)).astype(int)
    return peaks


def peaks_to_binary(peaks, signals):
    '''
    Receives the peaks dictionary with the array of peak indexes of each signal and converts those arrays to binary
    signals of the same length as the ECG signal with 1's corresponding to the peaks
    :param peaks: peaks dictionary (peaks indexes array of each ECG signal)
    :return: peaks_binary dictionary
    '''
    peaks_binary = {}
    for key in peaks:
        binary = np.zeros(len(signals[key]))
        binary[peaks[key]] = 1
        peaks_binary[key] = binary
    return peaks_binary


def split_signals(signals, peaks_binary, peaks_idx, sampling_rate, segment_duration_min, segment_duration_max):
    """
    Splits each signal in the signals dictionary into smaller segments and stores them in new dictionaries
    for signals, binary peaks, and peaks as indexes.

    :param signals: A dictionary containing signals.
    :param peaks_binary: A dictionary containing binary peak signals.
    :param peaks_idx: A dictionary containing peak indexes.
    :param sampling_rate: The sampling rate of the signals (in Hz).
    :param segment_duration_min: Minimum duration of each segment in seconds.
    :param segment_duration_max: Maximum duration of each segment in seconds.
    :return: Three dictionaries with segmented signals, binary peaks, and peak indexes.
    """
    segmented_signals = {}
    segmented_peaks_binary = {}
    segmented_peaks_idx = {}

    for key in signals:
        segment_duration = random.randint(segment_duration_min, segment_duration_max)
        print(f"segment duration: {segment_duration}")
        signal_segments = split_signal(signals[key], sampling_rate, segment_duration)
        binary_segments = split_signal(peaks_binary[key], sampling_rate, segment_duration)

        assert len(signal_segments) == len(binary_segments), f"Mismatch in number of segments for key {key}: " \
                                                             f"signal_segments={len(signal_segments)}, " \
                                                             f"binary_segments={len(binary_segments)}"

        for i, segment in enumerate(signal_segments):
            new_key = key + (i + 1,)
            segmented_signals[new_key] = segment
            segmented_peaks_binary[new_key] = binary_segments[i]

            start_idx = i * segment_duration * sampling_rate
            end_idx = (i + 1) * segment_duration * sampling_rate

            # Filter peaks that fall within the current segment
            segment_peaks = [p - start_idx for p in peaks_idx[key] if start_idx <= p < end_idx]
            segmented_peaks_idx[new_key] = segment_peaks

            print(f"Created segment {new_key} with signal shape: {segment.shape}, "
                  f"binary peaks shape: {binary_segments[i].shape}, "
                  f"and peaks indexes: {segment_peaks}")

    return segmented_signals, segmented_peaks_binary, segmented_peaks_idx


def split_keys(keys, test_percentage=0.2, val_percentage=0.2):
    '''
    Split the data keys to be used for training, validation and testing. The test set should include data from subjects
    that are not included in the training or validation sets.

    :param keys: all data keys (subject, session)
    :param test_percentage: percentage of subjects for the test set
    :param val_percentage: percentage of the training keys for the validation set
    :return: the keys of each set (train_keys, val_keys, test_keys)
    '''

    subjects = list(set(k[0] for k in keys))  # all subjects
    n_sub = len(subjects)  # number of subjects
    n_sub_test = int(n_sub*test_percentage)  # number of subjects for the test set
    test_subjects = random.sample(subjects, n_sub_test)
    print(f"Test subjects: {test_subjects}")

    test_keys = [k for k in keys if k[0] in test_subjects]
    remaining_keys = [k for k in keys if k[0] not in test_subjects]

    n_val_keys = int(len(remaining_keys)*val_percentage)
    val_keys = random.sample(remaining_keys, n_val_keys)

    train_keys = [k for k in remaining_keys if k not in val_keys]

    return train_keys, val_keys, test_keys


def save_segments(segment_dict, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, segment in segment_dict.items():
        subject, session, segment_num = key  # Assuming key is (subject, session, segment_num)
        filename = os.path.join(directory, f"subject{subject}_session{session}_segment{segment_num}.npy")
        np.save(filename, segment)


def main():
    # Step 1: Load the data
    signals, peaks = load_ecg_data()

    # Step 2: Inspect the data
    inspect_data(signals)
    inspect_peaks(peaks)
    print(signals[(1, 1)])
    print(peaks[(1, 1)])

    # Step 3: Preprocess each signal (resample to 360 Hz)
    target_sampling_rate = 360
    for key in signals:
        signals[key] = resampling(signals[key], target_sampling_rate, original_sampling_rate=1440)
    peaks = resample_peaks_idx(peaks, target_sampling_rate, original_sampling_rate=1440)
    peaks_binary = peaks_to_binary(peaks, signals)
    print(f"peaks_resamp: {peaks[(1, 1)]}")
    print(f"peaks_bin: {peaks_binary[(1, 1)]}")

    # Step 4: Split data keys into train, test, and validation sets
    train_keys, val_keys, test_keys = split_keys(list(signals.keys()))
    print(f"train_keys: {train_keys}")
    print(f"test_keys: {test_keys}")
    print(f"val_keys: {val_keys}")

    train_signals = {key: signals[key] for key in train_keys}
    train_peaks_idx = {key: peaks[key] for key in train_keys}
    train_peaks_binary = {key: peaks_binary[key] for key in train_keys}

    val_signals = {key: signals[key] for key in val_keys}
    val_peaks_idx = {key: peaks[key] for key in val_keys}
    val_peaks_binary = {key: peaks_binary[key] for key in val_keys}

    test_signals = {key: signals[key] for key in test_keys}
    test_peaks_idx = {key: peaks[key] for key in test_keys}
    test_peaks_binary = {key: peaks_binary[key] for key in test_keys}

    # Step 5: Split each signal into segments of 10-40 seconds
    train_segs, train_peaks_binary_segs, train_peaks_idx_segs = split_signals(train_signals, train_peaks_binary,
                                                                              train_peaks_idx, target_sampling_rate,
                                                                              segment_duration_min=10,
                                                                              segment_duration_max=30)
    val_segs, val_peaks_binary_segs, val_peaks_idx_segs = split_signals(val_signals, val_peaks_binary, val_peaks_idx,
                                                                        target_sampling_rate, segment_duration_min=10,
                                                                        segment_duration_max=30)
    test_segs, test_peaks_binary_segs, test_peaks_idx_segs = split_signals(test_signals, test_peaks_binary,
                                                                           test_peaks_idx, target_sampling_rate,
                                                                           segment_duration_min=10,
                                                                           segment_duration_max=30)

    # Step 6: Normalize the ECG signals
    train_segs = {key: pp.minmax_scale(signal) for key, signal in train_segs.items()}
    val_segs = {key: pp.minmax_scale(signal) for key, signal in val_segs.items()}
    test_segs = {key: pp.minmax_scale(signal) for key, signal in test_segs.items()}

    # Step 7: Save datasets
    save_segments(train_segs, os.path.join(X, 'train'))
    save_segments(val_segs, os.path.join(X, 'val'))
    save_segments(test_segs, os.path.join(X, 'test'))

    save_segments(train_peaks_binary_segs, os.path.join(Y_BIN, 'train'))
    save_segments(val_peaks_binary_segs, os.path.join(Y_BIN, 'val'))
    save_segments(test_peaks_binary_segs,  os.path.join(Y_BIN, 'test'))

    save_segments(train_peaks_idx_segs, os.path.join(Y_IDX, 'train'))
    save_segments(val_peaks_idx_segs, os.path.join(Y_IDX, 'val'))
    save_segments(test_peaks_idx_segs, os.path.join(Y_IDX, 'test'))


if __name__ == "__main__":
    # run_ = ['Turing', 'laptop']
    # run = run_[1]  # change here. save files separated as validation, train and test sets only if run == 'machine'
    main()

