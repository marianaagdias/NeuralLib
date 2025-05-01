from scipy.signal import resample


def resampling(signal, target_sampling_rate, original_sampling_rate=1440):
    num_samples = int(len(signal) * (target_sampling_rate / original_sampling_rate))
    resampled_signal = resample(signal, num_samples)
    print(f"Resampled signal to {num_samples} samples")
    return resampled_signal


def split_signal(signal, sampling_rate, segment_duration):
    """
    Splits the signal into smaller segments.

    :param signal: The original signal to split.
    :param sampling_rate: The sampling rate of the signal (in Hz).
    :param segment_duration: The duration of each segment in seconds.
    :return: A list of segments.
    """
    segment_length = sampling_rate * segment_duration  # number of samples per segment
    num_segments = len(signal) // segment_length  # how many full segments we can get

    segments = []
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segments.append(signal[start_idx:end_idx])

    # Here, I'm ignoring any remaining samples that don't fit into a full segment - change?

    return segments
