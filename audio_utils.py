from __future__ import division
from builtins import range
from past.utils import old_div
import numpy
from numpy import float32
import librosa

neg80point8db = 0.00009120108393559096
bit_depth = 16
default_silence_threshold = (neg80point8db * (2 ** (bit_depth - 1))) * 4


def root_mean_square(data):
    """Return the root mean square of a one-dimensional numpy array."""
    return float(numpy.sqrt(numpy.mean(numpy.square(data))))


def split_into(data, num):
    """Syntactic sugar for splitting an array into a fixed number of parts."""
    return numpy.array_split(data, num)


def poorly_estimate_fundamental(y, sample_rate):
    """
    Like it says on the tin, this method returns a rough estimate of the
    fundamental pitch in the given signal over time, along with its stddev.
    """
    pitches, magnitudes = librosa.core.piptrack(
        y=y, sr=sample_rate, fmin=10, fmax=1600)
    fundamental_over_time = [
        pitches[magnitudes[:, t].argmax(), t]
        for t in range(pitches.shape[1])
    ]

    return float(numpy.amin(fundamental_over_time)), \
        float(numpy.std(fundamental_over_time))


def average_eq_bands(y, num_bands=15):
    """
    Returns the average power in each EQ band, where the spectrum is split
    into `num_bands` equal bands.
    """
    frequency_spectrogram_data = librosa.amplitude_to_db(
        librosa.magphase(librosa.stft(y, num_bands + 1))[0], ref=numpy.max)
    return list([float(x) for x in numpy.mean(
        frequency_spectrogram_data, axis=1)])


def eq_vector(data, bands=3, num_windows=100):
    frequency_spectrogram_data = librosa.stft(data, bands + 1)
    window_size = old_div(frequency_spectrogram_data.shape[1], num_windows)
    for i in range(num_windows):
        start = i * window_size
        end = min(len(frequency_spectrogram_data), start + window_size)
        yield list([float(x) for x in numpy.mean(
            frequency_spectrogram_data[:, start:end], axis=1).astype(float32)])


def loudness_at(data, pos, window_size=100):
    window_start = max(0, pos - old_div(window_size, 2))
    window_end = min(len(data), window_start + window_size)
    if (window_end - window_start) < window_size:
        window_start = max(0, window_end - window_size)
    windowed = data[window_start:window_end]
    return root_mean_square(windowed)


def loudness_of(data):
    """Syntactic sugar for root mean square."""
    return root_mean_square(data)


def loudness_vector(data, num_windows=100):
    window_size = old_div(len(data), num_windows)
    for i in range(num_windows):
        start = i * window_size
        end = min(len(data), start + window_size)
        yield root_mean_square(data[start:end])


def normalized(list):
    """Given an audio buffer, return it with the loudest value scaled to 1.0"""
    return list.astype(numpy.float32) / float(numpy.amax(numpy.abs(list)))


def start_of(list, threshold=default_silence_threshold, samples_before=1):
    """Estimate where the start of a given signal is, in samples."""
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (bit_depth - 1))
    index = numpy.argmax(numpy.absolute(list) > threshold)
    if index > (samples_before - 1):
        return index - samples_before
    else:
        return 0


def end_of(list, threshold=default_silence_threshold, samples_after=1):
    """Estimate where the end of a given signal is, in samples."""
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (bit_depth - 1))
    rev_index = numpy.argmax(
        numpy.flipud(numpy.absolute(list)) > threshold
    )
    if rev_index > (samples_after - 1):
        return len(list) - (rev_index - samples_after)
    else:
        return len(list)


def trim_data(
    data,
    start_threshold=default_silence_threshold,
    end_threshold=default_silence_threshold
):
    """Returns a trimmed signal based on its estimated start and end."""
    start = start_of(data, start_threshold)
    end = end_of(data, end_threshold)

    return data[start:end]
