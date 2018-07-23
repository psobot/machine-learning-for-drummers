import numpy
from numpy import float32
import librosa

neg80point8db = 0.00009120108393559096
bit_depth = 16
default_silence_threshold = (neg80point8db * (2 ** (bit_depth - 1))) * 4


def root_mean_square(data):
    return float(numpy.sqrt(numpy.mean(numpy.square(data))))


def split_into(data, num):
    return numpy.array_split(data, num)


def poorly_estimate_fundamental(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=10, fmax=1600)
    fundamental_over_time = [
        pitches[magnitudes[:, t].argmax(), t]
        for t in xrange(pitches.shape[1])
    ]

    return float(numpy.amin(fundamental_over_time)), \
        float(numpy.std(fundamental_over_time))


def average_eq_bands(y, bands=15):
    frequency_spectrogram_data = librosa.amplitude_to_db(
        librosa.magphase(librosa.stft(y, bands + 1))[0], ref=numpy.max)
    return list([float(x) for x in numpy.mean(
        frequency_spectrogram_data, axis=1)])


def eq_vector(data, bands=3, num_windows=100):
    frequency_spectrogram_data = librosa.stft(data, bands + 1)
    window_size = frequency_spectrogram_data.shape[1] / num_windows
    for i in xrange(num_windows):
        start = i * window_size
        end = min(len(frequency_spectrogram_data), start + window_size)
        yield list([float(x) for x in numpy.mean(
            frequency_spectrogram_data[:, start:end], axis=1).astype(float32)])


def loudness_at(data, pos, window_size=100):
    window_start = max(0, pos - window_size / 2)
    window_end = min(len(data), window_start + window_size)
    if (window_end - window_start) < window_size:
        window_start = max(0, window_end - window_size)
    windowed = data[window_start:window_end]
    return root_mean_square(windowed)


def loudness_of(data):
    return root_mean_square(data)


def loudness_vector(data, num_windows=100):
    window_size = len(data) / num_windows
    for i in xrange(num_windows):
        start = i * window_size
        end = min(len(data), start + window_size)
        yield root_mean_square(data[start:end])


def normalized(list):
    """Given an audio buffer, return it with the loudest value scaled to 1.0"""
    return list.astype(numpy.float32) / float(numpy.amax(numpy.abs(list)))


def start_of(list, threshold=default_silence_threshold, samples_before=1):
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (bit_depth - 1))
    index = numpy.argmax(numpy.absolute(list) > threshold)
    if index > (samples_before - 1):
        return index - samples_before
    else:
        return 0


def end_of(list, threshold=default_silence_threshold, samples_after=1):
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
    start = start_of(data, start_threshold)
    end = end_of(data, end_threshold)

    return data[start:end]
