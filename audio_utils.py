import numpy
import librosa

neg80point8db = 0.00009120108393559096
bit_depth = 16
default_silence_threshold = (neg80point8db * (2 ** (bit_depth - 1))) * 4


def root_mean_square(data):
    return float(numpy.sqrt(numpy.mean(numpy.square(data))))


def poorly_estimate_fundamental(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=10, fmax=1600)
    fundamental_over_time = [
        pitches[magnitudes[:, t].argmax(), t]
        for t in xrange(pitches.shape[1])
    ]

    return float(numpy.amin(fundamental_over_time)), \
        float(numpy.std(fundamental_over_time))


def average_eq_bands(y, bands=3):
    frequency_spectrogram_data = librosa.stft(y, bands + 1)
    return list([float(x) for x in numpy.mean(
        frequency_spectrogram_data, axis=1).astype(numpy.float32)])


def loudness_at(data, pos, window=40):
    window_start = max(0, pos - window / 2)
    window_end = min(len(data), window_start + window)
    windowed = data[window_start:window_end]
    return root_mean_square(windowed)


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
