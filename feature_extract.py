from __future__ import print_function

# importing os gives us access to methods we need to manipulate paths
import os
import sys
import json

# librosa is a great all-around audio handling library in Python.
# here we'll be using it to extract features like length, loudness,
# and frequency content.
import librosa

# tqdm is a progress bar helper that will show us how quickly
# our code is running.
from tqdm import tqdm

# glob lets us find files by using wildcards, like "*"
from glob import glob

from audio_utils import normalized, \
    trim_data, \
    loudness_at, \
    poorly_estimate_fundamental, \
    average_eq_bands


def features_for(file):
    y, rate = librosa.load(file, mono=True)
    y = normalized(y)
    trimmed = trim_data(y)

    loudness_window_in_seconds = 0.10
    loudness_window_in_samples = int(rate * loudness_window_in_seconds)

    fundamental, fundamental_stddev = poorly_estimate_fundamental(y, rate)

    # No matter how many EQ bands we have, let's add them to the features dict.
    eq_bands = {
        "average_eq_%d" % i: value
        for (i, value)
        in enumerate(average_eq_bands(y, 3))
    }

    features = {
        "duration": librosa.get_duration(trimmed, rate),
        "start_loudness": loudness_at(
            trimmed, 0, loudness_window_in_samples),
        "mid_loudness": loudness_at(
            trimmed, len(trimmed) / 2, loudness_window_in_samples),
        "end_loudness": loudness_at(
            trimmed,
            len(trimmed) - loudness_window_in_samples / 2,
            loudness_window_in_samples),
        "fundamental_in_hertz": fundamental,
        "fundamental_deviation": fundamental_stddev
    }

    features.update(eq_bands)

    return features


def extract_features(data_dir="./data/"):
    audio_files = glob(os.path.join(data_dir, '**', '*'))
    features = {}
    for file in tqdm(audio_files):
        try:
            features[file] = features_for(file)
        except Exception as e:
            sys.stderr.write("Failed to run on %s: %s\n" % (file, e))
    print(json.dumps(features, indent=4, sort_keys=True))

if __name__ == "__main__":
    extract_features()
