from __future__ import print_function

# importing os gives us access to methods we need to manipulate paths
import os
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
    loudness_of, \
    poorly_estimate_fundamental, \
    average_eq_bands, \
    split_into


def load_and_trim(file):
    y, rate = librosa.load(file, mono=True)
    y = normalized(y)
    trimmed = trim_data(y)
    return trimmed, rate


def features_for(file):
    # Load and trim the audio file to only the portions that aren't silent.
    audio, rate = load_and_trim(file)

    features = {"duration": librosa.get_duration(audio, rate)}

    # Let's split up the audio file into chunks
    for (i, section) in enumerate(split_into(audio, 10)):
        # And in each of those chunks:
        # ...get the loudness for that chunk
        features["loudness_%d" % i] = loudness_of(section)

        # Use poorly_estimate_fundamental to figure out
        # what the rough pitch is, along with the standard
        # deviation - how much that pitch varies.
        fundamental, fundamental_stddev = \
            poorly_estimate_fundamental(section, rate)
        features["fundamental_%d" % i] = fundamental
        features["fundamental_stddev_%d" % i] = fundamental_stddev

        # ...make a feature out of each of 25 EQ bands.
        for (j, value) in enumerate(average_eq_bands(section, 99)):
            features["average_eq_%d_%d" % (i, j)] = value

    return features


def extract_features(data_dir="./data/"):
    audio_files = glob(os.path.join(data_dir, '**', '*'))
    features = {}
    for file in tqdm(audio_files):
        try:
            features[file] = features_for(file)
        except Exception as e:
            sys.stderr.write("Failed to run on %s: %s\n" % (file, e))
    return features


if __name__ == "__main__":
    import sys
    if sys.argv[-1].endswith('.py'):
        features = extract_features()
    else:
        features = extract_features(sys.argv[-1])

    print(json.dumps(features, indent=4, sort_keys=True))
