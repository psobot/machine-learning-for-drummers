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


def load_and_trim(file):
    y, rate = librosa.load(file, mono=True)
    y = normalized(y)
    trimmed = trim_data(y)
    return trimmed, rate


def features_for(file):
    # Load and trim the audio file to only the portions that aren't silent.
    audio, rate = load_and_trim(file)

    # Use poorly_estimate_fundamental to figure out what the rough pitch is,
    # along with the standard deviation - how much that pitch varies.
    fundamental, fundamental_stddev = poorly_estimate_fundamental(audio, rate)

    # Like an equalizer, find out how loud each "frequency band" is here.
    # In this case, we're just splitting up the audio spectrum into three
    # very wide sections, low, mid, and high.
    low, mid, high = average_eq_bands(audio, 3)

    features = {
        "duration":              librosa.get_duration(audio, rate),
        "start_loudness":        loudness_at(audio, 0),
        "mid_loudness":          loudness_at(audio, len(audio) / 2),
        "end_loudness":          loudness_at(audio, len(audio)),
        "fundamental_freq":      fundamental,
        "fundamental_deviation": fundamental_stddev,
        "average_eq_low":        low,
        "average_eq_mid":        mid,
        "average_eq_high":       high,
    }

    # No matter how many additional EQ bands we have, let's add them to
    # the features dict.
    features.update({
        "average_eq_%d" % i: value
        for (i, value)
        in enumerate(average_eq_bands(audio, 40))
    })

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
    import sys
    if sys.argv[-1].endswith('.py'):
        extract_features()
    else:
        extract_features(sys.argv[-1])
