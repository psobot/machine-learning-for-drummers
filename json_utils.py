import os
import sys
import json
import math
import random


STDIN_CACHED_DATA = None


def read_data():
    filename = sys.argv[-1]
    if sys.stdin.isatty():
        if not filename.endswith('json'):
            filename = './features.json'
        data = json.load(open(filename))
    else:
        global STDIN_CACHED_DATA
        if not STDIN_CACHED_DATA:
            STDIN_CACHED_DATA = json.load(sys.stdin)
        data = STDIN_CACHED_DATA

    feature_names = sorted(list(data.values())[0])

    class_names = set()
    for path, features_dict in data.items():
        # Take the folder name of the sample as the class
        class_names.add(path.split(os.sep)[-2])
    class_names = sorted(class_names)

    sample_names = []
    features = []
    classes = []

    examples = list(data.items())
    random.shuffle(examples)

    for path, features_dict in examples:
        sample_names.append(path.split(os.sep)[-1])

        # Add the index of this class in class_names to the output list,
        # as Scikit requires classes to be numbers
        classes.append(class_names.index(path.split(os.sep)[-2]))

        feature_vector = []
        for feature_key in feature_names:
            feature_value = features_dict[feature_key]
            # Scikit doesn't like infinite values, so let's replace them.
            if feature_value > 1000000.:
                feature_value = 1000000.
            if feature_value < -1000000.:
                feature_value = -1000000.
            if math.isnan(feature_value):
                feature_value = 0
            feature_vector.append(feature_value)

        features.append(feature_vector)
    return features, classes, sample_names, feature_names, class_names
