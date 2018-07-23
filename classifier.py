#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is to give us a bit more compatibility with Python 3.
from __future__ import print_function

# Here we import a bunch of utility functions that have little to do with
# the machine learning at hand, but are useful for parsing our features from
# the JSON format we wrote them in and evaluating the resulting machine
# learning model that we've created.
from json_utils import read_data
from model_utils import evaluate_model

# Here we import DecisionTreeClassifier, which is the machine learning
# algorithm we'll be using to create our model.
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_model():
    # First, let's read all of the features that we got from feature_extract.
    # Fun fact: you could do ./feature_extract.py | ./classifier.py to execute
    # both the feature extraction and classification steps at once, without
    # writing the results to JSON first. Very handy for iterating on features.
    features, classes, sample_names, feature_names, class_names = read_data()

    # We'll use this percentage of the data to train, and the rest for testing.
    # Why not just train on all the data? That would result in a model that is
    # overfitted, or overly good at the data that it's seen and does poorly
    # with data that it hasn't seen.
    training_percentage = 0.75
    num_training_samples = int(len(features) * training_percentage)

    # Here we separate all of our features and classes into just the ones
    # we want to train on...
    training_features, training_classes = \
        features[:num_training_samples], classes[:num_training_samples]

    # ...and we do the training, which creates our model!
    # vvv MACHINE LEARNING HAPPENS ON THIS LINE BELOW vvv
    model = DecisionTreeClassifier(random_state=2).fit(
        training_features, training_classes)
    # ^^^ MACHINE LEARNING HAPPENS ON THIS LINE ABOVE ^^^

    # These two lines write out a .pdf file of the model's decision tree.
    # It's useful if you want to explain the model, but requires
    # you to have Graphviz installed, so I've left it commented out.
    # from model_utils import explain_model
    # explain_model(model, feature_names, class_names)

    print("Evaluating training accuracy...")
    evaluate_model(
        model,
        training_features,
        training_classes,
        sample_names[:num_training_samples],
        class_names,
        output=False
    )

    # Now, here we take the other portion of our input data and use
    # that to test the model and ensure it performs well on data it
    # hasn't seen before.
    num_test_samples = len(features) - num_training_samples
    test_features, test_classes = \
        features[-num_test_samples:], classes[-num_test_samples:]

    print("Evaluating test accuracy...")
    evaluate_model(
        model,
        test_features,
        test_classes,
        sample_names[-num_test_samples:],
        class_names
    )


if __name__ == "__main__":
    train_and_evaluate_model()
