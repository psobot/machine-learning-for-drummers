#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy

from sklearn.tree import export_graphviz


def evaluate_model(
    model,
    features,
    classes,
    sample_names,
    class_names,
    output=True
):
    total_correct = 0.0
    for i, (feature, _class) in \
            enumerate(zip(features, classes)):

        feature = numpy.array(feature).reshape(1, -1)
        predicted_class = model.predict(feature)[0]

        accurate = predicted_class == _class
        accuracy_string = u"✅" if accurate else u"❌"
        if accurate:
            total_correct += 1.
            if output:
                print(
                    "\t%s Predicted %s as %s." % (
                        accuracy_string,
                        sample_names[i],
                        class_names[predicted_class]
                    )
                )
        else:
            if output:
                print(
                    "\t%s Predicted %s as %s, was actually %s." % (
                        accuracy_string,
                        sample_names[i],
                        class_names[predicted_class],
                        class_names[_class]
                    )
                )
    print("Total of %d correct of %d. (%2.2f%%)" % (
        total_correct,
        len(features),
        100. * total_correct / len(features)))


def explain_model(model, feature_names, class_names):
    import graphviz
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("graph")
