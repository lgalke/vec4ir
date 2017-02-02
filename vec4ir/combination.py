#!/usr/bin/env python3
# coding: utf-8
from sklearn.base import BaseEstimator
from collections import defaultdict, OrderedDict
from sklearn.preprocessing import maxabs_scale
from functools import reduce
from operator import itemgetter
from numpy import product


def aggregate_dicts(dicts, agg_fn=sum):
    """
    Aggregates the contents of two dictionaries by key
    @param agg_fn is used to aggregate the values (defaults to sum)
    >>> dict1 = {'a': 0.8, 'b': 0.4, 'd': 0.4}
    >>> dict2 = {'a': 0.7, 'c': 0.3, 'd': 0.3}
    >>> agg = aggregate_dicts([dict1, dict2])
    >>> OrderedDict(sorted(agg.items(), key=itemgetter(1), reverse=True))
    OrderedDict([('a', 1.5), ('d', 0.7), ('b', 0.4), ('c', 0.3)])
    """
    acc = defaultdict(list)
    for d in dicts:
        for k in d:
            acc[k].append(d[k])

    for key, values in acc.items():
        acc[key] = agg_fn(values)

    return dict(acc)  # no need to default to list anymore


def fuzzy_or(values):
    """
    Applies fuzzy-or to a list of values
    >>> fuzzy_or([0.5])
    0.5
    >>> fuzzy_or([0.5, 0.5])
    0.75
    >>> fuzzy_or([0.5, 0.5, 0.5])
    0.875
    """
    if min(values) < 0 or max(values) > 1:
        raise ValueError("fuzzy_or expects values in [0,1]")
    return reduce(lambda x, y: 1 - (1 - x) * (1 - y), values)


CombinCombinatorMixin


CombinatorMixin
