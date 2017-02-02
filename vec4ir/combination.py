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


class CombinatorMixin(object):
    """ Creates a computational tree with retrieval models as leafs
    """
    def __get_weights(self, other):
        if not isinstance(other, CombinatorMixIn):
            raise ValueError("other is not Combinable")

        if hasattr(self, '__weight'):
            weight = self.__weight
        else:
            weight = 1.0

        if hasattr(other, '__weight'):
            otherweight = other.__weight
        else:
            otherweight = 1.0

        return weight, otherweight

    # This is evil since it can exceed [0,1], rescaling at the end would be not
    # that beautiful
    # def __add__(self, other):
    #     weights = self.__get_weights(other)
    #     return Combined([self, other], weights=weights, agg_fn=sum)

    def __and__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, agg_fn=product)

    def __or__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, agg_fn=fuzzy_or)

    def __mul__(self, scalar):
        self.__weight = scalar
        return self


class Combined(BaseEstimator, CombinatorMixin):
    def __init__(self, retrieval_models, weights=None, aggregation_fn=sum):
        self.retrieval_models = retrieval_models
        self.aggregation_fn = aggregation_fn
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(retrieval_models)

    def query(self, query, k=1, sort=True):
        models = self.retrieval_models
        weights = maxabs_scale(self.weights)  # max 1 does not crash [0,1]
        agg_fn = self.aggregation_fn
        # we only need to sort in the final run
        combined = [m.query(query, k=k, sort=False) for m in models]

        if weights is not None:
            combined = [{k: v * w for k, v in r.items()} for r, w in
                        zip(combined, weights)]

        combined = aggregate_dicts(combined, agg_fn=agg_fn, sort=True)

        if sort:
            # only cut-off at k if this is the final (sorted) output
            combined = OrderedDict(sorted(combined.items(), key=itemgetter(1),
                                          reverse=True)[:k])
        return combined
