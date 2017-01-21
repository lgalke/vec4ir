#!/usr/bin/env python3
# coding: utf-8

def analogy2query(analogy):
    """ Decompose analogy string of n words into n-1 query words and 1 target word (last one)
    zips with +-
    >>> analogy2query("Athens Greece Baghdad Iraq")
    ('+Athens -Greece +Baghdad', 'Iraq')
    """
    words = analogy.split()
    terms, target = words[:-1], words[-1]
    pm_terms = [("+" if idx % 2 == 0 else "-") + term
                for idx, term in enumerate(terms)]
    query = " ".join(pm_terms)
    return query, target


def parse_analogy_file(fh):
    X, Y = [], []
    for line in fh:
        if line.startswith(":"):
            continue
        query, target = analogy2query(line.strip())
        X.append(query)
        Y.append(target)

    return X, Y


if __name__ == "__main__":
    import doctest
    doctest.testmod()

