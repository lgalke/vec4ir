from numpy import array

import scipy.sparse as sp

from vec4ir.base import match_bool_or



def test_matching():

    X = array([[0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]])
    assert (match_bool_or(X, array([[0,0,0]])) == array([])).all()
    assert (match_bool_or(X, array([[0,0,1]])) == array([0, 2, 4])).all()
    assert (match_bool_or(X, array([[0,1,0]])) == array([1, 2, 5])).all()
    assert (match_bool_or(X, array([[0,1,1]])) == array([0, 1, 2, 4, 5])).all()
    assert (match_bool_or(X, array([[1,0,0]])) == array([3, 4, 5])).all()
    assert (match_bool_or(X, array([[1,0,1]])) == array([0, 2, 3, 4, 5])).all()
    assert (match_bool_or(X, array([[1,1,0]])) == array([1, 2, 3, 4, 5])).all()
    assert (match_bool_or(X, array([[1,1,1]])) == array([0, 1, 2, 3, 4, 5])).all()

def test_matching_sparse():

    X = sp.csr_matrix(array([[0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]]))
    assert (match_bool_or(X, sp.csr_matrix(array([[0,0,0]]))) == array([])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[0,0,1]]))) == array([0, 2, 4])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[0,1,0]]))) == array([1, 2, 5])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[0,1,1]]))) == array([0, 1, 2, 4, 5])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[1,0,0]]))) == array([3, 4, 5])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[1,0,1]]))) == array([0, 2, 3, 4, 5])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[1,1,0]]))) == array([1, 2, 3, 4, 5])).all()
    assert (match_bool_or(X, sp.csr_matrix(array([[1,1,1]]))) == array([0, 1, 2, 3, 4, 5])).all()
