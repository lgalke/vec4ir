import numpy as np
from numpy import array
from vec4ir.utils import argtopk

def test_argtopk():
    A = np.asarray([5,4,3,6,7,8,9,0])

    assert (A[argtopk(A, 3)] == array([9, 8, 7])).all()
    assert (argtopk(A, 1) == array([6])).all()
    assert (argtopk(A, 6) == array([6, 5, 4, 3, 0, 1])).all()
    assert (argtopk(A, 10) == array([6, 5, 4, 3, 0, 1, 2, 7])).all()
    assert (argtopk(A, 28) == array([6, 5, 4, 3, 0, 1, 2, 7])).all()
    assert (argtopk(A, None) == array([6, 5, 4, 3, 0, 1, 2, 7])).all()
    # Inverted A should reverse result
    assert (argtopk(-A, None) == array([6, 5, 4, 3, 0, 1, 2, 7])[::-1]).all()

    X = np.arange(20)
    assert (argtopk(X, 10) == array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10])).all()
    assert (argtopk(X, -5) == array([0, 1, 2, 3, 4])).all()
