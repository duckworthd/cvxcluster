import  numpy as np
cimport numpy as np
cimport cython

from _problem cimport project


@cython.boundscheck(False)
@cython.wraparound(False)
def coordinate_ascent_iteration(
    np.ndarray[np.float_t, ndim=2] X,
    np.ndarray[np.float_t, ndim=2] Delta,
    np.ndarray[np.float_t, ndim=2] w,
    np.ndarray[np.float_t, ndim=2] lmbd,
    double       gamma,
    double       p,
  ):
  # Perform one iteration of coordinate descent. Will modify Delta and lmbd
  # directly.

  cdef int n_pairs = w.shape[0]
  cdef int i, j, l
  cdef double w_l
  cdef np.ndarray[np.float_t, ndim=1] l_start, l_mid, l_new

  for l in range(n_pairs):
    i   = int(w[l,0])
    j   = int(w[l,1])
    w_l =     w[l,2]

    l_start    = lmbd[l,:]
    Delta[i]  -= l_start
    Delta[j]  += l_start
    l_mid      = -0.5 * (Delta[i] - Delta[j] + X[i] - X[j])
    l_new      = project(l_mid, gamma * w_l, p)
    lmbd[l]    = l_new
    Delta[i]  += l_new
    Delta[j]  -= l_new

  return lmbd
